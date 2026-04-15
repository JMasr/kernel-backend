"""Chunk validation, concatenation, and cleanup for the chunked signing pipeline."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from kernel_backend.core.domain.chunk import (
    ChunkManifest,
    ChunkResult,
    ChunkValidation,
)

# Tolerance windows (seconds) for probe-vs-expected duration drift. FFmpeg's
# trim with -c copy can round to the closest keyframe, so a small slack is
# expected even for a valid chunk.
_PER_CHUNK_DURATION_TOL_S = 0.5
_TOTAL_DURATION_TOL_S = 1.0


def validate_chunks(
    manifest: ChunkManifest,
    results: list[ChunkResult] | tuple[ChunkResult, ...],
    media_probe_fn: Callable[[Path], object],
) -> ChunkValidation:
    """Run every pre-concat safety check; accumulate errors, never raise.

    ``media_probe_fn`` is a callable returning an object with a
    ``duration_s`` attribute (production: ``MediaService.probe``). Stays as
    a callable so tests can stub it without pulling MediaService in.
    """
    errors: list[str] = []
    expected_duration_s = sum(c.expected_payload_duration_s for c in manifest.chunks)

    # 1. Cardinality
    all_present = len(results) == manifest.total_chunks
    if not all_present:
        errors.append(
            f"expected {manifest.total_chunks} chunk results, got {len(results)}"
        )

    # 2. All succeeded
    for r in results:
        if not r.success:
            errors.append(
                f"chunk {r.chunk_id} failed: {r.error or 'unknown error'}"
            )

    # 3. chunk_id set is exactly {0, .., N-1}
    seen_ids: dict[int, int] = {}
    for r in results:
        seen_ids[r.chunk_id] = seen_ids.get(r.chunk_id, 0) + 1
    duplicates = [cid for cid, count in seen_ids.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate chunk_ids: {sorted(duplicates)}")
    missing_ids = [
        cid for cid in range(manifest.total_chunks) if cid not in seen_ids
    ]
    if missing_ids:
        all_present = False
        errors.append(f"missing chunk_ids: {missing_ids}")

    # 4 + 5. Files exist, non-empty, durations within tolerance.
    expected_by_id = {c.chunk_id: c.expected_payload_duration_s for c in manifest.chunks}
    total_actual_s = 0.0
    all_durations_valid = True
    for r in results:
        if not r.success:
            all_durations_valid = False
            continue
        path = Path(r.output_path)
        if not path.exists():
            errors.append(f"chunk {r.chunk_id}: output missing at {path}")
            all_durations_valid = False
            continue
        if path.stat().st_size == 0:
            errors.append(f"chunk {r.chunk_id}: output at {path} is empty")
            all_durations_valid = False
            continue

        try:
            probe = media_probe_fn(path)
            actual_s = float(getattr(probe, "duration_s"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"chunk {r.chunk_id}: probe failed: {exc}")
            all_durations_valid = False
            continue

        expected_s = expected_by_id.get(r.chunk_id, 0.0)
        if abs(actual_s - expected_s) > _PER_CHUNK_DURATION_TOL_S:
            errors.append(
                f"chunk {r.chunk_id}: duration {actual_s:.3f}s "
                f"drifted from expected {expected_s:.3f}s"
            )
            all_durations_valid = False
        total_actual_s += actual_s

    # 6. Sum-of-durations check
    if abs(total_actual_s - expected_duration_s) > _TOTAL_DURATION_TOL_S:
        errors.append(
            f"total duration {total_actual_s:.3f}s drifted from "
            f"expected {expected_duration_s:.3f}s"
        )
        all_durations_valid = False

    return ChunkValidation(
        all_present=all_present,
        all_durations_valid=all_durations_valid,
        total_duration_s=total_actual_s,
        expected_duration_s=expected_duration_s,
        errors=tuple(errors),
    )


def concatenate_chunks(
    results: list[ChunkResult] | tuple[ChunkResult, ...],
    output_path: Path,
) -> None:
    """Concatenate chunks in ``chunk_id`` order via the ffmpeg concat demuxer.

    Uses stream copy — no re-encode. Raises ``ValueError`` on a non-zero
    ffmpeg return code. Does not delete the input chunks; the caller owns
    cleanup.
    """
    ordered = sorted(results, key=lambda r: r.chunk_id)
    if not ordered:
        raise ValueError("no chunks to concatenate")

    list_fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="chunk_concat_")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as fh:
            for r in ordered:
                abspath = str(Path(r.output_path).resolve())
                # ffmpeg concat demuxer: escape single quotes by splitting.
                escaped = abspath.replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-loglevel", "quiet",
            str(output_path),
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace")[:500]
            raise ValueError(
                f"ffmpeg concat failed (returncode={proc.returncode}): {detail}"
            )
    finally:
        try:
            Path(list_path).unlink()
        except FileNotFoundError:
            pass


def cleanup_chunks(
    results: list[ChunkResult] | tuple[ChunkResult, ...],
) -> None:
    """Best-effort removal of per-chunk temp files. Never raises."""
    for r in results:
        if not r.output_path:
            continue
        try:
            Path(r.output_path).unlink(missing_ok=True)
        except Exception:
            # Cleanup is best-effort; swallow EACCES/EBUSY/etc.
            pass
