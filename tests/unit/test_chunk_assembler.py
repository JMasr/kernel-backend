"""Unit tests for chunk_assembler — validate_chunks + concatenate_chunks + cleanup."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_backend.core.domain.chunk import (
    ChunkManifest,
    ChunkResult,
    ChunkSpec,
)
from kernel_backend.core.services.chunk_assembler import (
    cleanup_chunks,
    concatenate_chunks,
    validate_chunks,
)


def _manifest(tmp_path: Path, chunk_count: int = 4, seg_per_chunk: int = 15):
    chunks: list[ChunkSpec] = []
    cursor = 0
    for i in range(chunk_count):
        end = cursor + seg_per_chunk
        lead = 0 if i == 0 else 1
        trail = 0 if i == chunk_count - 1 else 1
        chunks.append(
            ChunkSpec(
                chunk_id=i,
                payload_seg_start=cursor,
                payload_seg_end=end,
                guard_lead_segments=lead,
                guard_trail_segments=trail,
                decode_start_s=max(0.0, (cursor - lead) * 5.0),
                decode_end_s=(end + trail) * 5.0,
                payload_start_s=lead * 5.0,
                expected_payload_duration_s=seg_per_chunk * 5.0,
            )
        )
        cursor = end
    return ChunkManifest(
        total_chunks=chunk_count,
        total_payload_segments=cursor,
        segment_duration_s=5.0,
        guard_segments=1,
        chunks=tuple(chunks),
    )


def _result(chunk_id: int, tmp_path: Path, size_bytes: int = 1024) -> ChunkResult:
    path = tmp_path / f"chunk_{chunk_id:04d}.mp4"
    path.write_bytes(b"x" * size_bytes)
    return ChunkResult(
        chunk_id=chunk_id,
        output_path=str(path),
        n_segments_processed=15,
        success=True,
    )


def _probe_with_duration(duration_s: float):
    def probe(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(duration_s=duration_s)

    return probe


def test_validate_all_ok(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, chunk_count=4, seg_per_chunk=15)
    results = [_result(i, tmp_path) for i in range(4)]

    validation = validate_chunks(manifest, results, _probe_with_duration(75.0))
    assert validation.is_valid is True
    assert validation.errors == ()
    assert validation.total_duration_s == pytest.approx(300.0)
    assert validation.expected_duration_s == pytest.approx(300.0)


def test_validate_missing_chunk(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, chunk_count=4, seg_per_chunk=15)
    # Drop chunk_id=2 from results entirely
    results = [_result(i, tmp_path) for i in (0, 1, 3)]

    validation = validate_chunks(manifest, results, _probe_with_duration(75.0))
    assert validation.is_valid is False
    assert any("missing" in e for e in validation.errors)
    assert validation.all_present is False


def test_validate_failed_chunk(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, chunk_count=2, seg_per_chunk=15)
    results = [
        _result(0, tmp_path),
        ChunkResult(
            chunk_id=1,
            output_path="",
            n_segments_processed=0,
            success=False,
            error="encoder exploded",
        ),
    ]

    validation = validate_chunks(manifest, results, _probe_with_duration(75.0))
    assert validation.is_valid is False
    assert any("encoder exploded" in e for e in validation.errors)


def test_validate_duration_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, chunk_count=2, seg_per_chunk=15)
    results = [_result(i, tmp_path) for i in range(2)]

    # Each chunk should be 75s; probe reports 3s for both → huge drift.
    validation = validate_chunks(manifest, results, _probe_with_duration(3.0))
    assert validation.is_valid is False
    assert any("drifted" in e for e in validation.errors)
    assert validation.all_durations_valid is False


def test_validate_duplicate_ids(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, chunk_count=2, seg_per_chunk=15)
    results = [_result(0, tmp_path), _result(1, tmp_path), _result(1, tmp_path)]

    validation = validate_chunks(manifest, results, _probe_with_duration(75.0))
    assert validation.is_valid is False
    assert any("duplicate" in e for e in validation.errors)


def test_validate_empty_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, chunk_count=1, seg_per_chunk=15)
    zero_path = tmp_path / "chunk_0000.mp4"
    zero_path.write_bytes(b"")
    results = [
        ChunkResult(
            chunk_id=0,
            output_path=str(zero_path),
            n_segments_processed=15,
            success=True,
        )
    ]

    validation = validate_chunks(manifest, results, _probe_with_duration(75.0))
    assert validation.is_valid is False
    assert any("empty" in e for e in validation.errors)


def test_concatenate_creates_file(monkeypatch, tmp_path: Path) -> None:
    # Stub subprocess.run; verify that ffmpeg is invoked with a concat list
    # in chunk_id order.
    results = [_result(i, tmp_path) for i in (2, 0, 1)]  # intentionally unsorted
    captured: dict = {}

    def fake_run(cmd, stdout=None, stderr=None):
        captured["cmd"] = cmd
        # Read the list file (the -i argument) and stash its contents.
        idx = cmd.index("-i") + 1
        captured["list_path"] = cmd[idx]
        captured["list_contents"] = Path(cmd[idx]).read_text(encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_path = tmp_path / "final.mp4"
    concatenate_chunks(results, out_path)

    assert captured["cmd"][0] == "ffmpeg"
    assert "-f" in captured["cmd"] and "concat" in captured["cmd"]
    assert "-safe" in captured["cmd"] and "0" in captured["cmd"]
    assert "-c" in captured["cmd"] and "copy" in captured["cmd"]
    lines = [
        line for line in captured["list_contents"].splitlines() if line.strip()
    ]
    assert len(lines) == 3
    # Lines must be in chunk_id order: 0, 1, 2
    for expected_id, line in enumerate(lines):
        assert f"chunk_{expected_id:04d}.mp4" in line
    # Concat list is cleaned up after run
    assert not Path(captured["list_path"]).exists()


def test_concatenate_raises_on_ffmpeg_error(monkeypatch, tmp_path: Path) -> None:
    results = [_result(0, tmp_path)]

    def fake_run(cmd, stdout=None, stderr=None):
        return SimpleNamespace(returncode=1, stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="concat failed"):
        concatenate_chunks(results, tmp_path / "out.mp4")


def test_cleanup_removes_files(tmp_path: Path) -> None:
    results = [_result(i, tmp_path) for i in range(3)]
    paths = [Path(r.output_path) for r in results]
    for p in paths:
        assert p.exists()
    cleanup_chunks(results)
    for p in paths:
        assert not p.exists()


def test_cleanup_tolerates_missing(tmp_path: Path) -> None:
    results = [
        ChunkResult(
            chunk_id=0,
            output_path=str(tmp_path / "does_not_exist.mp4"),
            n_segments_processed=0,
            success=False,
        )
    ]
    # Must not raise
    cleanup_chunks(results)
