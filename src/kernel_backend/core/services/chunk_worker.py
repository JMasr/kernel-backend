"""Per-chunk signing worker — runs inside a ``ProcessPoolExecutor``.

``process_video_chunk`` decodes a time slice of the source video, embeds the
Reed-Solomon symbol on payload segments (and passes guard segments through
untouched), encodes to an intermediate MP4, then stream-copy trims to the
payload boundary. The function is sync, picklable, and never raises: all
failures are folded into a :class:`ChunkResult` with ``success=False``.

This module is the one place in ``core/`` that reaches into ``infrastructure``
directly, because the worker body runs in a subprocess where MediaService is
the concrete driver for FFmpeg. The alternative (a port abstraction) would
only add pickling and DI complexity without buying anything.
"""
from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from kernel_backend.core.domain.chunk import ChunkResult, ChunkSpec
from kernel_backend.core.domain.watermark import (
    VideoEmbeddingParams,
    embedding_params_from_dict,
)
from kernel_backend.engine.video.wid_watermark import (
    embed_video_frame_yuvj420_planes,
    frame_to_yuvj420_planes,
)


def _rehydrate_spec(spec_dict: dict) -> ChunkSpec:
    return ChunkSpec(**spec_dict)


def _rehydrate_video_params(video_params: dict) -> VideoEmbeddingParams:
    """Turn a pure ``VideoEmbeddingParams`` dict back into the dataclass.

    Accepts both the flat dataclass layout and the ``{"audio": ..., "video":
    ...}`` envelope emitted by ``embedding_params_to_dict`` — the caller
    typically passes just the video leaf.
    """
    if "video" in video_params or "audio" in video_params:
        return embedding_params_from_dict(video_params).video  # type: ignore[return-value]
    data = dict(video_params)
    data.setdefault("min_block_variance", 0.0)
    data.setdefault("block_oversample", 1)
    return VideoEmbeddingParams(**data)


def process_video_chunk(
    source_path: str,
    chunk_spec: dict,
    rs_symbols: list[int],
    content_id: str,
    author_public_key: str,
    pepper: bytes,
    video_params: dict,
    width: int,
    height: int,
    fps: float,
    crf: int,
    output_dir: str,
    segment_duration_s: float = 5.0,
) -> dict:
    """Process one chunk: decode → embed/guard → encode → trim.

    Returns ``asdict(ChunkResult)``. Always returns (never raises) — the
    caller inspects ``success`` and ``error`` to decide whether to abort the
    whole job. Guard segments are decoded and re-encoded untouched; only
    segments inside ``[payload_seg_start, payload_seg_end)`` receive a
    watermark.
    """
    spec = _rehydrate_spec(chunk_spec)
    chunk_id = spec.chunk_id
    chunk_raw = Path(output_dir) / f"chunk_raw_{chunk_id:04d}.mp4"
    chunk_final = Path(output_dir) / f"chunk_{chunk_id:04d}.mp4"

    try:
        vp = _rehydrate_video_params(video_params)

        # Deferred import — keeps the top-level module free of the
        # infrastructure symbol during pytest collection boundary checks.
        from kernel_backend.infrastructure.media.media_service import MediaService

        media = MediaService()

        encoder_proc = media.open_video_encode_stream(
            width,
            height,
            fps,
            chunk_raw,
            crf=crf,
            force_keyframes_every_s=segment_duration_s,
        )

        n_processed = 0
        try:
            for local_seg_idx, frames, _ in media.iter_video_segments_range(
                Path(source_path),
                decode_start_s=spec.decode_start_s,
                decode_end_s=spec.decode_end_s,
                segment_duration_s=segment_duration_s,
                frame_stride=1,
            ):
                # Translate local → global segment index (guard_lead first).
                global_seg_idx = (
                    spec.payload_seg_start
                    - spec.guard_lead_segments
                    + local_seg_idx
                )
                is_payload = (
                    spec.payload_seg_start
                    <= global_seg_idx
                    < spec.payload_seg_end
                )

                if is_payload:
                    symbol = rs_symbols[global_seg_idx]
                    symbol_bits = np.array(
                        [(symbol >> (7 - k)) & 1 for k in range(8)],
                        dtype=np.uint8,
                    )
                    for frame in frames:
                        y, u, v = embed_video_frame_yuvj420_planes(
                            frame,
                            symbol_bits,
                            content_id,
                            author_public_key,
                            global_seg_idx,
                            pepper,
                            use_jnd_adaptive=vp.jnd_adaptive,
                            jnd_params=vp,
                        )
                        encoder_proc.stdin.write(y)
                        encoder_proc.stdin.write(u)
                        encoder_proc.stdin.write(v)
                    n_processed += 1
                else:
                    for frame in frames:
                        y, u, v = frame_to_yuvj420_planes(frame)
                        encoder_proc.stdin.write(y)
                        encoder_proc.stdin.write(u)
                        encoder_proc.stdin.write(v)
        finally:
            if encoder_proc.stdin:
                encoder_proc.stdin.close()
            encoder_proc.wait()

        if encoder_proc.returncode != 0:
            chunk_raw.unlink(missing_ok=True)
            raise RuntimeError(
                f"encoder exited with returncode={encoder_proc.returncode}"
            )

        trim_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{spec.payload_start_s:.6f}",
            "-i", str(chunk_raw),
            "-t", f"{spec.expected_payload_duration_s:.6f}",
            "-c", "copy",
            "-loglevel", "quiet",
            str(chunk_final),
        ]
        trim_proc = subprocess.run(
            trim_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if trim_proc.returncode != 0:
            chunk_final.unlink(missing_ok=True)
            raise RuntimeError(
                f"trim exited with returncode={trim_proc.returncode}: "
                f"{trim_proc.stderr.decode('utf-8', errors='replace')[:200]}"
            )

        chunk_raw.unlink(missing_ok=True)

        return asdict(
            ChunkResult(
                chunk_id=chunk_id,
                output_path=str(chunk_final),
                n_segments_processed=n_processed,
                success=True,
            )
        )

    except Exception as exc:  # noqa: BLE001 — never propagate; fold into result
        chunk_raw.unlink(missing_ok=True)
        chunk_final.unlink(missing_ok=True)
        return asdict(
            ChunkResult(
                chunk_id=chunk_id,
                output_path="",
                n_segments_processed=0,
                success=False,
                error=f"chunk {chunk_id}: {exc}",
            )
        )
