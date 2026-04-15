"""Unit tests for process_video_chunk — mocked encoder and range decode.

Real video integration lives in tests/polygon/test_chunked_signing.py (T8).
Here we verify the guard / payload branching, failure handling, and the
local → global segment index translation — without touching ffmpeg.
"""
from __future__ import annotations

import io
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kernel_backend.core.domain.chunk import ChunkSpec
from kernel_backend.core.domain.watermark import VideoEmbeddingParams


def _spec_dict(
    chunk_id: int,
    start: int,
    end: int,
    guard_lead: int = 0,
    guard_trail: int = 0,
    seg_s: float = 5.0,
) -> dict:
    spec = ChunkSpec(
        chunk_id=chunk_id,
        payload_seg_start=start,
        payload_seg_end=end,
        guard_lead_segments=guard_lead,
        guard_trail_segments=guard_trail,
        decode_start_s=max(0.0, (start - guard_lead) * seg_s),
        decode_end_s=(end + guard_trail) * seg_s,
        payload_start_s=guard_lead * seg_s,
        expected_payload_duration_s=(end - start) * seg_s,
    )
    return asdict(spec)


def _fake_video_params() -> dict:
    return asdict(
        VideoEmbeddingParams(
            jnd_adaptive=False,
            qim_step_base=48.0,
            qim_step_min=32.0,
            qim_step_max=64.0,
            qim_quantize_to=1.0,
        )
    )


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def close(self) -> None:  # match subprocess.Popen.stdin
        pass


class _FakeEncoderProc:
    def __init__(self, returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


class _FakeMediaService:
    def __init__(
        self,
        segments_by_range: list[tuple[int, list[np.ndarray], float]],
        encoder_returncode: int = 0,
    ) -> None:
        self._segments = segments_by_range
        self._encoder_returncode = encoder_returncode
        self.encoder_procs: list[_FakeEncoderProc] = []
        self.open_calls: list[dict] = []
        self.range_calls: list[dict] = []

    def open_video_encode_stream(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: Path,
        crf: int = 18,
        force_keyframes_every_s: float | None = None,
    ) -> _FakeEncoderProc:
        self.open_calls.append(
            {
                "width": width,
                "height": height,
                "fps": fps,
                "output_path": output_path,
                "crf": crf,
                "force_keyframes_every_s": force_keyframes_every_s,
            }
        )
        # Pretend the encoder produced a file.
        Path(output_path).write_bytes(b"fake-mp4")
        proc = _FakeEncoderProc(returncode=self._encoder_returncode)
        self.encoder_procs.append(proc)
        return proc

    def iter_video_segments_range(
        self,
        path: Path,
        decode_start_s: float,
        decode_end_s: float,
        segment_duration_s: float = 5.0,
        frame_stride: int = 1,
    ):
        self.range_calls.append(
            {
                "path": path,
                "decode_start_s": decode_start_s,
                "decode_end_s": decode_end_s,
                "segment_duration_s": segment_duration_s,
            }
        )
        for item in self._segments:
            yield item


@pytest.fixture
def patched_worker(monkeypatch, tmp_path):
    """Patch MediaService, subprocess.run (trim), and the embed helper.

    Returns a factory `make(segments, encoder_rc, trim_rc)` plus recorders
    for the embed calls and trim invocations.
    """
    from kernel_backend.core.services import chunk_worker as cw

    embed_calls: list[dict] = []

    def fake_embed(
        frame,
        symbol_bits,
        content_id,
        author_public_key,
        segment_idx,
        pepper,
        use_jnd_adaptive=False,
        jnd_params=None,
    ):
        embed_calls.append(
            {
                "segment_idx": segment_idx,
                "symbol_bits": tuple(int(b) for b in symbol_bits),
            }
        )
        return b"y", b"u", b"v"

    passthrough_calls: list[int] = []

    def fake_passthrough(frame):
        passthrough_calls.append(1)
        return b"y", b"u", b"v"

    monkeypatch.setattr(cw, "embed_video_frame_yuvj420_planes", fake_embed)
    monkeypatch.setattr(cw, "frame_to_yuvj420_planes", fake_passthrough)

    trim_calls: list[list[str]] = []

    def make(segments, encoder_rc=0, trim_rc=0):
        fake_media = _FakeMediaService(segments, encoder_returncode=encoder_rc)

        import kernel_backend.infrastructure.media.media_service as media_mod

        monkeypatch.setattr(media_mod, "MediaService", lambda: fake_media)

        def fake_run(cmd, stdout=None, stderr=None):
            trim_calls.append(cmd)
            return SimpleNamespace(returncode=trim_rc, stderr=b"trim fake error")

        monkeypatch.setattr(subprocess, "run", fake_run)
        return fake_media

    return SimpleNamespace(
        make=make,
        embed_calls=embed_calls,
        passthrough_calls=passthrough_calls,
        trim_calls=trim_calls,
    )


def _dummy_frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_guard_frames_not_watermarked(patched_worker, tmp_path) -> None:
    from kernel_backend.core.services.chunk_worker import process_video_chunk

    # Middle chunk: payload segments [15, 18), guard_lead=1 (seg 14),
    # guard_trail=1 (seg 18). Yield 5 local segments.
    spec = _spec_dict(1, 15, 18, guard_lead=1, guard_trail=1)
    rs_symbols = list(range(20))  # arbitrary symbols
    segments = [(i, [_dummy_frame()], 25.0) for i in range(5)]
    patched_worker.make(segments)

    result = process_video_chunk(
        source_path=str(tmp_path / "src.mp4"),
        chunk_spec=spec,
        rs_symbols=rs_symbols,
        content_id="cid",
        author_public_key="pk",
        pepper=b"pepper",
        video_params=_fake_video_params(),
        width=16,
        height=16,
        fps=25.0,
        crf=18,
        output_dir=str(tmp_path),
    )

    assert result["success"] is True
    assert result["n_segments_processed"] == 3  # payload-only count

    # Only segments with global index in [15, 18) should have been embedded.
    embedded_idxs = [c["segment_idx"] for c in patched_worker.embed_calls]
    assert embedded_idxs == [15, 16, 17]
    # Guard segments go through the pass-through helper — 2 guard segs × 1 frame.
    assert len(patched_worker.passthrough_calls) == 2


def test_rs_symbols_correct_mapping(patched_worker, tmp_path) -> None:
    from kernel_backend.core.services.chunk_worker import process_video_chunk

    spec = _spec_dict(1, 15, 16, guard_lead=1, guard_trail=0)
    rs_symbols = [0] * 20
    rs_symbols[15] = 0xA5  # 1010_0101
    segments = [(i, [_dummy_frame()], 25.0) for i in range(2)]
    patched_worker.make(segments)

    process_video_chunk(
        source_path=str(tmp_path / "src.mp4"),
        chunk_spec=spec,
        rs_symbols=rs_symbols,
        content_id="cid",
        author_public_key="pk",
        pepper=b"pepper",
        video_params=_fake_video_params(),
        width=16,
        height=16,
        fps=25.0,
        crf=18,
        output_dir=str(tmp_path),
    )

    # Exactly one embed call, for global segment 15 with bits of 0xA5.
    assert len(patched_worker.embed_calls) == 1
    call = patched_worker.embed_calls[0]
    assert call["segment_idx"] == 15
    expected = tuple((0xA5 >> (7 - k)) & 1 for k in range(8))
    assert call["symbol_bits"] == expected


def test_chunk_result_on_encoder_failure(patched_worker, tmp_path) -> None:
    from kernel_backend.core.services.chunk_worker import process_video_chunk

    spec = _spec_dict(0, 0, 2)
    segments = [(i, [_dummy_frame()], 25.0) for i in range(2)]
    # Encoder returns non-zero — should become a failure result, not an exception.
    patched_worker.make(segments, encoder_rc=1)

    result = process_video_chunk(
        source_path=str(tmp_path / "src.mp4"),
        chunk_spec=spec,
        rs_symbols=list(range(4)),
        content_id="cid",
        author_public_key="pk",
        pepper=b"pepper",
        video_params=_fake_video_params(),
        width=16,
        height=16,
        fps=25.0,
        crf=18,
        output_dir=str(tmp_path),
    )

    assert result["success"] is False
    assert "returncode=1" in result["error"]
    assert result["chunk_id"] == 0
    # No chunk file left behind on failure.
    assert not (tmp_path / "chunk_raw_0000.mp4").exists()
    assert not (tmp_path / "chunk_0000.mp4").exists()


def test_chunk_result_on_trim_failure(patched_worker, tmp_path) -> None:
    from kernel_backend.core.services.chunk_worker import process_video_chunk

    spec = _spec_dict(0, 0, 2)
    segments = [(i, [_dummy_frame()], 25.0) for i in range(2)]
    patched_worker.make(segments, encoder_rc=0, trim_rc=2)

    result = process_video_chunk(
        source_path=str(tmp_path / "src.mp4"),
        chunk_spec=spec,
        rs_symbols=list(range(4)),
        content_id="cid",
        author_public_key="pk",
        pepper=b"pepper",
        video_params=_fake_video_params(),
        width=16,
        height=16,
        fps=25.0,
        crf=18,
        output_dir=str(tmp_path),
    )

    assert result["success"] is False
    assert "trim" in result["error"]


def test_encoder_called_with_keyframe_pin(patched_worker, tmp_path) -> None:
    from kernel_backend.core.services.chunk_worker import process_video_chunk

    spec = _spec_dict(0, 0, 2)
    segments = [(i, [_dummy_frame()], 25.0) for i in range(2)]
    fake_media = patched_worker.make(segments)

    process_video_chunk(
        source_path=str(tmp_path / "src.mp4"),
        chunk_spec=spec,
        rs_symbols=list(range(4)),
        content_id="cid",
        author_public_key="pk",
        pepper=b"pepper",
        video_params=_fake_video_params(),
        width=16,
        height=16,
        fps=25.0,
        crf=18,
        output_dir=str(tmp_path),
    )

    # Worker must request keyframes every segment so -c copy trim is exact.
    assert fake_media.open_calls[0]["force_keyframes_every_s"] == 5.0
