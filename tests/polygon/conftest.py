"""Polygon-local conftest.

Provides synthetic ffmpeg-generated video fixtures used by the chunked-signing
regression tests. Complements the dataset-driven fixtures in
``tests/fixtures/polygon/conftest.py`` (clip registry); nothing here depends
on ``data/manifest.yaml``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _generate_video(
    duration_s: int,
    out: Path,
    *,
    width: int = 640,
    height: int = 360,
    fps: int = 25,
) -> Path:
    """Generate a synthetic video with motion + per-frame noise.

    ``testsrc2`` alone has large flat regions; after libx264 quantization the
    DCT AC coefficients used by the watermark collapse to zero in those
    regions, driving WID recovery below the RS correction budget. The
    ``noise=c0s=100:allf=t`` overlay keeps enough entropy in every block for
    the embed + verify roundtrip to survive — same trick used by
    ``synthetic_video_120s`` in tests/unit/test_pipeline_sign_verify.py.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", (
                f"testsrc2=duration={duration_s}:size={width}x{height}:rate={fps}"
                ",noise=c0s=100:allf=t"
            ),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-crf", "18",
            "-an",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def chunked_test_video(tmp_path_factory) -> Path:
    """150s synthetic video (30 payload segments) — enough for 2-chunk split."""
    tmp = tmp_path_factory.mktemp("chunked_video")
    return _generate_video(150, tmp / "video_150s.mp4")


@pytest.fixture(scope="module")
def chunked_test_video_90s(tmp_path_factory) -> Path:
    """90s synthetic video (18 payload segments) — just above the RS minimum."""
    tmp = tmp_path_factory.mktemp("chunked_video_short")
    return _generate_video(90, tmp / "video_90s.mp4")
