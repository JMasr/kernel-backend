"""
tests/unit/test_media_port_iter.py

Unit tests for MediaService.iter_video_segments() — Phase 4 Step 1.

Tests use the speech_01 polygon clip when available, otherwise skip.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

DATA_ROOT = Path(__file__).parents[2] / "data"
SPEECH_01 = DATA_ROOT / "video" / "speech" / "speech.mp4"
EXPECTED_SEGMENTS = 41  # 206 s // 5 s


def _requires_real_clip(path: Path):
    if not path.exists():
        pytest.skip(f"Polygon clip not found: {path}. Place video files in data/video/")


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_iter_segments_yields_correct_count():
    """[BLOCKING] speech_01 (206s / 5s) → 41 full segments."""
    _requires_real_clip(SPEECH_01)

    from kernel_backend.infrastructure.media.media_service import MediaService

    media = MediaService()
    segments = list(media.iter_video_segments(SPEECH_01, segment_duration_s=5.0))
    assert len(segments) >= EXPECTED_SEGMENTS, (
        f"Expected at least {EXPECTED_SEGMENTS} segments, got {len(segments)}"
    )


def test_iter_segments_frame_count_per_segment():
    """[BLOCKING] Each yielded segment has frames and fps > 0."""
    _requires_real_clip(SPEECH_01)

    from kernel_backend.infrastructure.media.media_service import MediaService

    media = MediaService()
    for seg_idx, frames, fps in media.iter_video_segments(SPEECH_01, segment_duration_s=5.0):
        assert len(frames) > 0, f"Segment {seg_idx} yielded 0 frames"
        assert fps > 0, f"Segment {seg_idx} yielded fps={fps}"
        # Spot-check: stop after 3 segments to keep the test fast
        if seg_idx >= 2:
            break


def test_iter_segments_does_not_buffer_all_frames():
    """
    [BLOCKING] Verifies lazy loading: at no point during iteration should the
    service hold more than one segment's worth of frames in memory.

    Proxy: the maximum length of the `frames` list yielded per segment must
    equal approximately fps * (segment_duration_s - frame_offset_s), NOT
    the total frame count of the video.
    """
    _requires_real_clip(SPEECH_01)

    from kernel_backend.infrastructure.media.media_service import MediaService

    media = MediaService()
    SEGMENT_S = 5.0
    OFFSET_S = 0.5

    for seg_idx, frames, fps in media.iter_video_segments(
        SPEECH_01,
        segment_duration_s=SEGMENT_S,
        frame_offset_s=OFFSET_S,
    ):
        expected_max = int((SEGMENT_S - OFFSET_S) * fps) + 5  # +5 tolerance
        assert len(frames) <= expected_max, (
            f"Segment {seg_idx}: yielded {len(frames)} frames which exceeds "
            f"single-segment capacity ({expected_max}). "
            "iter_video_segments may be buffering the entire file."
        )
        if seg_idx >= 2:
            break  # spot check — no need to iterate the full video
