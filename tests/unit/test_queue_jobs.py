"""Unit tests for ``infrastructure/queue/jobs.py`` worker-side validation.

``_validate_local_media`` used to live inside ``POST /sign``. Moving it into the
ARQ worker lets the API return 202 as soon as bytes are on disk — especially
valuable when ``normalize_video_input`` has to transcode a non-H.264 input (up
to 30–120 s for a large video). These tests cover the behaviour that used to
live as ``422`` responses on the endpoint and now surface via
``GET /sign/{job_id}`` with ``status="failed"`` + ``error`` strings.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kernel_backend.core.domain.media import MediaProfile
from kernel_backend.infrastructure.queue.jobs import (
    _MAX_DURATION,
    _MIN_AUDIO_DURATION,
    _MIN_VIDEO_DURATION,
    _validate_local_media,
)


def _profile(
    *,
    has_video: bool = False,
    has_audio: bool = True,
    duration_s: float = 60.0,
) -> MediaProfile:
    return MediaProfile(
        has_video=has_video,
        has_audio=has_audio,
        width=1280 if has_video else 0,
        height=720 if has_video else 0,
        fps=30.0 if has_video else 0.0,
        duration_s=duration_s,
        sample_rate=44100 if has_audio else 0,
    )


@pytest.mark.asyncio
async def test_validates_valid_audio_returns_path_and_profile(tmp_path: Path) -> None:
    """Audio-only within bounds → returns original path + profile, no transcode."""
    src = tmp_path / "ok.aac"
    src.write_bytes(b"fake")

    with patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.normalize_video_input",
        return_value=(src, False),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.probe",
        return_value=_profile(has_audio=True, duration_s=90.0),
    ):
        path, profile = await _validate_local_media(str(src))

    assert path == str(src)
    assert profile.has_audio is True
    assert src.exists(), "non-transcoded source must not be deleted"


@pytest.mark.asyncio
async def test_transcoded_video_returns_new_path_and_unlinks_original(tmp_path: Path) -> None:
    """Non-H.264 video → normalize_video_input returns a new path; original is unlinked."""
    src = tmp_path / "raw.mov"
    src.write_bytes(b"fake hevc")
    transcoded = tmp_path / "normalized.mp4"
    transcoded.write_bytes(b"fake h264")

    with patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.normalize_video_input",
        return_value=(transcoded, True),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.probe",
        return_value=_profile(has_video=True, has_audio=True, duration_s=120.0),
    ):
        path, profile = await _validate_local_media(str(src))

    assert path == str(transcoded)
    assert profile.has_video and profile.has_audio
    assert not src.exists(), "original upload must be unlinked when transcoded"
    assert transcoded.exists()


@pytest.mark.asyncio
async def test_rejects_too_long_file(tmp_path: Path) -> None:
    src = tmp_path / "long.mp4"
    src.write_bytes(b"fake")

    with patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.normalize_video_input",
        return_value=(src, False),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.probe",
        return_value=_profile(has_video=True, has_audio=True, duration_s=_MAX_DURATION + 1),
    ):
        with pytest.raises(ValueError, match="too long"):
            await _validate_local_media(str(src))


@pytest.mark.asyncio
async def test_rejects_too_short_video(tmp_path: Path) -> None:
    src = tmp_path / "short.mp4"
    src.write_bytes(b"fake")

    with patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.normalize_video_input",
        return_value=(src, False),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.probe",
        return_value=_profile(has_video=True, has_audio=True, duration_s=_MIN_VIDEO_DURATION - 1),
    ):
        with pytest.raises(ValueError, match="Video is too short"):
            await _validate_local_media(str(src))


@pytest.mark.asyncio
async def test_rejects_too_short_audio_only(tmp_path: Path) -> None:
    src = tmp_path / "short.aac"
    src.write_bytes(b"fake")

    with patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.normalize_video_input",
        return_value=(src, False),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.probe",
        return_value=_profile(has_audio=True, duration_s=_MIN_AUDIO_DURATION - 1),
    ):
        with pytest.raises(ValueError, match="Audio is too short"):
            await _validate_local_media(str(src))


@pytest.mark.asyncio
async def test_video_only_short_is_rejected_as_video(tmp_path: Path) -> None:
    """Silent video below the video threshold hits the video branch, not the audio one."""
    src = tmp_path / "silent.mp4"
    src.write_bytes(b"fake")

    with patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.normalize_video_input",
        return_value=(src, False),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.probe",
        return_value=_profile(has_video=True, has_audio=False, duration_s=_MIN_VIDEO_DURATION - 1),
    ):
        with pytest.raises(ValueError, match="Video is too short"):
            await _validate_local_media(str(src))


@pytest.mark.asyncio
async def test_probe_failure_surfaces_as_valueerror(tmp_path: Path) -> None:
    """ffprobe crash → user sees the same message via GET /sign/{job_id}."""
    src = tmp_path / "corrupt.mp4"
    src.write_bytes(b"fake")

    with patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.normalize_video_input",
        return_value=(src, False),
    ), patch(
        "kernel_backend.infrastructure.queue.jobs.MediaService.probe",
        side_effect=ValueError("ffprobe failed for corrupt.mp4: invalid container"),
    ):
        with pytest.raises(ValueError, match="ffprobe failed"):
            await _validate_local_media(str(src))
