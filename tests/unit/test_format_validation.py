"""Tests for input format validation and video normalization."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kernel_backend.core.services.format_validation import (
    ACCEPTED_EXTENSIONS,
    validate_media_extension,
)
from kernel_backend.infrastructure.media.media_service import MediaService


# ── Extension validation ──────────────────────────────────────────────


@pytest.mark.parametrize("ext", sorted(ACCEPTED_EXTENSIONS))
def test_accepted_extensions_pass(ext: str):
    validate_media_extension(f"file{ext}")


@pytest.mark.parametrize("ext", sorted(ACCEPTED_EXTENSIONS))
def test_accepted_extensions_case_insensitive(ext: str):
    validate_media_extension(f"FILE{ext.upper()}")


@pytest.mark.parametrize("ext", [".exe", ".zip", ".pdf", ".txt", ".py", ".gif", ".bmp"])
def test_rejected_extensions_raise(ext: str):
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_media_extension(f"file{ext}")


def test_no_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_media_extension("file_without_extension")


# ── normalize_video_input ─────────────────────────────────────────────


def _make_probe_result(codec_name: str = "h264", codec_type: str = "video"):
    return {
        "streams": [{"codec_type": codec_type, "codec_name": codec_name}],
        "format": {"duration": "60.0"},
    }


@patch("kernel_backend.infrastructure.media.media_service.ffmpeg")
def test_normalize_h264_mp4_is_noop(mock_ffmpeg):
    """H.264 in .mp4 container → no transcode."""
    mock_ffmpeg.probe.return_value = _make_probe_result("h264")
    svc = MediaService()
    path = Path("/tmp/test_video.mp4")
    result_path, was_transcoded = svc.normalize_video_input(path)
    assert result_path == path
    assert was_transcoded is False


@patch("kernel_backend.infrastructure.media.media_service.ffmpeg")
def test_normalize_audio_only_is_noop(mock_ffmpeg):
    """Audio-only file → no transcode."""
    mock_ffmpeg.probe.return_value = _make_probe_result("aac", codec_type="audio")
    svc = MediaService()
    path = Path("/tmp/test_audio.m4a")
    result_path, was_transcoded = svc.normalize_video_input(path)
    assert result_path == path
    assert was_transcoded is False


@patch("kernel_backend.infrastructure.media.media_service.ffmpeg")
def test_normalize_hevc_triggers_transcode(mock_ffmpeg):
    """HEVC codec → should transcode to H.264."""
    mock_ffmpeg.probe.return_value = _make_probe_result("hevc")

    # Mock the ffmpeg fluent chain
    mock_input = MagicMock()
    mock_ffmpeg.input.return_value = mock_input
    mock_output = MagicMock()
    mock_input.output.return_value = mock_output
    mock_overwrite = MagicMock()
    mock_output.overwrite_output.return_value = mock_overwrite
    mock_overwrite.run.return_value = None

    svc = MediaService()
    path = Path("/tmp/test_video.mov")
    result_path, was_transcoded = svc.normalize_video_input(path)

    assert was_transcoded is True
    assert result_path.suffix == ".mp4"
    assert result_path != path
    mock_ffmpeg.input.assert_called_once()
    mock_overwrite.run.assert_called_once_with(capture_stderr=True)

    # Clean up temp file reference (file won't exist in test)
    result_path.unlink(missing_ok=True)


@patch("kernel_backend.infrastructure.media.media_service.ffmpeg")
def test_normalize_h264_in_mkv_triggers_transcode(mock_ffmpeg):
    """H.264 codec but .mkv container → should transcode to .mp4."""
    mock_ffmpeg.probe.return_value = _make_probe_result("h264")

    mock_input = MagicMock()
    mock_ffmpeg.input.return_value = mock_input
    mock_output = MagicMock()
    mock_input.output.return_value = mock_output
    mock_overwrite = MagicMock()
    mock_output.overwrite_output.return_value = mock_overwrite
    mock_overwrite.run.return_value = None

    svc = MediaService()
    path = Path("/tmp/test_video.mkv")
    result_path, was_transcoded = svc.normalize_video_input(path)

    assert was_transcoded is True
    assert result_path.suffix == ".mp4"
    result_path.unlink(missing_ok=True)


@patch("kernel_backend.infrastructure.media.media_service.ffmpeg")
def test_normalize_probe_failure_raises(mock_ffmpeg):
    """ffprobe failure → ValueError."""
    mock_ffmpeg.Error = Exception
    mock_ffmpeg.probe.side_effect = Exception("probe failed")
    mock_ffmpeg.probe.side_effect.stderr = MagicMock()
    mock_ffmpeg.probe.side_effect.stderr.decode.return_value = "probe error"

    svc = MediaService()

    # Need to set up the actual ffmpeg.Error class properly
    import ffmpeg as real_ffmpeg
    mock_ffmpeg.Error = real_ffmpeg.Error
    err = real_ffmpeg.Error("ffprobe", stdout=b"", stderr=b"probe error detail")
    mock_ffmpeg.probe.side_effect = err

    with pytest.raises(ValueError, match="Cannot read media file"):
        svc.normalize_video_input(Path("/tmp/bad_file.mp4"))
