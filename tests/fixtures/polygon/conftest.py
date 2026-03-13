"""
Polygon pytest fixtures.

Scope rules:
  - DatasetRegistry is session-scoped (loaded once per run)
  - Clip fixtures are session-scoped (paths resolved once; audio loaded per test)
  - Clips that do not exist on disk are skipped at collection time via
    empty params lists — no runtime skip() calls needed

Marker: @pytest.mark.polygon
  Tests using these fixtures must be marked @pytest.mark.polygon.
  They are excluded from default CI runs:
    pytest tests/ -m "not polygon"
  Run polygon tests explicitly:
    pytest tests/ -m "polygon" -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from tests.fixtures.polygon.registry import (
    AudioClip,
    DatasetRegistry,
    MANIFEST_PATH,
    VideoClip,
)


# ── Registry ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def polygon() -> DatasetRegistry:
    """
    Session-scoped registry. Skips entire session if manifest is missing.
    Tests that use clip fixtures will be skipped automatically when
    the category has no available clips.
    """
    if not MANIFEST_PATH.exists():
        pytest.skip(
            "data/manifest.yaml not found. "
            "Run: uv run python scripts/setup_polygon_audio.py"
        )
    return DatasetRegistry()


# ── Audio fixtures ────────────────────────────────────────────────────────────

def _audio_params(category: str) -> list:
    """Resolved at collection time. Returns [] if manifest absent or clips missing."""
    if not MANIFEST_PATH.exists():
        return []
    try:
        registry = DatasetRegistry()
        clips = registry.get(category).available_audio()
        return [pytest.param(c, id=c.id) for c in clips]
    except Exception:
        return []


@pytest.fixture(
    params=_audio_params("audio.speech"),
    scope="session",
)
def speech_clip(request) -> AudioClip:
    """
    Parametrized: runs once per available clip in audio.speech.
    Returns AudioClip. Load audio in your test with load_polygon_audio().
    Skipped automatically if no clips are available.
    """
    return request.param


@pytest.fixture(
    params=_audio_params("audio.music"),
    scope="session",
)
def music_clip(request) -> AudioClip:
    """Parametrized: runs once per available clip in audio.music."""
    return request.param


# ── Video fixtures ────────────────────────────────────────────────────────────

def _video_params(category: str) -> list:
    if not MANIFEST_PATH.exists():
        return []
    try:
        registry = DatasetRegistry()
        clips = registry.get(category).available_video()
        return [pytest.param(c, id=c.id) for c in clips]
    except Exception:
        return []


@pytest.fixture(
    params=_video_params("video.speech"),
    scope="session",
)
def video_speech_clip(request) -> VideoClip:
    """Parametrized: runs once per available clip in video.speech."""
    return request.param


@pytest.fixture(
    params=_video_params("video.outside"),
    scope="session",
)
def video_outside_clip(request) -> VideoClip:
    """Parametrized: runs once per available clip in video.outside."""
    return request.param


@pytest.fixture(
    params=_video_params("video.without_audio"),
    scope="session",
)
def video_without_audio_clip(request) -> VideoClip:
    """Parametrized: runs once per available clip in video.without_audio."""
    return request.param


@pytest.fixture(
    params=_video_params("video.others"),
    scope="session",
)
def video_others_clip(request) -> VideoClip:
    """Parametrized: runs once per available clip in video.others."""
    return request.param


# ── Audio loader utility (not a fixture) ─────────────────────────────────────

def load_polygon_audio(clip: AudioClip) -> tuple[np.ndarray, int]:
    """
    Loads a WAV file and returns (float32 samples, sample_rate).
    Normalizes int16 PCM to [-1.0, 1.0].
    Use inside polygon tests — not a fixture, call directly.
    """
    sr, data = wavfile.read(str(clip.path))
    if data.ndim > 1:
        data = data[:, 0]
    samples = data.astype(np.float32)
    if data.dtype == np.int16:
        samples /= 32768.0
    elif data.dtype == np.int32:
        samples /= 2147483648.0
    return samples, sr
