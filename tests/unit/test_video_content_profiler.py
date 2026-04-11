"""Unit tests for the video content profiler engine module."""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.engine.video.content_profiler import (
    _CODE_HASH,
    profile_video,
)


# -- Synthetic frame generators ----------------------------------------------

def _bgr_frame_from_luma(y_value: int, width: int = 64, height: int = 64) -> np.ndarray:
    """Create a BGR frame where all pixels have approximately the given Y value."""
    import cv2
    ycrcb = np.full((height, width, 3), [y_value, 128, 128], dtype=np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _dark_frame(width: int = 64, height: int = 64) -> np.ndarray:
    """Create a very dark BGR frame (Y ~= 15)."""
    return _bgr_frame_from_luma(15, width, height)


def _bright_frame(width: int = 64, height: int = 64) -> np.ndarray:
    """Create a very bright BGR frame (Y ~= 240)."""
    return _bgr_frame_from_luma(240, width, height)


def _normal_frame(width: int = 64, height: int = 64, seed: int = 42) -> np.ndarray:
    """Create a mid-gray frame with edges and texture (normal content)."""
    rng = np.random.default_rng(seed)
    # Base mid-gray with texture
    frame = rng.integers(80, 180, size=(height, width, 3), dtype=np.uint8)
    # Add some strong edges (horizontal and vertical lines)
    frame[height // 4, :, :] = 255
    frame[3 * height // 4, :, :] = 0
    frame[:, width // 4, :] = 255
    frame[:, 3 * width // 4, :] = 0
    return frame


def _high_motion_frames(n: int = 5, width: int = 64, height: int = 64) -> list[np.ndarray]:
    """Create frames with large differences between consecutive frames."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n):
        # Each frame is substantially different from the previous
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def _static_frames(n: int = 5, width: int = 64, height: int = 64) -> list[np.ndarray]:
    """Create identical frames (static content, low edge density)."""
    # Use a uniform mid-gray frame (no edges)
    frame = _bgr_frame_from_luma(128, width, height)
    return [frame.copy() for _ in range(n)]


# -- Tests -------------------------------------------------------------------

class TestDeterminism:
    """Same input must always produce the exact same output."""

    def test_same_frames_same_profile(self):
        frames = [_normal_frame(seed=123) for _ in range(3)]
        p1 = profile_video([f.copy() for f in frames])
        p2 = profile_video([f.copy() for f in frames])
        assert p1 == p2

    def test_features_are_rounded(self):
        frames = [_normal_frame()]
        profile = profile_video(frames)
        for key, value in profile.features.items():
            assert round(value, 4) == value, f"{key}={value} not rounded to 4 decimals"


class TestDarkDetection:
    def test_dark_frames(self):
        frames = [_dark_frame() for _ in range(5)]
        profile = profile_video(frames)
        assert profile.content_type == "dark"
        assert profile.confidence >= 0.80

    def test_dark_features(self):
        frames = [_dark_frame()]
        profile = profile_video(frames)
        assert profile.features["mean_luminance"] < 50
        assert profile.features["dark_pixel_ratio"] > 0.40


class TestBrightDetection:
    def test_bright_frames(self):
        frames = [_bright_frame() for _ in range(5)]
        profile = profile_video(frames)
        assert profile.content_type == "bright"
        assert profile.confidence >= 0.80

    def test_bright_features(self):
        frames = [_bright_frame()]
        profile = profile_video(frames)
        assert profile.features["mean_luminance"] > 200
        assert profile.features["bright_pixel_ratio"] > 0.40


class TestNormalDetection:
    def test_normal_textured_frame(self):
        # Use a single base frame with slight per-frame noise to keep
        # temporal_motion moderate (not high_motion).
        base = _normal_frame(seed=42)
        rng = np.random.default_rng(99)
        frames = []
        for _ in range(5):
            noise = rng.integers(-5, 6, size=base.shape, dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)
        profile = profile_video(frames)
        assert profile.content_type == "normal"
        assert profile.confidence >= 0.90


class TestStaticDetection:
    def test_identical_frames_low_edges(self):
        frames = _static_frames(n=5)
        profile = profile_video(frames)
        assert profile.content_type == "static"
        assert profile.features["temporal_motion"] < 0.005


class TestHighMotionDetection:
    def test_random_frames(self):
        frames = _high_motion_frames(n=5)
        profile = profile_video(frames)
        assert profile.content_type == "high_motion"
        assert profile.features["temporal_motion"] > 0.08


class TestSingleFrame:
    def test_single_frame_motion_is_zero(self):
        frames = [_normal_frame()]
        profile = profile_video(frames)
        assert profile.features["temporal_motion"] == 0.0

    def test_empty_frames_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            profile_video([])


class TestCodeHash:
    def test_code_hash_is_valid_hex(self):
        assert _CODE_HASH.startswith("sha256:")
        hex_part = _CODE_HASH[7:]
        assert len(hex_part) == 16
        int(hex_part, 16)  # should not raise

    def test_code_hash_stable(self):
        from kernel_backend.engine.video.content_profiler import _CODE_HASH as h2
        assert _CODE_HASH == h2


class TestProfileMetadata:
    def test_descriptor_version(self):
        frames = [_normal_frame()]
        profile = profile_video(frames)
        assert profile.descriptor_version == "1.0.0"

    def test_features_contain_expected_keys(self):
        frames = [_normal_frame(seed=1), _normal_frame(seed=2)]
        profile = profile_video(frames)
        expected_keys = {
            "mean_luminance", "luminance_std", "dark_pixel_ratio",
            "bright_pixel_ratio", "edge_density", "spatial_complexity",
            "temporal_motion",
        }
        assert expected_keys == set(profile.features.keys())
