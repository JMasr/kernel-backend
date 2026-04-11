"""
Deterministic video content profiler for content-adaptive watermark routing.

Extracts 7 visual descriptors from sampled frames and classifies video into
five content types via a hard-coded decision tree.

No file I/O -- receives BGR numpy frames, returns VideoContentProfile.
No neural networks -- pure OpenCV/numpy arithmetic for determinism.

Determinism guarantees:
  - Fixed Canny thresholds (50, 150), Sobel ksize=3
  - All features rounded to 4 decimal places
  - Classification via if-else tree (no floating-point accumulation order risk)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from kernel_backend.core.domain.video_content_profile import (
    VideoContentProfile,
    VideoContentType,
)

# -- Deterministic constants -------------------------------------------------

_DECIMALS = 4
_DESCRIPTOR_VERSION = "1.0.0"

_CANNY_LOW = 50
_CANNY_HIGH = 150
_SOBEL_KSIZE = 3

# Dark/bright pixel thresholds (Y channel, 0-255)
_DARK_PIXEL_THRESHOLD = 30
_BRIGHT_PIXEL_THRESHOLD = 225

# -- Code hash (computed once at module load) --------------------------------

_CODE_HASH: str = "sha256:" + hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()[:16]

# -- Decision tree thresholds ------------------------------------------------

_DARK_LUMINANCE_MAX = 50.0
_DARK_PIXEL_RATIO_MIN = 0.40

_BRIGHT_LUMINANCE_MIN = 200.0
_BRIGHT_PIXEL_RATIO_MIN = 0.40

_STATIC_MOTION_MAX = 0.005
_STATIC_EDGE_DENSITY_MAX = 0.02

_HIGH_MOTION_MIN = 0.08

_CONFIDENCE_THRESHOLD = 0.7


def _extract_frame_features(y_channel: np.ndarray) -> dict[str, float]:
    """Extract per-frame spatial descriptors from Y (luma) channel.

    Args:
        y_channel: uint8 grayscale array, shape (H, W)

    Returns:
        dict of feature name -> rounded float value.
    """
    n_pixels = float(y_channel.size)

    mean_lum = round(float(np.mean(y_channel)), _DECIMALS)
    lum_std = round(float(np.std(y_channel)), _DECIMALS)
    dark_ratio = round(float(np.sum(y_channel < _DARK_PIXEL_THRESHOLD)) / n_pixels, _DECIMALS)
    bright_ratio = round(float(np.sum(y_channel > _BRIGHT_PIXEL_THRESHOLD)) / n_pixels, _DECIMALS)

    # Edge density via Canny
    edges = cv2.Canny(y_channel, _CANNY_LOW, _CANNY_HIGH)
    edge_density = round(float(np.sum(edges > 0)) / n_pixels, _DECIMALS)

    # Spatial complexity via Sobel gradient magnitude
    gx = cv2.Sobel(y_channel, cv2.CV_64F, 1, 0, ksize=_SOBEL_KSIZE)
    gy = cv2.Sobel(y_channel, cv2.CV_64F, 0, 1, ksize=_SOBEL_KSIZE)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    spatial_complexity = round(float(np.mean(grad_mag)), _DECIMALS)

    return {
        "mean_luminance": mean_lum,
        "luminance_std": lum_std,
        "dark_pixel_ratio": dark_ratio,
        "bright_pixel_ratio": bright_ratio,
        "edge_density": edge_density,
        "spatial_complexity": spatial_complexity,
    }


def _compute_temporal_motion(y_channels: list[np.ndarray]) -> float:
    """Compute mean absolute frame difference across consecutive frames.

    Returns value normalized to [0, 1] (divided by 255).
    If only 1 frame, returns 0.0.
    """
    if len(y_channels) < 2:
        return 0.0

    diffs = []
    for i in range(len(y_channels) - 1):
        diff = np.abs(y_channels[i].astype(np.float32) - y_channels[i + 1].astype(np.float32))
        diffs.append(float(np.mean(diff)) / 255.0)

    return round(float(np.mean(diffs)), _DECIMALS)


def _classify(features: dict[str, float]) -> tuple[VideoContentType, float]:
    """Hard-coded decision tree for video content classification.

    Returns (content_type, confidence).

    Hierarchy:
      1. Dark (low luminance + many dark pixels)
      2. Bright (high luminance + many bright pixels)
      3. Static (no motion + low edge density)
      4. High motion (large frame differences)
      5. Normal (everything else)
    """
    mean_lum = features["mean_luminance"]
    dark_ratio = features["dark_pixel_ratio"]
    bright_ratio = features["bright_pixel_ratio"]
    edge_density = features["edge_density"]
    temporal_motion = features["temporal_motion"]

    # Stage 1: Dark detection
    if mean_lum < _DARK_LUMINANCE_MAX and dark_ratio > _DARK_PIXEL_RATIO_MIN:
        confidence = min(0.99, 0.80 + (1.0 - mean_lum / _DARK_LUMINANCE_MAX) * 0.15)
        return "dark", round(confidence, 2)

    # Stage 2: Bright detection
    if mean_lum > _BRIGHT_LUMINANCE_MIN and bright_ratio > _BRIGHT_PIXEL_RATIO_MIN:
        confidence = min(0.99, 0.80 + (mean_lum - _BRIGHT_LUMINANCE_MIN) / 55.0 * 0.15)
        return "bright", round(confidence, 2)

    # Stage 3: Static detection
    if temporal_motion < _STATIC_MOTION_MAX and edge_density < _STATIC_EDGE_DENSITY_MAX:
        return "static", 0.90

    # Stage 4: High motion detection
    if temporal_motion > _HIGH_MOTION_MIN:
        confidence = min(0.99, 0.80 + (temporal_motion - _HIGH_MOTION_MIN) * 2.0)
        return "high_motion", round(confidence, 2)

    # Stage 5: Normal
    return "normal", 0.95


def profile_video(frames: list[np.ndarray]) -> VideoContentProfile:
    """Profile video content from sampled BGR frames.

    Args:
        frames: list of BGR uint8 frames (at least 1). Caller is responsible
                for sampling representative frames (e.g. 5 frames at
                10%, 25%, 50%, 75%, 90% of the video).

    Returns:
        Frozen VideoContentProfile with content_type, confidence, and features.
    """
    if not frames:
        raise ValueError("At least one frame is required for video profiling")

    # Convert to Y channel (luma)
    y_channels: list[np.ndarray] = []
    for frame in frames:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channels.append(ycrcb[:, :, 0])

    # Extract per-frame features and average
    all_frame_features: list[dict[str, float]] = []
    for y in y_channels:
        all_frame_features.append(_extract_frame_features(y))

    # Average spatial features across frames
    feature_keys = all_frame_features[0].keys()
    features: dict[str, float] = {}
    for key in feature_keys:
        values = [f[key] for f in all_frame_features]
        features[key] = round(float(np.mean(values)), _DECIMALS)

    # Temporal motion (cross-frame descriptor)
    features["temporal_motion"] = _compute_temporal_motion(y_channels)

    # Classify
    content_type, confidence = _classify(features)

    # Low-confidence fallback to normal (safest default)
    if confidence < _CONFIDENCE_THRESHOLD:
        content_type = "normal"

    return VideoContentProfile(
        content_type=content_type,
        confidence=confidence,
        features=features,
        descriptor_version=_DESCRIPTOR_VERSION,
        code_hash=_CODE_HASH,
    )
