"""
Unit tests for engine/video/pilot_tone.py — 48-bit pilot hash in 4×4 DCT luma blocks.

Uses random-noise frames (never uniform-color frames — analogous to pure tones
in audio: near-zero DCT variance causes false negatives).
"""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.engine.video.pilot_tone import (
    PilotDetection,
    embed_pilot,
    detect_pilot,
)

CONTENT_ID_A = "test-content-id-alpha"
CONTENT_ID_B = "test-content-id-beta"
PEPPER_A = b"test-pepper-32-bytes-padded-AAA!"
PEPPER_B = b"test-pepper-32-bytes-padded-BBB!"


def _random_frame(height: int = 540, width: int = 960, seed: int = 42) -> np.ndarray:
    """Generate a random-noise BGR frame (uint8)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_embed_detect_roundtrip():
    """Random frame, embed pilot, detect on same frame. Agreement >= 0.90."""
    frame = _random_frame()
    watermarked = embed_pilot(frame, CONTENT_ID_A, PEPPER_A)
    det = detect_pilot(watermarked, CONTENT_ID_A, PEPPER_A)
    assert det.detected, f"pilot not detected, agreement={det.agreement:.3f}"
    assert det.agreement >= 0.90, f"agreement={det.agreement:.3f} < 0.90"


def test_wrong_content_id_rejected():
    """Embed with content_id_A, detect with content_id_B → not detected."""
    frame = _random_frame()
    watermarked = embed_pilot(frame, CONTENT_ID_A, PEPPER_A)
    det = detect_pilot(watermarked, CONTENT_ID_B, PEPPER_A)
    assert not det.detected, f"wrong content_id accepted, agreement={det.agreement:.3f}"


def test_resolution_independence():
    """
    Embed+detect at 960×540, then embed+detect at 480×270.
    Both must succeed — validates that normalized block coordinates
    work at any resolution (same HMAC seed → same relative positions).
    """
    # Original resolution
    frame_hd = _random_frame(height=540, width=960, seed=42)
    wm_hd = embed_pilot(frame_hd, CONTENT_ID_A, PEPPER_A)
    det_hd = detect_pilot(wm_hd, CONTENT_ID_A, PEPPER_A)
    assert det_hd.detected, f"HD not detected, agreement={det_hd.agreement:.3f}"

    # Different resolution — same content_id/pepper → same relative block positions
    frame_sd = _random_frame(height=270, width=480, seed=99)
    wm_sd = embed_pilot(frame_sd, CONTENT_ID_A, PEPPER_A)
    det_sd = detect_pilot(wm_sd, CONTENT_ID_A, PEPPER_A)
    assert det_sd.detected, f"SD not detected, agreement={det_sd.agreement:.3f}"


def test_wrong_pepper_rejected():
    """Embed with pepper_A, detect with pepper_B → not detected."""
    frame = _random_frame()
    watermarked = embed_pilot(frame, CONTENT_ID_A, PEPPER_A)
    det = detect_pilot(watermarked, CONTENT_ID_A, PEPPER_B)
    assert not det.detected, f"wrong pepper accepted, agreement={det.agreement:.3f}"


def test_does_not_crash_on_tiny_frame():
    """8×8 frame. Should return detected == False gracefully."""
    frame = _random_frame(height=8, width=8)
    det = detect_pilot(frame, CONTENT_ID_A, PEPPER_A)
    assert not det.detected
