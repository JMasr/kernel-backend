"""
Unit tests for engine/video/fingerprint.py — video perceptual fingerprint.

Uses random-noise frames and the frame-list extraction API.
"""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.engine.video.fingerprint import (
    extract_hashes_from_frames,
    hamming_distance,
)

KEY_A = b"author-public-key-material-AAA"
KEY_B = b"author-public-key-material-BBB"
PEPPER = b"system-pepper-bytes-padded-32b!"


def _random_video_frames(
    n_frames: int = 130,
    height: int = 270,
    width: int = 480,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate a video as a list of random-noise BGR frames."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(n_frames)]


def test_deterministic():
    """Extract hashes from same frames twice → identical hash_hex values."""
    frames = _random_video_frames()
    h1 = extract_hashes_from_frames(frames, KEY_A, PEPPER, fps=25.0)
    h2 = extract_hashes_from_frames(frames, KEY_A, PEPPER, fps=25.0)
    assert len(h1) > 0
    assert len(h1) == len(h2)
    for a, b in zip(h1, h2):
        assert a.hash_hex == b.hash_hex


def test_keyed():
    """Same frames, different key_material → at least one hash differs."""
    frames = _random_video_frames()
    h_a = extract_hashes_from_frames(frames, KEY_A, PEPPER, fps=25.0)
    h_b = extract_hashes_from_frames(frames, KEY_B, PEPPER, fps=25.0)
    assert any(a.hash_hex != b.hash_hex for a, b in zip(h_a, h_b))


def test_hamming_reflexive():
    """hamming_distance(h, h) == 0 for all extracted hashes."""
    frames = _random_video_frames()
    hashes = extract_hashes_from_frames(frames, KEY_A, PEPPER, fps=25.0)
    for h in hashes:
        assert hamming_distance(h.hash_hex, h.hash_hex) == 0


def test_zero_mean_applied():
    """
    Two frames with same mean brightness but different textures must produce
    different hashes. Without zero-mean normalization, DC dominance would
    cause them to hash identically.
    """
    rng = np.random.default_rng(42)
    # Frame A: random texture centered at 128
    frame_a = (128 + rng.integers(-20, 20, (270, 480, 3))).clip(0, 255).astype(np.uint8)
    # Frame B: different random texture, also centered at 128
    frame_b = (128 + rng.integers(-20, 20, (270, 480, 3))).clip(0, 255).astype(np.uint8)

    frames_a = [frame_a] * 130
    frames_b = [frame_b] * 130

    h_a = extract_hashes_from_frames(frames_a, KEY_A, PEPPER, fps=25.0)
    h_b = extract_hashes_from_frames(frames_b, KEY_A, PEPPER, fps=25.0)
    assert len(h_a) > 0 and len(h_b) > 0
    # Different content with same mean brightness → different hashes
    assert h_a[0].hash_hex != h_b[0].hash_hex, (
        "same-brightness frames hash identically — zero-mean may not be applied"
    )


def test_different_content_discriminated():
    """
    10 random-frame videos (different seeds) → max Hamming distance > 10.
    Validates that the keyed projection distinguishes different content.
    """
    all_hashes = []
    for seed in range(10):
        frames = _random_video_frames(seed=seed * 100)
        hashes = extract_hashes_from_frames(frames, KEY_A, PEPPER, fps=25.0)
        if hashes:
            all_hashes.append(hashes[0].hash_hex)

    assert len(all_hashes) >= 2
    max_dist = max(
        hamming_distance(all_hashes[i], all_hashes[j])
        for i in range(len(all_hashes))
        for j in range(i + 1, len(all_hashes))
    )
    assert max_dist > 10, f"max Hamming distance = {max_dist} (expected > 10)"
