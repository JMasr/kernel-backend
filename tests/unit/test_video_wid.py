"""
Unit tests for engine/video/wid_watermark.py — RS symbol embedding via QIM on 4×4 DCT.

Uses random-noise frames (never uniform-color frames).
"""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.engine.video.wid_watermark import (
    SegmentWIDResult,
    _coeff_set,
    embed_segment,
    extract_segment,
    MANDATORY_COEFFS,
    WID_AGREEMENT_THRESHOLD,
)

CONTENT_ID_A = "test-content-id-alpha"
CONTENT_ID_B = "test-content-id-beta"
PUBKEY_A = "test-pubkey-material-AAA"
PUBKEY_B = "test-pubkey-material-BBB"
PEPPER = b"test-pepper-32-bytes-padded-AAA!"
SYMBOL_BITS = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)


def _random_frames(n: int = 5, height: int = 540, width: int = 960, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(n)]


def test_embed_extract_roundtrip_single_segment():
    """5 random frames, embed 8-bit symbol, extract. Agreement >= 0.90 on clean."""
    frames = _random_frames()
    embedded = embed_segment(frames, SYMBOL_BITS, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    result = extract_segment(embedded, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    assert result.agreement >= 0.90, f"agreement={result.agreement:.3f} < 0.90"
    assert not result.erasure


def test_agreement_threshold():
    """Unsigned frames → agreement near 0.50 (random baseline)."""
    frames = _random_frames()
    result = extract_segment(frames, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    assert result.agreement < 0.60, f"unsigned agreement={result.agreement:.3f} too high"


def test_wrong_seed_scores_near_random():
    """Embed with (A, A), extract with (B, B) → agreement near 0.50."""
    frames = _random_frames()
    embedded = embed_segment(frames, SYMBOL_BITS, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    result = extract_segment(embedded, CONTENT_ID_B, PUBKEY_B, 0, PEPPER)
    assert result.agreement < 0.60, f"wrong-seed agreement={result.agreement:.3f} too high"


def test_mandatory_coefficients_always_present():
    """_coeff_set always includes (0,1) and (1,0) for 100 different segment indices."""
    for seg_idx in range(100):
        coeffs = _coeff_set(CONTENT_ID_A, PUBKEY_A, seg_idx, PEPPER)
        for mc in MANDATORY_COEFFS:
            assert mc in coeffs, f"segment {seg_idx}: mandatory coeff {mc} missing"


def test_unsigned_agreement_well_below_embedded():
    """
    Extract from unembedded frames → agreement well below embedded level.
    Random baseline is ~0.50 + O(1/sqrt(votes)), so with finite votes per bit
    the agreement can be 0.52–0.58 even for random data. The key property is
    the separation from embedded agreement (~0.95).
    """
    frames = _random_frames(seed=42)
    # Embedded: should be high
    embedded = embed_segment(frames, SYMBOL_BITS, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    result_emb = extract_segment(embedded, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    # Unsigned: should be near random
    result_raw = extract_segment(frames, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    # At least 30 percentage points separation
    gap = result_emb.agreement - result_raw.agreement
    assert gap >= 0.30, (
        f"insufficient gap: embedded={result_emb.agreement:.3f} "
        f"unsigned={result_raw.agreement:.3f} gap={gap:.3f}"
    )
