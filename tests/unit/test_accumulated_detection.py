"""
Sprint 1 — S4: Accumulated bit decisions, Z-score detection, multi-band EGC.

Tests for:
- accumulated_bit_decisions (spread_spectrum)
- extract_segment and extract_symbol_segment with full tiling (wid_beacon)
- plan_audio_hopping with force_levels (hopping)
- multi-band roundtrip (embed_segment + extract_symbol_segment with extra_dwt_levels)
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pytest

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.audio.wid_beacon import (
    ERASURE_THRESHOLD_Z,
    embed_segment,
    extract_segment,
    extract_symbol_segment,
)
from kernel_backend.engine.codec.hopping import plan_audio_hopping
from kernel_backend.engine.codec.spread_spectrum import accumulated_bit_decisions, pn_sequence

SR = 44100
SEGMENT_S = 2.0
SEG_LEN = int(SR * SEGMENT_S)  # 88200 samples
PEPPER = b"sprint1-test-pepper-32bytes!!!!!"
CONTENT_ID = "sprint1-test-content-id"
PUBKEY = "-----BEGIN PUBLIC KEY-----\ntest\n-----END PUBLIC KEY-----\n"
N_BITS = 8
CHIPS_PER_BIT = 256


def _noise_segment(seed: int = 0) -> np.ndarray:
    """White noise segment at 44100 Hz, 2 s."""
    return np.random.default_rng(seed).standard_normal(SEG_LEN).astype(np.float32) * 0.3


def _pn_seed(i: int) -> int:
    msg = f"wid|{CONTENT_ID}|{PUBKEY}|{i}".encode()
    return int.from_bytes(hmac.new(PEPPER, msg, hashlib.sha256).digest()[:8], "big")


def _make_band(n: int, seed: int = 7) -> np.ndarray:
    """Gaussian band of length n."""
    return np.random.default_rng(seed).standard_normal(n).astype(np.float64)


# ---------------------------------------------------------------------------
# accumulated_bit_decisions — shape, determinism, edge cases
# ---------------------------------------------------------------------------


def test_accumulated_bit_decisions_shape() -> None:
    """bits.shape == (8,), z_scores.shape == (8,), n_tiles >= 1."""
    n_chips = N_BITS * CHIPS_PER_BIT
    band = _make_band(n_chips * 10)  # 10 tiles
    pn = pn_sequence(n_chips, seed=42)

    bits, z_scores, n_tiles = accumulated_bit_decisions(band, pn, N_BITS, CHIPS_PER_BIT)

    assert bits.shape == (N_BITS,), f"bits.shape={bits.shape}"
    assert z_scores.shape == (N_BITS,), f"z_scores.shape={z_scores.shape}"
    assert n_tiles >= 1


def test_accumulated_bit_decisions_deterministic() -> None:
    """Same inputs → same outputs on two consecutive calls."""
    n_chips = N_BITS * CHIPS_PER_BIT
    band = _make_band(n_chips * 5)
    pn = pn_sequence(n_chips, seed=13)

    bits1, z1, n1 = accumulated_bit_decisions(band, pn, N_BITS, CHIPS_PER_BIT)
    bits2, z2, n2 = accumulated_bit_decisions(band, pn, N_BITS, CHIPS_PER_BIT)

    np.testing.assert_array_equal(bits1, bits2)
    np.testing.assert_array_almost_equal(z1, z2)
    assert n1 == n2


def test_accumulated_bit_decisions_silence() -> None:
    """Band of zeros → no exception raised; z_scores == 0.0."""
    n_chips = N_BITS * CHIPS_PER_BIT
    band = np.zeros(n_chips * 5, dtype=np.float64)
    pn = pn_sequence(n_chips, seed=7)

    bits, z_scores, n_tiles = accumulated_bit_decisions(band, pn, N_BITS, CHIPS_PER_BIT)

    assert np.all(z_scores == 0.0), f"Expected all zeros, got {z_scores}"
    assert n_tiles >= 1


def test_z_score_grows_with_tiles() -> None:
    """
    Band 4× longer → Z-score grows (processing gain preserved).
    Uses a watermarked signal so the Z-score is non-trivial.
    """
    n_chips = N_BITS * CHIPS_PER_BIT
    pn = pn_sequence(n_chips, seed=3)
    amplitude = 0.05  # fixed, small watermark

    # 1-tile band
    band_1 = _make_band(n_chips, seed=0)
    band_1_wm = band_1.copy()
    band_1_wm += pn * amplitude  # embed one tile
    _, z1, _ = accumulated_bit_decisions(band_1_wm, pn, N_BITS, CHIPS_PER_BIT)

    # 4-tile band (same watermark, tiled)
    band_4 = np.tile(_make_band(n_chips, seed=0), 4)  # same noise 4 times
    band_4_wm = band_4.copy()
    for t in range(4):
        band_4_wm[t * n_chips:(t + 1) * n_chips] += pn * amplitude
    _, z4, _ = accumulated_bit_decisions(band_4_wm, pn, N_BITS, CHIPS_PER_BIT)

    assert float(np.mean(z4)) > float(np.mean(z1)), (
        f"Z-score did not grow with tiles: mean_z1={np.mean(z1):.4f}, mean_z4={np.mean(z4):.4f}"
    )


# ---------------------------------------------------------------------------
# Full tiling WID roundtrip — single band
# ---------------------------------------------------------------------------


def test_tiled_wid_roundtrip_white_noise() -> None:
    """
    embed_segment at −14 dB on white noise → extract_symbol_segment recovers symbol.
    BandConfig level=1, no extra_dwt_levels (single-band tiling).
    -14 dB gives Z ≈ 14.7 at level 1 (21 tiles × cpb=256), well above ERASURE_THRESHOLD_Z.
    """
    seg = _noise_segment(seed=42)
    bc = BandConfig(segment_index=0, coeff_positions=[], dwt_level=1)
    seed = _pn_seed(0)
    symbol_in = 0b10110101  # 181

    embedded = embed_segment(
        seg, rs_symbol=symbol_in, band_config=bc, pn_seed=seed,
        chips_per_bit=CHIPS_PER_BIT, target_snr_db=-14.0,
        perceptual_shaping=False,
    )
    symbol_out, mean_z = extract_symbol_segment(embedded, bc, seed, chips_per_bit=CHIPS_PER_BIT)

    assert symbol_out == symbol_in, (
        f"Symbol mismatch: embedded={symbol_in}, recovered={symbol_out}, mean_z={mean_z:.4f}"
    )
    assert mean_z >= ERASURE_THRESHOLD_Z


def test_tiled_wid_roundtrip_multi_band() -> None:
    """
    embed_segment at −14 dB with extra_dwt_levels=(2,) → extract_symbol_segment recovers symbol.
    BandConfig level=1, extra_dwt_levels=(2,) — exercises the EGC multi-band code path.
    """
    seg = _noise_segment(seed=99)
    bc = BandConfig(
        segment_index=0, coeff_positions=[], dwt_level=1, extra_dwt_levels=(2,)
    )
    seed = _pn_seed(0)
    symbol_in = 0b01001101  # 77

    embedded = embed_segment(
        seg, rs_symbol=symbol_in, band_config=bc, pn_seed=seed,
        chips_per_bit=CHIPS_PER_BIT, target_snr_db=-14.0,
        perceptual_shaping=False,
    )
    symbol_out, mean_z = extract_symbol_segment(embedded, bc, seed, chips_per_bit=CHIPS_PER_BIT)

    assert symbol_out == symbol_in, (
        f"Symbol mismatch: embedded={symbol_in}, recovered={symbol_out}, mean_z={mean_z:.4f}"
    )
    assert mean_z >= ERASURE_THRESHOLD_Z


# ---------------------------------------------------------------------------
# Erasure threshold: Z-score below/above ERASURE_THRESHOLD_Z
# ---------------------------------------------------------------------------


def test_erasure_threshold_no_watermark() -> None:
    """extract_segment on signal with no watermark → mean_z < ERASURE_THRESHOLD_Z."""
    seg = _noise_segment(seed=5)
    bc = BandConfig(segment_index=0, coeff_positions=[], dwt_level=1)
    seed = _pn_seed(0)

    mean_z = extract_segment(seg, bc, seed, chips_per_bit=CHIPS_PER_BIT)

    assert mean_z < ERASURE_THRESHOLD_Z, (
        f"Unwatermarked segment Z-score {mean_z:.4f} >= ERASURE_THRESHOLD_Z {ERASURE_THRESHOLD_Z}"
    )


def test_erasure_threshold_with_watermark() -> None:
    """extract_segment on watermarked signal at −14 dB → mean_z >= ERASURE_THRESHOLD_Z."""
    seg = _noise_segment(seed=6)
    bc = BandConfig(segment_index=0, coeff_positions=[], dwt_level=1)
    seed = _pn_seed(0)

    embedded = embed_segment(
        seg, rs_symbol=0b11001100, band_config=bc, pn_seed=seed,
        chips_per_bit=CHIPS_PER_BIT, target_snr_db=-14.0,
        perceptual_shaping=False,
    )
    mean_z = extract_segment(embedded, bc, seed, chips_per_bit=CHIPS_PER_BIT)

    assert mean_z >= ERASURE_THRESHOLD_Z, (
        f"Watermarked segment Z-score {mean_z:.4f} < ERASURE_THRESHOLD_Z {ERASURE_THRESHOLD_Z}"
    )


# ---------------------------------------------------------------------------
# plan_audio_hopping — force_levels
# ---------------------------------------------------------------------------


def test_plan_audio_hopping_force_levels() -> None:
    """With force_levels=[1, 2], all BandConfigs have dwt_level=1 and extra_dwt_levels==(2,)."""
    configs = plan_audio_hopping(
        n_segments=20,
        content_id=CONTENT_ID,
        author_pubkey=PUBKEY,
        pepper=PEPPER,
        force_levels=[1, 2],
    )

    assert len(configs) == 20
    for cfg in configs:
        assert cfg.dwt_level == 1, f"Expected dwt_level=1, got {cfg.dwt_level}"
        assert cfg.extra_dwt_levels == (2,), (
            f"Expected extra_dwt_levels==(2,), got {cfg.extra_dwt_levels}"
        )


def test_plan_audio_hopping_legacy() -> None:
    """With force_levels=None (legacy), extra_dwt_levels == () for all segments."""
    configs = plan_audio_hopping(
        n_segments=20,
        content_id=CONTENT_ID,
        author_pubkey=PUBKEY,
        pepper=PEPPER,
        force_levels=None,
    )

    assert len(configs) == 20
    for cfg in configs:
        assert cfg.extra_dwt_levels == (), (
            f"Legacy path should have empty extra_dwt_levels, got {cfg.extra_dwt_levels}"
        )
        # Legacy path: dwt_level is 1 or 2 (HMAC-derived)
        assert cfg.dwt_level in (1, 2), f"Unexpected dwt_level={cfg.dwt_level}"
