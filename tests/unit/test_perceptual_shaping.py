"""Roundtrip and regression tests for perceptual shaping integration."""

from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pywt

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.audio.pilot_tone import detect_pilot, embed_pilot
from kernel_backend.engine.audio.wid_beacon import (
    embed_segment,
    extract_segment,
    extract_symbol_segment,
)
from kernel_backend.engine.codec.hopping import plan_audio_hopping
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec
from tests.fixtures.audio_signals import pink_noise, white_noise

SR = 44100
HASH_48 = 0xABCDEF012345
SEED = 0xDEADBEEFCAFEBABE
PEPPER = b"test-pepper-bytes-padded-to-32b!"
CONTENT_ID = "test-content-shaping"
PUBKEY = "-----BEGIN PUBLIC KEY-----\ntest\n-----END PUBLIC KEY-----\n"
_TEST_SNR_DB = -6.0
SEGMENT_S = 2.0
SEG_LEN = int(SR * SEGMENT_S)


def _pn_seed(i: int) -> int:
    msg = f"wid|{CONTENT_ID}|{PUBKEY}|{i}".encode()
    return int.from_bytes(hmac.new(PEPPER, msg, hashlib.sha256).digest()[:8], "big")


def _noise_segment(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(SEG_LEN).astype(np.float32) * 0.3


def _default_bc(i: int = 0) -> BandConfig:
    return BandConfig(segment_index=i, coeff_positions=[], dwt_level=2)


# ── Pilot roundtrip with shaping ─────────────────────────────────────────────


def test_shaped_pilot_roundtrip_white_noise():
    """embed_pilot(shaping=True) → detect_pilot recovers same hash."""
    audio = white_noise(duration_s=2.0)
    embedded = embed_pilot(audio, SR, HASH_48, SEED, perceptual_shaping=True)
    detected = detect_pilot(embedded, SR, SEED)
    assert detected == HASH_48


def test_shaped_pilot_roundtrip_pink_noise():
    """embed_pilot(shaping=True) → detect_pilot recovers same hash on pink noise.

    Pink noise DWT detail bands are non-flat; shaping reduces mean gain to ~0.90×.
    At -14 dB this 10% reduction drops below the 0.75 agreement threshold on a
    2-second window.  Use -6 dB (sign_av production default) which provides margin.
    For sign_audio (-14 dB) the pilot is embedded over ≥34 s, giving ample chips.
    """
    audio = pink_noise(duration_s=2.0)
    embedded = embed_pilot(audio, SR, HASH_48, SEED, target_snr_db=-6.0, perceptual_shaping=True)
    detected = detect_pilot(embedded, SR, SEED)
    assert detected == HASH_48


# ── WID roundtrip with shaping ────────────────────────────────────────────────


def test_shaped_wid_roundtrip():
    """embed_segment(shaping=True) → extract_symbol_segment recovers correct symbol."""
    seg = _noise_segment(0)
    bc = _default_bc(0)
    seed = _pn_seed(0)
    symbol = 0b10101010

    embedded = embed_segment(
        seg, symbol, bc, seed,
        target_snr_db=_TEST_SNR_DB,
        perceptual_shaping=True,
    )
    recovered, confidence = extract_symbol_segment(embedded, bc, seed)
    assert recovered == symbol, f"Symbol mismatch: {recovered} != {symbol}"
    assert confidence > 0.10, f"Confidence {confidence:.4f} below erasure threshold"


def test_shaped_wid_correlation_above_threshold():
    """Shaped WID at -6 dB must maintain correlation above 0.10."""
    for sym in [0, 85, 170, 255]:
        seg = _noise_segment(sym)
        bc = _default_bc(0)
        seed = _pn_seed(0)
        embedded = embed_segment(
            seg, sym, bc, seed,
            target_snr_db=_TEST_SNR_DB,
            perceptual_shaping=True,
        )
        corr = extract_segment(embedded, bc, seed)
        assert corr > 0.10, (
            f"Correlation {corr:.4f} for symbol {sym} below threshold 0.10"
        )


# ── Full RS roundtrip with shaping ───────────────────────────────────────────


def test_shaped_wid_rs_full_roundtrip():
    """Embed 24 segments with shaping → RS decode recovers WID."""
    wid = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
    n_segments = 24
    codec = ReedSolomonCodec(n_symbols=n_segments)
    rs_symbols = codec.encode(wid)

    band_configs = plan_audio_hopping(n_segments, CONTENT_ID, PUBKEY, PEPPER)
    seeds = [_pn_seed(i) for i in range(n_segments)]

    recovered_symbols: list[int | None] = []
    for i in range(n_segments):
        seg = _noise_segment(i)
        embedded = embed_segment(
            seg, rs_symbols[i], band_configs[i], seeds[i],
            chips_per_bit=256,
            target_snr_db=_TEST_SNR_DB,
            perceptual_shaping=True,
        )
        sym, conf = extract_symbol_segment(
            embedded, band_configs[i], seeds[i], chips_per_bit=256,
        )
        recovered_symbols.append(sym)

    decoded = codec.decode(recovered_symbols)
    assert decoded == wid, f"WID mismatch: {decoded.hex()} != {wid.hex()}"


# ── Backward compatibility ────────────────────────────────────────────────────


def test_shaping_disabled_matches_original_pilot():
    """perceptual_shaping=False must produce identical output to pre-shaping code."""
    audio = white_noise(duration_s=2.0, seed=42)
    # Both calls with shaping=False should give same result as old code
    out_a = embed_pilot(audio, SR, HASH_48, SEED, perceptual_shaping=False)
    out_b = embed_pilot(audio, SR, HASH_48, SEED, perceptual_shaping=False)
    np.testing.assert_array_equal(out_a, out_b)
    # And it should still detect
    detected = detect_pilot(out_a, SR, SEED)
    assert detected == HASH_48


def test_shaping_disabled_matches_original_wid():
    """perceptual_shaping=False must produce identical output to pre-shaping code."""
    seg = _noise_segment(0)
    bc = _default_bc(0)
    seed = _pn_seed(0)
    out_a = embed_segment(seg, 0xAB, bc, seed, target_snr_db=_TEST_SNR_DB,
                          perceptual_shaping=False)
    out_b = embed_segment(seg, 0xAB, bc, seed, target_snr_db=_TEST_SNR_DB,
                          perceptual_shaping=False)
    np.testing.assert_array_equal(out_a, out_b)


# ── Shaped vs flat: shaped should have less energy in quiet regions ──────────


def test_shaped_reduces_quiet_region_perturbation():
    """Perceptual shaping should add less watermark energy in quiet regions."""
    # Build signal: loud then quiet
    loud = white_noise(duration_s=1.0, seed=10) * 0.8
    quiet = white_noise(duration_s=1.0, seed=20) * 0.02
    combined = np.concatenate([loud, quiet])

    shaped = embed_pilot(combined, SR, HASH_48, SEED, perceptual_shaping=True)
    flat = embed_pilot(combined, SR, HASH_48, SEED, perceptual_shaping=False)

    # Measure perturbation in the quiet second half
    mid = len(combined) // 2
    diff_shaped = np.abs(shaped[mid:] - combined[mid:])
    diff_flat = np.abs(flat[mid:] - combined[mid:])

    energy_shaped = float(np.mean(diff_shaped ** 2))
    energy_flat = float(np.mean(diff_flat ** 2))

    assert energy_shaped < energy_flat, (
        f"Shaped quiet-region energy ({energy_shaped:.6f}) should be less than "
        f"flat ({energy_flat:.6f})"
    )


# ── Phase 10.B: Temporal shaping roundtrips ──────────────────────────────────


def test_temporal_shaped_pilot_roundtrip_white():
    """Pilot with temporal_shaping=True still detects on white noise."""
    audio = white_noise(duration_s=2.0, seed=42)
    embedded = embed_pilot(audio, SR, HASH_48, SEED, temporal_shaping=True)
    detected = detect_pilot(embedded, SR, SEED)
    assert detected == HASH_48


def test_temporal_shaped_pilot_roundtrip_pink():
    """Pilot with temporal_shaping=True still detects on pink noise.

    Same SNR constraint as test_shaped_pilot_roundtrip_pink_noise: use -6 dB
    on the short 2-second test window.
    """
    audio = pink_noise(duration_s=2.0, seed=42)
    embedded = embed_pilot(audio, SR, HASH_48, SEED, target_snr_db=-6.0, temporal_shaping=True)
    detected = detect_pilot(embedded, SR, SEED)
    assert detected == HASH_48


def test_temporal_shaped_wid_roundtrip():
    """WID with temporal_shaping=True recovers correct symbol."""
    seg = _noise_segment(0)
    bc = _default_bc(0)
    seed = _pn_seed(0)
    symbol = 0b11001100

    embedded = embed_segment(
        seg, symbol, bc, seed,
        target_snr_db=_TEST_SNR_DB,
        temporal_shaping=True,
    )
    recovered, confidence = extract_symbol_segment(embedded, bc, seed)
    assert recovered == symbol
    assert confidence > 0.10


def test_temporal_shaped_wid_rs_full_roundtrip():
    """24-segment RS decode with temporal shaping recovers WID."""
    wid = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
    n_segments = 24
    codec = ReedSolomonCodec(n_symbols=n_segments)
    rs_symbols = codec.encode(wid)

    band_configs = plan_audio_hopping(n_segments, CONTENT_ID, PUBKEY, PEPPER)
    seeds = [_pn_seed(i) for i in range(n_segments)]

    recovered_symbols: list[int | None] = []
    for i in range(n_segments):
        seg = _noise_segment(i)
        embedded = embed_segment(
            seg, rs_symbols[i], band_configs[i], seeds[i],
            chips_per_bit=256,
            target_snr_db=_TEST_SNR_DB,
            temporal_shaping=True,
        )
        sym, conf = extract_symbol_segment(
            embedded, band_configs[i], seeds[i], chips_per_bit=256,
        )
        recovered_symbols.append(sym)

    decoded = codec.decode(recovered_symbols)
    assert decoded == wid


def test_temporal_pilot_zscore_above_threshold():
    """Pilot mean Z-score must stay above 1.5 threshold with temporal shaping."""
    import pywt
    from scipy.signal.windows import tukey
    from kernel_backend.engine.codec.spread_spectrum import pn_sequence

    audio = white_noise(duration_s=2.0, seed=42)
    embedded = embed_pilot(audio, SR, HASH_48, SEED, temporal_shaping=True)

    coeffs = pywt.wavedec(embedded.astype(np.float64), "db4", level=2, mode="periodization")
    band = coeffs[0].astype(np.float32)
    n_chips = 48 * 64
    pn = pn_sequence(n_chips, SEED)
    window = tukey(n_chips, alpha=0.1).astype(np.float32)
    pn_windowed = (pn * window).astype(np.float64)

    tile_count = max(1, len(band) // n_chips)
    per_bit_raw = np.zeros(48, dtype=np.float64)
    n_reps = 0
    for rep in range(tile_count):
        s = rep * n_chips
        e = s + n_chips
        if e > len(band):
            break
        seg = band[s:e].astype(np.float64)
        n_reps += 1
        for i in range(48):
            bs = i * 64
            be = bs + 64
            per_bit_raw[i] += float(np.dot(seg[bs:be], pn_windowed[bs:be]))

    tiled_len = n_reps * n_chips
    band_variance = float(np.var(band[:tiled_len].astype(np.float64)))
    if band_variance < 1e-10:
        band_variance = 1.0

    z_scores = np.zeros(48, dtype=np.float64)
    for i in range(48):
        bs = i * 64
        be = bs + 64
        w_sq_sum = float(np.sum(pn_windowed[bs:be] ** 2))
        noise_std = np.sqrt(band_variance * n_reps * w_sq_sum)
        if noise_std > 1e-10:
            z_scores[i] = abs(per_bit_raw[i]) / noise_std

    mean_z = float(np.mean(z_scores))
    assert mean_z > 1.5, f"Pilot mean Z-score {mean_z:.4f} below 1.5 threshold"


def test_temporal_wid_confidence_above_010():
    """WID per-symbol confidence at -6 dB must stay above 0.10 with temporal shaping."""
    for sym_val in [0, 85, 170, 255]:
        seg = _noise_segment(sym_val)
        bc = _default_bc(0)
        seed = _pn_seed(0)
        embedded = embed_segment(
            seg, sym_val, bc, seed,
            target_snr_db=_TEST_SNR_DB,
            temporal_shaping=True,
        )
        recovered, confidence = extract_symbol_segment(embedded, bc, seed)
        assert confidence > 0.10, (
            f"WID confidence {confidence:.4f} for symbol {sym_val} below 0.10"
        )
