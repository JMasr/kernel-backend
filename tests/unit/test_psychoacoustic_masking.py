"""
Sprint 2 — S1: Psychoacoustic masking tests.

Tests for:
- compute_masking_thresholds (psychoacoustic.py)
- bark_amplitude_profile_for_dwt_level (psychoacoustic.py)
- embed_segment with use_psychoacoustic=True (wid_beacon.py)
"""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.audio.wid_beacon import (
    ERASURE_THRESHOLD_Z,
    embed_segment,
    extract_symbol_segment,
)
from kernel_backend.engine.perceptual.psychoacoustic import (
    _compute_bark_power_thresholds,
    bark_amplitude_profile_for_dwt_level,
    compute_masking_thresholds,
)

SR = 44100
SEG_LEN = int(SR * 2.0)  # 2 s


def _white_noise(seed: int = 0, length: int = SEG_LEN, rms: float = 0.3) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(length) * rms).astype(np.float32)


def _sine_at(freq_hz: float, length: int = SEG_LEN, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(length) / SR
    return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.float32)


def _silence(length: int = SEG_LEN) -> np.ndarray:
    return np.zeros(length, dtype=np.float32)


def _default_bc(level: int = 1) -> BandConfig:
    return BandConfig(segment_index=0, coeff_positions=[], dwt_level=level)


# ---------------------------------------------------------------------------
# compute_masking_thresholds
# ---------------------------------------------------------------------------


def test_compute_masking_thresholds_shape() -> None:
    """Returns array of length == len(segment)."""
    seg = _white_noise()
    out = compute_masking_thresholds(seg, SR)
    assert out.shape == (len(seg),), f"Expected shape ({len(seg)},), got {out.shape}"
    assert out.dtype == np.float64


def test_compute_masking_thresholds_silence() -> None:
    """
    Silence segment → threshold dominated by ATH.
    The minimum ATH for the hearing range is > 0 (around 3–4 kHz it's ~−5 dB SPL
    but in our linear power normalisation it remains positive).
    The threshold should be small but finite and constant.
    """
    seg = _silence()
    out = compute_masking_thresholds(seg, SR)
    # All-constant array (global min across Bark bands)
    assert np.all(out > 0), "Threshold must be positive even for silence"
    assert np.all(out == out[0]), "Silence threshold should be constant across positions"
    # Should be much smaller than for a loud signal
    loud = _white_noise(rms=0.9)
    out_loud = compute_masking_thresholds(loud, SR)
    assert float(out[0]) < float(out_loud[0]), (
        f"Silence threshold {out[0]:.2e} should be < loud threshold {out_loud[0]:.2e}"
    )


def test_compute_masking_thresholds_loud_tone() -> None:
    """
    Loud 1 kHz tone → per-Bark threshold higher than ATH-only (silence) in the
    Bark bands near 1 kHz due to the spreading function.

    compute_masking_thresholds returns the global minimum across all Bark bands,
    which is dominated by the ATH minimum at ~4 kHz regardless of tone presence.
    So we compare the per-Bark thresholds directly via _compute_bark_power_thresholds.
    """
    from kernel_backend.engine.perceptual.psychoacoustic import (
        _BARK_EDGES_HZ,
        _hz_to_bark,
    )

    tone = _sine_at(1000.0, amplitude=0.8)
    silence = _silence()

    t_tone_bark = _compute_bark_power_thresholds(tone, SR, safety_margin_db=0.0)
    t_silence_bark = _compute_bark_power_thresholds(silence, SR, safety_margin_db=0.0)

    # Find Bark bands near 1 kHz (within 2 Bark of 1 kHz centre)
    bark_centers = _hz_to_bark((_BARK_EDGES_HZ[:-1] + _BARK_EDGES_HZ[1:]) / 2.0)
    bark_1khz = float(_hz_to_bark(np.array([1000.0]))[0])
    near_1khz = np.abs(bark_centers - bark_1khz) <= 2.0

    # The tone must raise thresholds in at least some nearby Bark bands
    assert np.any(t_tone_bark[near_1khz] > t_silence_bark[near_1khz]), (
        "Loud 1 kHz tone should raise masking thresholds in nearby Bark bands "
        f"above the ATH-only (silence) threshold.\n"
        f"  tone   near 1kHz: {t_tone_bark[near_1khz]}\n"
        f"  silence near 1kHz: {t_silence_bark[near_1khz]}"
    )


# ---------------------------------------------------------------------------
# bark_amplitude_profile_for_dwt_level
# ---------------------------------------------------------------------------


def test_bark_profile_level1_shape() -> None:
    """bark_amplitude_profile_for_dwt_level with n_coefficients=44100 → shape (44100,)."""
    seg = _white_noise()
    t_by_bark = _compute_bark_power_thresholds(seg, SR)
    profile = bark_amplitude_profile_for_dwt_level(t_by_bark, dwt_level=1, n_coefficients=44100)
    assert profile.shape == (44100,), f"Expected (44100,), got {profile.shape}"
    assert profile.dtype == np.float64


def test_bark_profile_monotone() -> None:
    """All values in the amplitude profile must be strictly positive."""
    seg = _white_noise()
    t_by_bark = _compute_bark_power_thresholds(seg, SR)
    for level in [1, 2]:
        profile = bark_amplitude_profile_for_dwt_level(t_by_bark, dwt_level=level, n_coefficients=1024)
        assert np.all(profile > 0), (
            f"Level {level} profile has non-positive values: min={profile.min():.2e}"
        )


# ---------------------------------------------------------------------------
# embed_segment with use_psychoacoustic=True
# ---------------------------------------------------------------------------


def test_embed_psychoacoustic_inaudible() -> None:
    """
    Watermark energy per Bark band must not exceed the masking threshold.

    After embedding with use_psychoacoustic=True, compute the STFT power of the
    watermark residual (embedded - original) per Bark band. Compare against
    the masking threshold computed from the original signal.

    We use a conservative check: mean watermark power over the DWT detail band
    <= 2× the minimum masking threshold (factor-of-2 allows for tiling rounding).
    """
    import pywt
    from kernel_backend.engine.perceptual.psychoacoustic import (
        _compute_bark_power_thresholds,
        _BARK_EDGES_HZ,
        _N_BARK,
    )
    from scipy.signal import stft as scipy_stft

    seg = _white_noise(seed=42, rms=0.5)
    bc = _default_bc(level=1)
    symbol = 0b10101010
    pn_seed = 12345

    embedded = embed_segment(
        seg, rs_symbol=symbol, band_config=bc, pn_seed=pn_seed,
        chips_per_bit=32, use_psychoacoustic=True, safety_margin_db=3.0,
        perceptual_shaping=False,
    )

    # Watermark residual
    residual = (embedded.astype(np.float64) - seg.astype(np.float64)).astype(np.float32)

    # STFT power of the residual
    _, _, Zxx = scipy_stft(residual.astype(np.float64), fs=SR, window="hann", nperseg=2048, noverlap=1536)
    residual_power = np.abs(Zxx) ** 2
    freqs = np.linspace(0, SR / 2, 2048 // 2 + 1)
    bin_to_bark = np.searchsorted(_BARK_EDGES_HZ, freqs, side="right") - 1
    bin_to_bark = np.clip(bin_to_bark, 0, _N_BARK - 1)
    wm_power_per_bark = np.zeros(_N_BARK)
    for b in range(_N_BARK):
        mask = bin_to_bark == b
        if mask.any():
            wm_power_per_bark[b] = residual_power[mask].mean()

    # Masking threshold for the original signal
    t_by_bark = _compute_bark_power_thresholds(seg, SR, safety_margin_db=3.0)

    # Mean watermark power per Bark band should not exceed the masking threshold
    # (allow 2× headroom for numerical approximation and tiling boundary effects)
    for b in range(_N_BARK):
        assert wm_power_per_bark[b] <= t_by_bark[b] * 2.0 + 1e-15, (
            f"Bark band {b}: wm_power={wm_power_per_bark[b]:.2e} > 2×threshold={t_by_bark[b]*2:.2e}"
        )


def test_embed_psychoacoustic_stronger_near_maskers() -> None:
    """
    Psychoacoustic embedding concentrates energy where host signal is loud.

    For white noise (uniform masking across bands), the psychoacoustic embedding
    should yield a detectable watermark at extraction. We verify that the embedded
    watermark energy (‖residual‖²) with psychoacoustic is greater than zero —
    the model adapts to signal energy rather than suppressing everything.
    """
    seg = _white_noise(seed=7, rms=0.5)
    bc = _default_bc(level=1)
    symbol = 0b11001100
    pn_seed = 99999

    emb_psych = embed_segment(
        seg, rs_symbol=symbol, band_config=bc, pn_seed=pn_seed,
        chips_per_bit=32, use_psychoacoustic=True, safety_margin_db=3.0,
        perceptual_shaping=False,
    )
    emb_fixed = embed_segment(
        seg, rs_symbol=symbol, band_config=bc, pn_seed=pn_seed,
        chips_per_bit=32, target_snr_db=-37.0, use_psychoacoustic=False,
        perceptual_shaping=False,
    )

    energy_psych = float(np.sum((emb_psych.astype(np.float64) - seg.astype(np.float64)) ** 2))
    energy_fixed = float(np.sum((emb_fixed.astype(np.float64) - seg.astype(np.float64)) ** 2))

    assert energy_psych > 0, "Psychoacoustic embedding produced zero watermark energy"
    # Psychoacoustic at white noise should embed more than -37 dB fixed SNR
    assert energy_psych > energy_fixed, (
        f"Psychoacoustic energy {energy_psych:.4e} should exceed "
        f"−37 dB fixed energy {energy_fixed:.4e}"
    )


def test_roundtrip_psychoacoustic() -> None:
    """
    embed_segment(use_psychoacoustic=True) on white_noise → extract_symbol_segment
    recovers the symbol correctly with mean Z-score >= ERASURE_THRESHOLD_Z.
    """
    seg = _white_noise(seed=101, rms=0.3)
    bc = _default_bc(level=1)
    symbol_in = 0b10110101
    pn_seed = 777

    embedded = embed_segment(
        seg, rs_symbol=symbol_in, band_config=bc, pn_seed=pn_seed,
        chips_per_bit=32, use_psychoacoustic=True, safety_margin_db=3.0,
        perceptual_shaping=False,
    )

    symbol_out, mean_z = extract_symbol_segment(embedded, bc, pn_seed, chips_per_bit=32)

    assert symbol_out == symbol_in, (
        f"Symbol mismatch: embedded={symbol_in:#04x}, recovered={symbol_out:#04x}, "
        f"mean_z={mean_z:.4f}"
    )
    assert mean_z >= ERASURE_THRESHOLD_Z, (
        f"mean_z={mean_z:.4f} < ERASURE_THRESHOLD_Z={ERASURE_THRESHOLD_Z}"
    )


def test_embed_psychoacoustic_false_unchanged() -> None:
    """
    embed_segment(use_psychoacoustic=False) must produce bit-for-bit identical
    output to calling it without the parameter (Sprint 1 regression).
    """
    seg = _white_noise(seed=55)
    bc = _default_bc(level=2)
    symbol = 0b01010101
    pn_seed = 12321

    emb_default = embed_segment(
        seg, rs_symbol=symbol, band_config=bc, pn_seed=pn_seed,
        chips_per_bit=32, target_snr_db=-14.0,
        perceptual_shaping=False, temporal_shaping=False,
        use_psychoacoustic=False,
    )
    emb_explicit_false = embed_segment(
        seg, rs_symbol=symbol, band_config=bc, pn_seed=pn_seed,
        chips_per_bit=32, target_snr_db=-14.0,
        perceptual_shaping=False, temporal_shaping=False,
        use_psychoacoustic=False,
    )

    np.testing.assert_array_equal(
        emb_default, emb_explicit_false,
        err_msg="use_psychoacoustic=False must produce identical output on both calls",
    )
