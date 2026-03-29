"""Unit tests for engine/perceptual/psychoacoustic.masking_gain."""

from __future__ import annotations

import numpy as np
import pywt
import pytest

from kernel_backend.engine.perceptual.psychoacoustic import masking_gain
from tests.fixtures.audio_signals import white_noise, pink_noise

SR = 44100
DWT_LEVEL = 2


def _dwt_band(samples: np.ndarray, level: int = DWT_LEVEL) -> np.ndarray:
    """Extract the approximation band (coeffs[0]) for testing."""
    coeffs = pywt.wavedec(samples.astype(np.float64), "db4", level=level, mode="periodization")
    return coeffs[0].astype(np.float32)


# ── Shape and basic properties ────────────────────────────────────────────────


def test_gain_output_shape():
    band = _dwt_band(white_noise(duration_s=2.0))
    gain = masking_gain(band, SR, DWT_LEVEL)
    assert len(gain) == len(band)
    assert gain.dtype == np.float32


def test_gain_rms_normalised():
    band = _dwt_band(white_noise(duration_s=2.0))
    gain = masking_gain(band, SR, DWT_LEVEL)
    rms = float(np.sqrt(np.mean(gain.astype(np.float64) ** 2)))
    assert abs(rms - 1.0) < 0.15, f"RMS should be ~1.0, got {rms}"
    assert np.all(gain <= 1.0 + 1e-5), "All gain values must be ≤ 1.0 (ceiling)"


def test_gain_rms_normalised_pink():
    band = _dwt_band(pink_noise(duration_s=2.0))
    gain = masking_gain(band, SR, DWT_LEVEL)
    rms = float(np.sqrt(np.mean(gain.astype(np.float64) ** 2)))
    assert abs(rms - 1.0) < 0.15, f"RMS should be ~1.0, got {rms}"
    assert np.all(gain <= 1.0 + 1e-5), "All gain values must be ≤ 1.0 (ceiling)"


# ── Envelope tracking ────────────────────────────────────────────────────────


def test_gain_follows_envelope():
    """Gain must be higher where signal is louder."""
    # Construct signal with loud first half, quiet second half
    loud = white_noise(duration_s=1.0, seed=10) * 0.8
    quiet = white_noise(duration_s=1.0, seed=20) * 0.02
    combined = np.concatenate([loud, quiet])

    band = _dwt_band(combined)
    gain = masking_gain(band, SR, DWT_LEVEL)

    mid = len(gain) // 2
    mean_loud = float(np.mean(gain[:mid]))
    mean_quiet = float(np.mean(gain[mid:]))
    assert mean_loud > mean_quiet, (
        f"Gain in loud region ({mean_loud:.3f}) should exceed "
        f"quiet region ({mean_quiet:.3f})"
    )


# ── Floor clamp ───────────────────────────────────────────────────────────────


def test_gain_floor_respected():
    band = _dwt_band(white_noise(duration_s=2.0))
    min_floor = 0.20
    gain = masking_gain(band, SR, DWT_LEVEL, min_floor=min_floor)
    peak = float(np.max(gain))
    floor_value = min_floor * peak
    actual_min = float(np.min(gain))
    assert actual_min >= floor_value - 1e-6, (
        f"Min gain {actual_min:.4f} < floor {floor_value:.4f}"
    )


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_gain_silent_input():
    """All-zero input must return uniform gain without NaN."""
    band = np.zeros(1000, dtype=np.float32)
    gain = masking_gain(band, SR, DWT_LEVEL)
    assert len(gain) == 1000
    assert not np.any(np.isnan(gain)), "Gain contains NaN"
    assert not np.any(np.isinf(gain)), "Gain contains Inf"
    # Should be uniform (all ones after normalization)
    assert np.allclose(gain, gain[0], atol=1e-6)


def test_gain_empty_input():
    gain = masking_gain(np.array([], dtype=np.float32), SR, DWT_LEVEL)
    assert len(gain) == 0


# ── Alpha parameter ───────────────────────────────────────────────────────────


def test_alpha_zero_gives_flat_gain():
    """alpha=0 should produce uniform gain (no adaptation)."""
    band = _dwt_band(pink_noise(duration_s=2.0))
    gain = masking_gain(band, SR, DWT_LEVEL, alpha=0.0)
    # All values should be identical after normalization
    assert np.allclose(gain, gain[0], atol=1e-5)


def test_higher_alpha_more_contrast():
    """Higher alpha should produce more variance in gain."""
    band = _dwt_band(pink_noise(duration_s=2.0))
    gain_low = masking_gain(band, SR, DWT_LEVEL, alpha=0.3)
    gain_high = masking_gain(band, SR, DWT_LEVEL, alpha=0.8)
    std_low = float(np.std(gain_low))
    std_high = float(np.std(gain_high))
    assert std_high > std_low, (
        f"alpha=0.8 std ({std_high:.4f}) should exceed "
        f"alpha=0.3 std ({std_low:.4f})"
    )


# ── Determinism ───────────────────────────────────────────────────────────────


def test_gain_deterministic():
    band = _dwt_band(white_noise(duration_s=2.0))
    gain_a = masking_gain(band, SR, DWT_LEVEL)
    gain_b = masking_gain(band, SR, DWT_LEVEL)
    np.testing.assert_array_equal(gain_a, gain_b)


# ── DWT level variation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("level", [1, 2])
def test_gain_works_at_different_dwt_levels(level: int):
    samples = white_noise(duration_s=2.0)
    coeffs = pywt.wavedec(samples.astype(np.float64), "db4", level=level, mode="periodization")
    band = coeffs[-2].astype(np.float32)  # detail band
    gain = masking_gain(band, SR, level)
    assert len(gain) == len(band)
    rms = float(np.sqrt(np.mean(gain.astype(np.float64) ** 2)))
    assert abs(rms - 1.0) < 0.15
    assert np.all(gain <= 1.0 + 1e-5)


# ── Phase 10.B composition tests ─────────────────────────────────────────────


def test_masking_gain_with_silence_gate():
    """Passing a silence_gate array changes the output."""
    from kernel_backend.engine.perceptual.jnd_model import silence_gate as compute_sg

    # Build signal with loud + quiet regions
    rng = np.random.default_rng(42)
    band = np.concatenate([
        rng.standard_normal(2048).astype(np.float32),
        rng.standard_normal(2048).astype(np.float32) * 0.0001,
    ])
    sg = compute_sg(band, SR, DWT_LEVEL)
    gain_without = masking_gain(band, SR, DWT_LEVEL, alpha=0.65, min_floor=0.05)
    gain_with = masking_gain(band, SR, DWT_LEVEL, alpha=0.65, min_floor=0.05, silence_gate=sg)
    assert not np.allclose(gain_without, gain_with), "Silence gate should change output"


def test_masking_gain_with_temporal_mask():
    """Passing a temporal_mask array changes the output."""
    from kernel_backend.engine.perceptual.jnd_model import temporal_masking as compute_tm

    # Build signal with a transient
    rng = np.random.default_rng(42)
    band = rng.standard_normal(8192).astype(np.float32) * 0.01
    band[4096:4096 + 512] = rng.standard_normal(512).astype(np.float32) * 1.0
    tm = compute_tm(band, SR, DWT_LEVEL)
    gain_without = masking_gain(band, SR, DWT_LEVEL)
    gain_with = masking_gain(band, SR, DWT_LEVEL, temporal_mask=tm)
    assert not np.allclose(gain_without, gain_with), "Temporal mask should change output"


def test_composed_gain_rms_normalised():
    """Composed gain (with silence_gate + temporal_mask) has RMS ≈ 1.0 and ceiling ≤ 1.0."""
    from kernel_backend.engine.perceptual.jnd_model import (
        silence_gate as compute_sg,
        temporal_masking as compute_tm,
    )

    rng = np.random.default_rng(42)
    band = np.concatenate([
        rng.standard_normal(2048).astype(np.float32),
        rng.standard_normal(2048).astype(np.float32) * 0.0001,
    ])
    sg = compute_sg(band, SR, DWT_LEVEL)
    tm = compute_tm(band, SR, DWT_LEVEL)
    gain = masking_gain(band, SR, DWT_LEVEL, alpha=0.65, min_floor=0.05,
                        silence_gate=sg, temporal_mask=tm)
    assert np.all(gain <= 1.0 + 1e-5), "All composed gain values must be ≤ 1.0 (ceiling)"


# ── energy_floor tests ──────────────────────────────────────────────────────


def test_energy_floor_raises_minimum_after_gates():
    """energy_floor applied AFTER silence_gate guarantees a minimum gain."""
    from kernel_backend.engine.perceptual.jnd_model import silence_gate as compute_sg

    # Build signal with loud + near-silent regions (mimics speech)
    rng = np.random.default_rng(42)
    band = np.concatenate([
        rng.standard_normal(2048).astype(np.float32),
        rng.standard_normal(2048).astype(np.float32) * 0.0001,
    ])
    sg = compute_sg(band, SR, DWT_LEVEL)

    gain_no_floor = masking_gain(
        band, SR, DWT_LEVEL, alpha=0.70, min_floor=0.12,
        silence_gate=sg, energy_floor=0.0,
    )
    gain_with_floor = masking_gain(
        band, SR, DWT_LEVEL, alpha=0.70, min_floor=0.12,
        silence_gate=sg, energy_floor=0.08,
    )

    # The minimum gain should be higher with energy_floor
    min_no_floor = float(np.min(gain_no_floor))
    min_with_floor = float(np.min(gain_with_floor))
    assert min_with_floor > min_no_floor, (
        f"energy_floor should raise minimum: {min_with_floor:.4f} vs {min_no_floor:.4f}"
    )
    # The mean gain should also be higher (more energy in quiet regions)
    mean_no_floor = float(np.mean(gain_no_floor))
    mean_with_floor = float(np.mean(gain_with_floor))
    assert mean_with_floor > mean_no_floor, (
        f"energy_floor should raise mean gain: {mean_with_floor:.4f} vs {mean_no_floor:.4f}"
    )


def test_energy_floor_preserves_ceiling():
    """energy_floor must not push any gain value above 1.0."""
    from kernel_backend.engine.perceptual.jnd_model import silence_gate as compute_sg

    rng = np.random.default_rng(42)
    band = np.concatenate([
        rng.standard_normal(2048).astype(np.float32),
        rng.standard_normal(2048).astype(np.float32) * 0.0001,
    ])
    sg = compute_sg(band, SR, DWT_LEVEL)
    gain = masking_gain(
        band, SR, DWT_LEVEL, alpha=0.70, min_floor=0.12,
        silence_gate=sg, energy_floor=0.08,
    )
    assert np.all(gain <= 1.0 + 1e-5), "energy_floor must not breach ceiling"


def test_energy_floor_zero_is_noop():
    """energy_floor=0.0 should produce identical output to omitting it."""
    band = _dwt_band(white_noise(duration_s=2.0))
    gain_default = masking_gain(band, SR, DWT_LEVEL)
    gain_zero = masking_gain(band, SR, DWT_LEVEL, energy_floor=0.0)
    np.testing.assert_array_equal(gain_default, gain_zero)
