"""Tests for engine/perceptual/jnd_model.py — Phase 10.B."""

from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.engine.perceptual.jnd_model import (
    compute_mean_rms_ratio,
    silence_gate,
    temporal_masking,
)

SR = 44100
DWT_LEVEL = 2


# ── helpers ───────────────────────────────────────────────────────────────────

def _loud_quiet_band(n: int = 4096, seed: int = 42) -> np.ndarray:
    """Band with loud first half and near-silent second half."""
    rng = np.random.default_rng(seed)
    band = rng.standard_normal(n).astype(np.float32)
    band[n // 2 :] *= 0.0001  # near-silence (100× below loud)
    return band


def _transient_band(n: int = 8192, seed: int = 42) -> np.ndarray:
    """Band with a sharp onset at the midpoint."""
    rng = np.random.default_rng(seed)
    band = rng.standard_normal(n).astype(np.float32) * 0.01  # quiet baseline
    mid = n // 2
    band[mid : mid + n // 8] = rng.standard_normal(n // 8).astype(np.float32) * 1.0
    return band


def _white_noise_band(n: int = 4096, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32) * 0.3


# ── silence_gate tests ────────────────────────────────────────────────────────


class TestSilenceGate:
    def test_output_shape(self):
        band = _white_noise_band(4096)
        gate = silence_gate(band, SR, DWT_LEVEL)
        assert len(gate) == len(band)

    def test_values_in_range(self):
        band = _white_noise_band(4096)
        gate = silence_gate(band, SR, DWT_LEVEL)
        assert np.all(gate >= 0.0)
        assert np.all(gate <= 1.0)

    def test_suppresses_silence(self):
        band = _loud_quiet_band(4096)
        gate = silence_gate(band, SR, DWT_LEVEL)
        # Quiet half should have substantially lower gate values
        loud_mean = float(np.mean(gate[: len(band) // 2]))
        quiet_mean = float(np.mean(gate[len(band) // 2 :]))
        assert quiet_mean < loud_mean * 0.5

    def test_passes_loud_regions(self):
        band = _loud_quiet_band(4096)
        gate = silence_gate(band, SR, DWT_LEVEL)
        loud_mean = float(np.mean(gate[: len(band) // 4]))
        assert loud_mean > 0.7

    def test_smooth_transition(self):
        band = _loud_quiet_band(8192)
        gate = silence_gate(band, SR, DWT_LEVEL)
        # Median adjacent-sample diff should be small (smooth overall)
        diffs = np.abs(np.diff(gate))
        median_diff = float(np.median(diffs))
        p99_diff = float(np.percentile(diffs, 99))
        assert median_diff < 0.01
        assert p99_diff < 0.10

    def test_all_zero_input(self):
        band = np.zeros(4096, dtype=np.float32)
        gate = silence_gate(band, SR, DWT_LEVEL)
        assert len(gate) == 4096
        assert not np.any(np.isnan(gate))
        # All ones for zero input (threshold < 1e-10 → early return)
        assert np.allclose(gate, 1.0)

    def test_deterministic(self):
        band = _white_noise_band(4096, seed=77)
        g1 = silence_gate(band, SR, DWT_LEVEL)
        g2 = silence_gate(band, SR, DWT_LEVEL)
        np.testing.assert_array_equal(g1, g2)

    def test_adapts_to_signal_level(self):
        """Uniform-energy signals produce transparent gates regardless of level."""
        band_loud = _white_noise_band(4096, seed=42) * 10.0
        band_quiet = _white_noise_band(4096, seed=42) * 0.01
        gate_loud = silence_gate(band_loud, SR, DWT_LEVEL)
        gate_quiet = silence_gate(band_quiet, SR, DWT_LEVEL)
        # Both uniform noise signals should produce transparent (all ~1.0) gates
        assert np.allclose(gate_loud, 1.0)
        assert np.allclose(gate_quiet, 1.0)

    def test_empty_input(self):
        gate = silence_gate(np.array([], dtype=np.float32), SR, DWT_LEVEL)
        assert len(gate) == 0


# ── temporal_masking tests ────────────────────────────────────────────────────


class TestTemporalMasking:
    def test_output_shape(self):
        band = _white_noise_band(4096)
        mask = temporal_masking(band, SR, DWT_LEVEL)
        assert len(mask) == len(band)

    def test_neutral_for_stationary(self):
        band = _white_noise_band(8192, seed=42)
        mask = temporal_masking(band, SR, DWT_LEVEL)
        # Uniform-amplitude noise: no strong transients → mostly 1.0
        assert float(np.mean(np.abs(mask - 1.0))) < 0.15

    def test_pre_echo_suppression(self):
        band = _transient_band(8192)
        mask = temporal_masking(band, SR, DWT_LEVEL)
        mid = len(band) // 2
        # Pre-onset region should have suppressed gain
        coeff_rate = SR / (2 ** DWT_LEVEL)
        pre_len = max(1, int(3.0 * coeff_rate / 1000.0))
        pre_region = mask[max(0, mid - pre_len) : mid]
        if len(pre_region) > 0:
            assert float(np.mean(pre_region)) < 0.7

    def test_post_onset_boost(self):
        band = _transient_band(8192)
        mask = temporal_masking(band, SR, DWT_LEVEL)
        mid = len(band) // 2
        # Post-onset region should have boosted gain
        coeff_rate = SR / (2 ** DWT_LEVEL)
        post_len = max(1, int(10.0 * coeff_rate / 1000.0))
        post_region = mask[mid : mid + post_len]
        if len(post_region) > 0:
            assert float(np.max(post_region)) > 1.05

    def test_deterministic(self):
        band = _transient_band(4096, seed=77)
        m1 = temporal_masking(band, SR, DWT_LEVEL)
        m2 = temporal_masking(band, SR, DWT_LEVEL)
        np.testing.assert_array_equal(m1, m2)

    def test_empty_input(self):
        mask = temporal_masking(np.array([], dtype=np.float32), SR, DWT_LEVEL)
        assert len(mask) == 0


# ── compute_mean_rms_ratio tests ─────────────────────────────────────────────


class TestMeanRmsRatio:
    def test_flat_gain_returns_one(self):
        gain = np.ones(1000, dtype=np.float32) * 2.5
        ratio = compute_mean_rms_ratio(gain)
        assert abs(ratio - 1.0) < 0.01

    def test_variable_gain_less_than_one(self):
        gain = np.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], dtype=np.float32)
        ratio = compute_mean_rms_ratio(gain)
        assert ratio < 1.0

    def test_empty_returns_one(self):
        ratio = compute_mean_rms_ratio(np.array([], dtype=np.float32))
        assert ratio == 1.0
