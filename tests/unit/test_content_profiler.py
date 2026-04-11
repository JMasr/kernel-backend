"""Unit tests for the content profiler engine module."""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.engine.audio.content_profiler import (
    _CODE_HASH,
    _SR,
    profile_audio,
    profile_audio_from_segments,
)


# ── Synthetic signal generators ─────────────────────────────────────────────

def _silence(duration_s: float = 2.0, sr: int = _SR) -> np.ndarray:
    """Near-zero signal (silence)."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _white_noise(duration_s: float = 5.0, sr: int = _SR, seed: int = 42) -> np.ndarray:
    """White noise with flat spectrum (ambient-like)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration_s)).astype(np.float32) * 0.1


def _speech_like(duration_s: float = 5.0, sr: int = _SR, seed: int = 42) -> np.ndarray:
    """Synthetic speech-like signal with pauses and formant-like structure.

    Alternates between voiced (low-freq buzz + formants) and silence to
    simulate the temporal structure of speech.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples, dtype=np.float32)

    # Alternating 200ms voiced / 100ms silence segments
    segment_len = int(0.2 * sr)
    pause_len = int(0.1 * sr)
    pos = 0
    while pos < n_samples:
        end = min(pos + segment_len, n_samples)
        # Fundamental + formants typical of speech (200, 800, 2500 Hz)
        voiced = (
            0.3 * np.sin(2 * np.pi * 200 * t[pos:end])
            + 0.2 * np.sin(2 * np.pi * 800 * t[pos:end])
            + 0.1 * np.sin(2 * np.pi * 2500 * t[pos:end])
        )
        signal[pos:end] = voiced.astype(np.float32)
        pos = end + pause_len  # skip pause (stays zero)

    # Add slight noise for realism
    signal += rng.standard_normal(n_samples).astype(np.float32) * 0.005
    return signal


# ── Tests ───────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Same input must always produce the exact same output."""

    def test_same_audio_same_profile(self):
        samples = _white_noise(duration_s=3.0, seed=123)
        p1 = profile_audio(samples.copy(), _SR)
        p2 = profile_audio(samples.copy(), _SR)
        assert p1 == p2

    def test_features_are_rounded(self):
        samples = _white_noise(duration_s=3.0)
        profile = profile_audio(samples, _SR)
        for key, value in profile.features.items():
            # All feature values should have at most 4 decimal places
            assert round(value, 4) == value, f"{key}={value} not rounded to 4 decimals"


class TestSilenceDetection:
    def test_zero_signal(self):
        profile = profile_audio(_silence(2.0), _SR)
        assert profile.content_type == "silence"
        assert profile.confidence >= 0.9

    def test_very_quiet_signal(self):
        # -60 dBFS signal (well below -40 dBFS threshold)
        samples = np.ones(int(_SR * 2), dtype=np.float32) * 1e-5
        profile = profile_audio(samples, _SR)
        assert profile.content_type == "silence"


class TestAmbientDetection:
    def test_white_noise_classified_as_ambient(self):
        """White noise has high spectral flatness -> ambient."""
        samples = _white_noise(duration_s=5.0, seed=42)
        profile = profile_audio(samples, _SR)
        assert profile.content_type == "ambient"
        # librosa spectral_flatness on finite white noise is ~0.12-0.20
        assert profile.features["spectral_flatness_mean"] > 0.10

    def test_ambient_has_higher_flatness_than_tonal(self):
        """White noise should have higher flatness than tonal content."""
        noise = _white_noise(duration_s=5.0, seed=99)
        tonal = _speech_like(duration_s=5.0)
        noise_profile = profile_audio(noise, _SR)
        tonal_profile = profile_audio(tonal, _SR)
        assert noise_profile.features["spectral_flatness_mean"] > tonal_profile.features["spectral_flatness_mean"]


class TestSpeechDetection:
    def test_speech_like_signal(self):
        """Synthetic speech with pauses should classify as speech."""
        samples = _speech_like(duration_s=5.0)
        profile = profile_audio(samples, _SR)
        assert profile.content_type == "speech"
        # Speech should have high low-energy ratio (pauses)
        assert profile.features["low_energy_ratio"] > 0.2


@pytest.mark.integration
class TestLibrosaExamples:
    """Test with real audio from librosa example files."""

    def test_trumpet_is_not_ambient_or_silence(self):
        """Trumpet is a solo instrument with rests — may classify as speech
        or music depending on rest ratio. Must not be ambient/silence."""
        librosa = pytest.importorskip("librosa")
        audio, sr = librosa.load(librosa.ex("trumpet"), sr=_SR, mono=True)
        profile = profile_audio(audio, sr)
        assert profile.content_type in ("music", "classical", "speech")

    def test_brahms_is_classical_or_music(self):
        librosa = pytest.importorskip("librosa")
        audio, sr = librosa.load(librosa.ex("brahms"), sr=_SR, mono=True)
        profile = profile_audio(audio, sr)
        assert profile.content_type in ("music", "classical")

    def test_vibeace_is_music(self):
        """Continuous electronic music with no pauses."""
        librosa = pytest.importorskip("librosa")
        audio, sr = librosa.load(librosa.ex("vibeace"), sr=_SR, mono=True)
        profile = profile_audio(audio, sr)
        assert profile.content_type in ("music", "classical")


class TestCodeHash:
    def test_code_hash_is_valid_hex(self):
        assert _CODE_HASH.startswith("sha256:")
        hex_part = _CODE_HASH[7:]
        assert len(hex_part) == 16
        int(hex_part, 16)  # should not raise

    def test_code_hash_stable(self):
        """Code hash should be the same across calls (computed once)."""
        from kernel_backend.engine.audio.content_profiler import _CODE_HASH as h2
        assert _CODE_HASH == h2


class TestProfileAudioFromSegments:
    def test_single_segment(self):
        samples = _white_noise(duration_s=5.0)
        p1 = profile_audio(samples, _SR)
        p2 = profile_audio_from_segments([samples], _SR)
        assert p1 == p2

    def test_multiple_segments_concatenated(self):
        """Multiple segments should be concatenated before profiling."""
        seg1 = _white_noise(duration_s=3.0, seed=1)
        seg2 = _white_noise(duration_s=3.0, seed=2)
        seg3 = _white_noise(duration_s=3.0, seed=3)
        profile = profile_audio_from_segments([seg1, seg2, seg3], _SR)
        assert profile.content_type is not None
        assert profile.descriptor_version == "1.0.0"

    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            profile_audio_from_segments([], _SR)


class TestProfileMetadata:
    def test_descriptor_version(self):
        samples = _white_noise(duration_s=3.0)
        profile = profile_audio(samples, _SR)
        assert profile.descriptor_version == "1.0.0"

    def test_features_contain_expected_keys(self):
        samples = _white_noise(duration_s=3.0)
        profile = profile_audio(samples, _SR)
        expected_keys = {
            "spectral_flatness_mean", "zcr_std", "low_energy_ratio",
            "rms_mean", "rms_db", "spectral_centroid_mean",
            "spectral_flux_mean", "spectral_rolloff_mean",
        }
        # Plus 26 MFCC keys (13 mean + 13 std)
        for i in range(1, 14):
            expected_keys.add(f"mfcc_{i}_mean")
            expected_keys.add(f"mfcc_{i}_std")
        assert expected_keys.issubset(set(profile.features.keys()))

    def test_resampling_from_44100(self):
        """Profiler should internally resample non-22050 Hz input."""
        samples_44k = _white_noise(duration_s=3.0, sr=44100)
        profile = profile_audio(samples_44k, 44100)
        assert profile.content_type is not None
