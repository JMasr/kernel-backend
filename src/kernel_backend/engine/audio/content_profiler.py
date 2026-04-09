"""
Deterministic audio content profiler for content-adaptive watermark routing.

Extracts 7 acoustic descriptors from a single shared STFT and classifies
audio into five content types via a hard-coded decision tree.

No file I/O — receives numpy arrays, returns ContentProfile.
No neural networks — pure arithmetic comparisons for determinism.

Determinism guarantees:
  - Fixed sr=22050, mono, n_fft=2048, hop_length=512, center=False
  - All features rounded to 4 decimal places
  - Classification via if-else tree (no floating-point accumulation order risk)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np

from kernel_backend.core.domain.content_profile import ContentProfile, ContentType

# ── Deterministic constants ─────────────────────────────────────────────────

_SR = 22050
_N_FFT = 2048
_HOP_LENGTH = 512
_CENTER = False
_DECIMALS = 4
_DESCRIPTOR_VERSION = "1.0.0"

# Subsampling strategy for long files
SUBSAMPLE_THRESHOLD_S = 600   # 10 minutes
SUBSAMPLE_N_SEGMENTS = 3
SUBSAMPLE_DURATION_S = 30

# ── Code hash (computed once at module load) ────────────────────────────────

_CODE_HASH: str = "sha256:" + hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()[:16]

# ── Decision tree thresholds ────────────────────────────────────────────────
# Derived from reference ranges in the design document (midpoint of ranges).
# These are conservative initial thresholds — a calibration script can refine
# them using GTZAN + ESC-50 datasets and export updated values.

_SILENCE_RMS_DB = -40.0
_AMBIENT_FLATNESS = 0.12
_AMBIENT_RMS_DB = -15.0
_AMBIENT_LOW_ENERGY_MAX = 0.30
_SPEECH_LOW_ENERGY_RATIO = 0.35
_SPEECH_ZCR_STD = 0.04
_MUSIC_CENTROID_HZ = 3500.0
_MUSIC_FLUX = 1.5
_MUSIC_ROLLOFF_HZ = 4000.0
_CONFIDENCE_THRESHOLD = 0.5


def _rms_to_db(rms_mean: float) -> float:
    """Convert mean RMS amplitude to dBFS."""
    return float(20.0 * np.log10(max(rms_mean, 1e-10)))


def _extract_features(samples: np.ndarray, sr: int) -> dict[str, float]:
    """Extract all 7 descriptor groups from a single shared STFT.

    Returns a dict of feature name → rounded float value.
    """
    # Shared STFT (magnitude spectrogram)
    S = np.abs(librosa.stft(samples, n_fft=_N_FFT, hop_length=_HOP_LENGTH, center=_CENTER))
    S_power = S ** 2

    features: dict[str, float] = {}

    # 1. Spectral flatness (geometric / arithmetic mean of power spectrum)
    sf = librosa.feature.spectral_flatness(S=S_power)
    features["spectral_flatness_mean"] = round(float(np.mean(sf)), _DECIMALS)

    # 2. ZCR std (speech has high variance due to voiced/unvoiced alternation)
    zcr = librosa.feature.zero_crossing_rate(
        y=samples, hop_length=_HOP_LENGTH, center=_CENTER
    )
    features["zcr_std"] = round(float(np.std(zcr)), _DECIMALS)

    # 3. Low-energy frame ratio (speech has 40-60% low-energy frames)
    rms = librosa.feature.rms(S=S)
    rms_mean = float(np.mean(rms))
    features["rms_mean"] = round(rms_mean, _DECIMALS)
    features["rms_db"] = round(_rms_to_db(rms_mean), _DECIMALS)
    low_energy_threshold = 0.5 * rms_mean
    features["low_energy_ratio"] = round(
        float(np.mean(rms.flatten() < low_energy_threshold)), _DECIMALS
    )

    # 4. MFCCs (mean and std of 13 coefficients — spectral envelope shape)
    S_db = librosa.power_to_db(S_power)
    mfcc = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = round(float(np.mean(mfcc[i])), _DECIMALS)
        features[f"mfcc_{i + 1}_std"] = round(float(np.std(mfcc[i])), _DECIMALS)

    # 5. Spectral centroid (speech: 1-3 kHz, music: 2-6 kHz)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    features["spectral_centroid_mean"] = round(float(np.mean(centroid)), _DECIMALS)

    # 6. Spectral flux (frame-to-frame spectral change; music > speech)
    flux = librosa.onset.onset_strength(S=S_db, sr=sr)
    features["spectral_flux_mean"] = round(float(np.mean(flux)), _DECIMALS)

    # 7. Spectral rolloff at 85% (frequency below which 85% of energy lies)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    features["spectral_rolloff_mean"] = round(float(np.mean(rolloff)), _DECIMALS)

    return features


def _classify(features: dict[str, float]) -> tuple[ContentType, float]:
    """Hard-coded decision tree for audio content classification.

    Returns (content_type, confidence).

    Hierarchy:
      1. Silence (energy threshold)
      2. Ambient/noise (spectral flatness + energy)
      3. Speech vs music (low-energy ratio, ZCR, centroid, flux)
    """
    rms_db = features["rms_db"]
    flatness = features["spectral_flatness_mean"]
    low_energy = features["low_energy_ratio"]
    zcr_std = features["zcr_std"]
    centroid = features["spectral_centroid_mean"]
    flux = features["spectral_flux_mean"]

    # Stage 1: Silence detection
    if rms_db < _SILENCE_RMS_DB:
        return "silence", 0.99

    # Stage 2: Ambient/noise detection
    # Ambient signals have high spectral flatness (noise-like) and low energy.
    # Also require low low-energy ratio (continuous, not speech-like pauses).
    if (flatness > _AMBIENT_FLATNESS
            and rms_db < _AMBIENT_RMS_DB
            and low_energy < _AMBIENT_LOW_ENERGY_MAX):
        confidence = min(0.99, 0.7 + (flatness - _AMBIENT_FLATNESS) * 2.0)
        return "ambient", round(confidence, 2)

    # Stage 3: Speech vs music discrimination
    # Use a weighted scoring system across multiple features.
    rolloff = features["spectral_rolloff_mean"]

    speech_score = 0.0
    music_score = 0.0

    # Low-energy ratio: speech has 40-60% low-energy frames (pauses).
    # But solo instruments with rests also have high ratios, so weight moderately.
    if low_energy > _SPEECH_LOW_ENERGY_RATIO:
        speech_score += 0.20
    else:
        music_score += 0.20

    # ZCR variance: speech alternates voiced/unvoiced -> high variance.
    # Music with varying dynamics also has moderate ZCR variance.
    if zcr_std > _SPEECH_ZCR_STD:
        speech_score += 0.15
    else:
        music_score += 0.10

    # Spectral centroid: speech concentrates 1-3 kHz, music spans wider.
    if centroid < _MUSIC_CENTROID_HZ:
        speech_score += 0.10
    else:
        music_score += 0.20

    # Spectral flux: music has higher frame-to-frame spectral change.
    if flux > _MUSIC_FLUX:
        music_score += 0.20
    else:
        speech_score += 0.05

    # Spectral rolloff: music distributes energy higher in the spectrum.
    if rolloff > _MUSIC_ROLLOFF_HZ:
        music_score += 0.15
    else:
        speech_score += 0.05

    # Spectral flatness: music tends to be slightly flatter than speech formants.
    if flatness > 0.10:
        music_score += 0.10
    elif flatness < 0.05:
        speech_score += 0.10

    # MFCC variance: speech has higher MFCC variance due to formant transitions.
    # Use MFCC 2 std (captures broad spectral shape changes).
    mfcc2_std = features.get("mfcc_2_std", 0.0)
    if mfcc2_std > 150.0:
        music_score += 0.10  # High MFCC variance can indicate rich timbral content
    elif mfcc2_std < 50.0:
        speech_score += 0.05

    total = speech_score + music_score
    if total < 1e-6:
        return "speech", 0.5

    if speech_score >= music_score:
        confidence = round(speech_score / total, 2)
        return "speech", confidence

    # Music detected — distinguish pop/rock from classical
    music_confidence = round(music_score / total, 2)

    # Classical tends to have lower centroid and lower flux than pop/rock
    if centroid < _MUSIC_CENTROID_HZ and flux < _MUSIC_FLUX:
        return "classical", music_confidence

    return "music", music_confidence


def profile_audio(samples: np.ndarray, sample_rate: int) -> ContentProfile:
    """Profile audio content from raw samples.

    Args:
        samples: float32 mono audio in [-1.0, 1.0]
        sample_rate: must be 22050 (canonical rate for profiling)

    Returns:
        Frozen ContentProfile with content_type, confidence, and features.
    """
    if sample_rate != _SR:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=_SR)

    # Ensure mono
    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    samples = samples.astype(np.float32)

    features = _extract_features(samples, _SR)
    content_type, confidence = _classify(features)

    # Low-confidence fallback to speech (safest default)
    if confidence < _CONFIDENCE_THRESHOLD:
        content_type = "speech"

    return ContentProfile(
        content_type=content_type,
        confidence=confidence,
        features=features,
        descriptor_version=_DESCRIPTOR_VERSION,
        code_hash=_CODE_HASH,
    )


def profile_audio_from_segments(
    segments: list[np.ndarray],
    sample_rate: int,
) -> ContentProfile:
    """Profile audio from pre-loaded segments (for subsampling strategy).

    The caller is responsible for loading the segments (3x30s for files >10min,
    or the full file for shorter content). This function concatenates and profiles.

    Args:
        segments: list of float32 mono arrays
        sample_rate: sample rate of all segments (must be consistent)

    Returns:
        ContentProfile from the concatenated audio.
    """
    if not segments:
        raise ValueError("At least one audio segment is required for profiling")
    combined = np.concatenate(segments)
    return profile_audio(combined, sample_rate)
