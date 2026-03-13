from __future__ import annotations

import hashlib
import hmac

import cv2
import numpy as np
from scipy.signal import stft

from kernel_backend.core.domain.watermark import SegmentFingerprint


def extract_hashes_from_stream(
    segment_stream,  # Generator yielding np.ndarray chunks
    sample_rate: int,
    key_material: bytes,
    pepper: bytes,
    segment_duration_s: float = 2.0,
    overlap: float = 0.5,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> list[SegmentFingerprint]:
    """
    Extract perceptual hashes from an iterative stream of audio chunks.
    This safely handles overlapping windows without loading the entire
    audio file into memory.
    """
    segment_samples = int(segment_duration_s * sample_rate)
    hop_samples = int(segment_samples * (1.0 - overlap))
    if hop_samples <= 0:
        hop_samples = segment_samples

    fingerprints = []
    overlap_buffer = np.zeros(0, dtype=np.float32)
    time_offset_samples = 0

    for chunk in segment_stream:
        overlap_buffer = np.concatenate((overlap_buffer, chunk))
        start = 0
        while start + segment_samples <= len(overlap_buffer):
            segment = overlap_buffer[start : start + segment_samples]
            hash_hex = _compute_hash(segment, sample_rate, key_material, pepper,
                                     f_min=f_min, f_max=f_max)
            fingerprints.append(SegmentFingerprint(
                time_offset_ms=int((time_offset_samples + start) * 1000 / sample_rate),
                hash_hex=hash_hex,
            ))
            start += hop_samples
            
        overlap_buffer = overlap_buffer[start:]
        time_offset_samples += start

    return fingerprints


def extract_hashes(
    samples: np.ndarray,
    sample_rate: int,
    key_material: bytes,
    pepper: bytes,
    segment_duration_s: float = 2.0,
    overlap: float = 0.5,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> list[SegmentFingerprint]:
    """
    Extract perceptual hashes with overlapping windows.
    overlap=0.5 means hop = segment_duration_s * 0.5

    f_min/f_max: mel filterbank frequency bounds (Hz).
    Defaults optimised for speech identity (300–8000 Hz).

    Overlap is necessary for real-world audio where segment boundaries
    may fall in silence or transients, causing hash instability.
    Verification uses min(hamming_distance) across all overlapping
    segments — at least one window will be aligned.
    """
    segment_samples = int(segment_duration_s * sample_rate)
    hop_samples = int(segment_samples * (1.0 - overlap))
    if hop_samples <= 0:
        hop_samples = segment_samples

    fingerprints = []
    start = 0
    while start + segment_samples <= len(samples):
        segment = samples[start : start + segment_samples]
        hash_hex = _compute_hash(segment, sample_rate, key_material, pepper,
                                 f_min=f_min, f_max=f_max)
        fingerprints.append(SegmentFingerprint(
            time_offset_ms=int(start * 1000 / sample_rate),
            hash_hex=hash_hex,
        ))
        start += hop_samples
    return fingerprints


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex strings."""
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    xor = a ^ b
    count = 0
    while xor:
        xor &= xor - 1
        count += 1
    return count


def _preemphasis(samples: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    High-pass pre-emphasis: y[t] = x[t] - coeff * x[t-1]
    Flattens the spectrum — compensates for low-frequency dominance
    in speech and real-world audio. Standard in speech processing.
    """
    return np.append(samples[0], samples[1:] - coeff * samples[:-1])


def _compute_hash(
    segment: np.ndarray,
    sample_rate: int,
    key_material: bytes,
    pepper: bytes,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> str:
    """
    Per-segment hash computation.

    DCT block shape: 12 frequency coefficients × 5 time coefficients = 60 dims
    Rationale: audio identity is more stable in frequency than in time.
    A 12×5 block is more robust to minor timing offsets than an 8×8 block.
    """
    # 1. Pre-emphasis
    segment = _preemphasis(segment)

    # 2. Log-mel spectrogram (speech-optimized frequency bounds,
    #    STFT-level spectral subtraction, energy-weighted time frames)
    log_mel = _log_mel_spectrogram(segment, sample_rate, f_min=f_min, f_max=f_max)

    # 3. Resize to 32×32
    resized = cv2.resize(
        log_mel.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA
    )

    # 4. Per-frequency-band mean removal (removes stationary noise floor per band)
    resized = resized - resized.mean(axis=1, keepdims=True)

    # 5. 2D DCT
    dct = cv2.dct(resized)

    # 6. Rectangular 12×5 block (freq × time) instead of 8×8
    dct_block = dct[:12, :5]       # 12 freq bins × 5 time bins = 60 values
    vector = dct_block.flatten()   # 60-dim vector

    # 7. L2 normalize before projection (preserves direction, not magnitude)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm > 1e-10:
        vector = vector / vector_norm

    # 8. Keyed projection (60×60 matrix)
    projection = _projection_matrix(key_material, pepper, dimension=len(vector))
    projected = projection @ vector

    # 9. Binarize
    median = float(np.median(projected))
    bits = projected >= median

    # 10. Pack to hex — 60 bits fits in 16 hex chars (pad to 64 bits)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _projection_matrix(
    key_material: bytes,
    pepper: bytes,
    dimension: int,
) -> np.ndarray:
    """Standard Gaussian random projection matrix."""
    seed_material = hmac.new(pepper, key_material, hashlib.sha256).digest()
    seed = int.from_bytes(seed_material[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dimension, dimension)).astype(np.float32)


def _log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
    f_min: float = 300.0,
    f_max: float = 8000.0,
    noise_floor_pct: float = 5.0,
    noise_oversub: float = 1.5,
    energy_weight: float = 2.0,
) -> np.ndarray:
    """
    Compute log-mel spectrogram with noise-robust processing.

    Steps:
      1. STFT power spectrum
      2. STFT-level spectral subtraction: subtract per-bin noise floor
         estimate (noise_floor_pct-th percentile × noise_oversub).
         Reduces stationary and slowly-varying noise before the mel filterbank.
      3. Mel filterbank in the cleaned power domain
      4. Log compression
      5. Energy-weighted time frames: upweight high-energy (high-SNR) frames
         and downweight low-energy (noise-dominated) frames.
         This makes the features robust to babble noise and silence frames.
    """
    nperseg = n_fft
    _, _, Zxx = stft(samples, fs=sample_rate, nperseg=nperseg,
                     noverlap=nperseg - hop_length, window="hann")
    power_spectrum = np.abs(Zxx) ** 2  # shape: (n_freqs, n_frames)

    # STFT-level spectral subtraction
    noise_floor = (
        np.percentile(power_spectrum, noise_floor_pct, axis=1, keepdims=True)
        * noise_oversub
    )
    power_spectrum = np.maximum(power_spectrum - noise_floor, 1e-10)

    mel_fb = _mel_filterbank(sample_rate, n_fft, n_mels,
                             f_min=f_min, f_max=min(f_max, sample_rate / 2.0))
    mel_spec = mel_fb @ power_spectrum
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)  # shape: (n_mels, n_frames)

    # Energy-weighted time frames: emphasize high-energy (speech-dominated) frames
    frame_energy = log_mel.max(axis=0)          # max log-mel per frame
    frame_energy = frame_energy - frame_energy.max()   # stabilize exp
    weights = np.exp(energy_weight * frame_energy)
    weights = weights / weights.sum()
    log_mel = log_mel * (weights * log_mel.shape[1])   # preserve overall scale

    return log_mel


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Build a mel filterbank matrix of shape (n_mels, n_fft//2 + 1)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_points = np.array([_mel_to_hz(m) for m in mel_points])

    filterbank = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        f_m_minus = freq_points[m - 1]
        f_m = freq_points[m]
        f_m_plus = freq_points[m + 1]
        for k, f in enumerate(fft_freqs):
            if f_m_minus <= f <= f_m:
                filterbank[m - 1, k] = (f - f_m_minus) / (f_m - f_m_minus)
            elif f_m < f <= f_m_plus:
                filterbank[m - 1, k] = (f_m_plus - f) / (f_m_plus - f_m)
    return filterbank


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
