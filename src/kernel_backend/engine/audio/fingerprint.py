from __future__ import annotations

import hashlib
import hmac
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
from scipy.signal import stft

from kernel_backend.core.domain.watermark import SegmentFingerprint

# A pepper-free intermediate: unit-normalized DCT block (direction only).
# Pairing with time_offset_ms lets callers project the same features against
# multiple (key_material, pepper) combinations without redecoding audio.
SegmentFeature = tuple[int, np.ndarray]


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
    features = list(iter_segment_features_from_stream(
        segment_stream, sample_rate,
        segment_duration_s=segment_duration_s, overlap=overlap,
        f_min=f_min, f_max=f_max,
    ))
    return project_features_to_fingerprints(features, key_material, pepper)


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
    features = list(iter_segment_features(
        samples, sample_rate,
        segment_duration_s=segment_duration_s, overlap=overlap,
        f_min=f_min, f_max=f_max,
    ))
    return project_features_to_fingerprints(features, key_material, pepper)


def iter_segment_features_from_stream(
    segment_stream: Iterable[np.ndarray],
    sample_rate: int,
    segment_duration_s: float = 2.0,
    overlap: float = 0.5,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> Iterator[SegmentFeature]:
    """Yield pepper-free per-segment feature vectors from a chunk stream.

    Pair with `project_features_to_fingerprints` to obtain hashes. Splitting
    the pipeline lets callers (notably the public verification endpoint)
    decode + feature-extract once, then project against many org peppers for
    cheap — the keyed projection is the only pepper-dependent step.
    """
    segment_samples = int(segment_duration_s * sample_rate)
    hop_samples = int(segment_samples * (1.0 - overlap))
    if hop_samples <= 0:
        hop_samples = segment_samples

    overlap_buffer = np.zeros(0, dtype=np.float32)
    time_offset_samples = 0

    for chunk in segment_stream:
        overlap_buffer = np.concatenate((overlap_buffer, chunk))
        start = 0
        while start + segment_samples <= len(overlap_buffer):
            segment = overlap_buffer[start : start + segment_samples]
            feature = _compute_features(segment, sample_rate, f_min=f_min, f_max=f_max)
            yield int((time_offset_samples + start) * 1000 / sample_rate), feature
            start += hop_samples

        overlap_buffer = overlap_buffer[start:]
        time_offset_samples += start


def iter_segment_features(
    samples: np.ndarray,
    sample_rate: int,
    segment_duration_s: float = 2.0,
    overlap: float = 0.5,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> Iterator[SegmentFeature]:
    """Buffered counterpart of `iter_segment_features_from_stream`."""
    segment_samples = int(segment_duration_s * sample_rate)
    hop_samples = int(segment_samples * (1.0 - overlap))
    if hop_samples <= 0:
        hop_samples = segment_samples

    start = 0
    while start + segment_samples <= len(samples):
        segment = samples[start : start + segment_samples]
        feature = _compute_features(segment, sample_rate, f_min=f_min, f_max=f_max)
        yield int(start * 1000 / sample_rate), feature
        start += hop_samples


def project_features_to_fingerprints(
    features: Iterable[SegmentFeature],
    key_material: bytes,
    pepper: bytes,
) -> list[SegmentFingerprint]:
    """Cheap keyed projection: reuse a single projection matrix across segments."""
    features_list = list(features)
    if not features_list:
        return []
    dimension = features_list[0][1].shape[0]
    projection = _projection_matrix(key_material, pepper, dimension=dimension)

    result: list[SegmentFingerprint] = []
    for time_offset_ms, feature in features_list:
        projected = projection @ feature
        median = float(np.median(projected))
        bits = projected >= median
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        result.append(SegmentFingerprint(
            time_offset_ms=time_offset_ms,
            hash_hex=f"{value:016x}",
        ))
    return result


def project_features_batch(
    features: Iterable[SegmentFeature],
    key_materials: Sequence[bytes],
    peppers: Sequence[bytes],
) -> list[list[SegmentFingerprint]]:
    """Project the same pepper-free features against N (key_material, pepper) pairs.

    Returns one fingerprint list per pair, in the same order as the input
    sequences. The multi-pepper public verify path uses this to replace the
    per-pepper loop over `project_features_to_fingerprints` with a single 3-D
    matmul that amortizes the feature matrix over all peppers.
    """
    if len(key_materials) != len(peppers):
        raise ValueError("key_materials and peppers must have the same length")

    features_list = list(features)
    n_peppers = len(peppers)
    if n_peppers == 0:
        return []
    if not features_list:
        return [[] for _ in range(n_peppers)]

    dimension = features_list[0][1].shape[0]
    time_offsets = [t for t, _ in features_list]
    feature_matrix = np.stack([f for _, f in features_list], axis=1)  # (dim, n_segments)

    projections = np.stack(
        [_projection_matrix(km, p, dimension=dimension)
         for km, p in zip(key_materials, peppers)],
        axis=0,
    )  # (n_peppers, dim, dim)

    projected = projections @ feature_matrix  # (n_peppers, dim, n_segments)

    medians = np.median(projected, axis=1, keepdims=True)  # (n_peppers, 1, n_segments)
    bits = (projected >= medians).astype(np.uint64)  # (n_peppers, dim, n_segments)

    powers = (np.uint64(1) << np.arange(dimension - 1, -1, -1, dtype=np.uint64))
    values = (bits * powers[None, :, None]).sum(axis=1, dtype=np.uint64)  # (n_peppers, n_segments)

    out: list[list[SegmentFingerprint]] = []
    for pi in range(n_peppers):
        per_pepper = [
            SegmentFingerprint(
                time_offset_ms=time_offsets[si],
                hash_hex=f"{int(values[pi, si]):016x}",
            )
            for si in range(len(time_offsets))
        ]
        out.append(per_pepper)
    return out


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


def _compute_features(
    segment: np.ndarray,
    sample_rate: int,
    f_min: float = 300.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """
    Pepper-free per-segment feature vector — steps 1–7 of the hash pipeline.

    Returns the unit-normalized 60-dim DCT block (12 freq × 5 time coefficients).
    The keyed projection + binarization (steps 8–10) run in
    `project_features_to_fingerprints`.

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
    vector = dct_block.flatten().astype(np.float32)

    # 7. L2 normalize before projection (preserves direction, not magnitude)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm > 1e-10:
        vector = vector / vector_norm

    return vector


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
