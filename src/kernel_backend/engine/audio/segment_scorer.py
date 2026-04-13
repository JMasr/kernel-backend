"""
Audio segment scoring for content-adaptive watermark placement.

Scores each 2-second audio segment on four dimensions that predict
watermark embedding quality and extraction reliability:
  - RMS energy (silence detection)
  - DWT band energy (energy in the actual target subband)
  - Spectral flatness (noise-like = good psychoacoustic masking)
  - Transient density (temporal masking opportunities)

The composite score ranks segments so the best N are selected for
RS symbol embedding, avoiding silence and low-energy regions.

Pure numpy — no file I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pywt


@dataclass(frozen=True)
class SegmentScore:
    """Quality score for a single audio segment."""
    index: int
    rms_db: float            # overall energy in dBFS
    band_energy_db: float    # energy in the DWT target subband (dBFS)
    spectral_flatness: float # 0..1, higher = more noise-like = better masking
    transient_density: float # fraction of frames with transient onsets
    composite: float         # weighted combination (0..1, higher = better)


# ── Scoring weights ─────────────────────────────────────────────────────────

_W_RMS = 0.30
_W_BAND = 0.30
_W_FLATNESS = 0.20
_W_TRANSIENT = 0.20

# Floor: segments below this are excluded from selection (near-silence)
DEFAULT_MIN_SCORE_DB = -40.0

# ── Helpers ──────────────────────────────────────────────────────────────────

_EPS = 1e-10  # avoid log(0)


def _rms_db(signal: np.ndarray) -> float:
    """RMS energy in dBFS."""
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, _EPS))


def _band_energy_db(
    signal: np.ndarray,
    dwt_level: int,
    target_subband: str,
) -> float:
    """Energy (dBFS) in the DWT target subband."""
    coeffs = pywt.wavedec(signal.astype(np.float64), "db4",
                          level=dwt_level, mode="periodization")
    band_idx = 0 if target_subband == "approximation" else -2
    band = coeffs[band_idx]
    rms = float(np.sqrt(np.mean(band ** 2)))
    return 20.0 * np.log10(max(rms, _EPS))


def _spectral_flatness(signal: np.ndarray) -> float:
    """Spectral flatness (Wiener entropy) in [0, 1]."""
    spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
    power = spectrum ** 2
    power = np.maximum(power, _EPS)
    log_mean = float(np.mean(np.log(power)))
    geo_mean = np.exp(log_mean)
    arith_mean = float(np.mean(power))
    if arith_mean < _EPS:
        return 0.0
    flatness = geo_mean / arith_mean
    return float(np.clip(flatness, 0.0, 1.0))


def _transient_density(signal: np.ndarray, sample_rate: int = 44100) -> float:
    """Fraction of short frames containing transient onsets.

    Uses a simple spectral flux onset detection: the fraction of 10ms frames
    where the energy increase exceeds 2× the median energy increase.
    """
    frame_len = max(int(sample_rate * 0.010), 1)  # 10ms frames
    n_frames = len(signal) // frame_len
    if n_frames < 2:
        return 0.0

    frames = signal[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.sum(frames.astype(np.float64) ** 2, axis=1)

    # Spectral flux: positive energy differences
    flux = np.maximum(np.diff(energies), 0.0)
    if len(flux) == 0:
        return 0.0

    median_flux = float(np.median(flux))
    threshold = max(median_flux * 2.0, _EPS)
    n_transients = int(np.sum(flux > threshold))
    return n_transients / len(flux)


# ── Public API ───────────────────────────────────────────────────────────────

def score_segment(
    segment: np.ndarray,
    sample_rate: int = 44100,
    dwt_level: int = 2,
    target_subband: str = "detail",
) -> SegmentScore:
    """Score a single 2s audio segment for watermark embedding quality.

    Returns a SegmentScore with raw metrics and a placeholder composite=0.0.
    Use score_segments() to compute population-normalized composites.
    """
    rms = _rms_db(segment)
    band = _band_energy_db(segment, dwt_level, target_subband)
    flatness = _spectral_flatness(segment)
    transient = _transient_density(segment, sample_rate)

    return SegmentScore(
        index=-1,  # set by caller
        rms_db=rms,
        band_energy_db=band,
        spectral_flatness=flatness,
        transient_density=transient,
        composite=0.0,  # computed after population normalization
    )


def score_segments(
    segments: Iterable[tuple[int, np.ndarray]],
    sample_rate: int = 44100,
    dwt_level: int = 2,
    target_subband: str = "detail",
) -> list[SegmentScore]:
    """Score all segments and compute population-normalized composites.

    Args:
        segments: iterable of (segment_index, audio_chunk) pairs.
        sample_rate: audio sample rate (Hz).
        dwt_level: primary DWT decomposition level.
        target_subband: "detail" or "approximation".

    Returns list of SegmentScores ordered by original index.
    """
    # Phase 1: compute raw metrics
    raw: list[tuple[int, float, float, float, float]] = []
    for idx, chunk in segments:
        sc = score_segment(chunk, sample_rate, dwt_level, target_subband)
        raw.append((idx, sc.rms_db, sc.band_energy_db,
                     sc.spectral_flatness, sc.transient_density))

    if not raw:
        return []

    # Phase 2: min-max normalize each dimension to [0, 1]
    arr = np.array([(r[1], r[2], r[3], r[4]) for r in raw], dtype=np.float64)

    def _normalize(col: np.ndarray) -> np.ndarray:
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < _EPS:
            return np.full_like(col, 0.5)
        return (col - lo) / (hi - lo)

    norm_rms = _normalize(arr[:, 0])
    norm_band = _normalize(arr[:, 1])
    norm_flat = _normalize(arr[:, 2])
    norm_trans = _normalize(arr[:, 3])

    composites = (
        _W_RMS * norm_rms
        + _W_BAND * norm_band
        + _W_FLATNESS * norm_flat
        + _W_TRANSIENT * norm_trans
    )

    # Phase 3: build final SegmentScore objects
    scores: list[SegmentScore] = []
    for i, (idx, rms, band, flat, trans) in enumerate(raw):
        scores.append(SegmentScore(
            index=idx,
            rms_db=rms,
            band_energy_db=band,
            spectral_flatness=flat,
            transient_density=trans,
            composite=float(composites[i]),
        ))

    scores.sort(key=lambda s: s.index)
    return scores


def scores_from_raw_metrics(
    raw: list[tuple[int, float, float, float, float]],
) -> list[SegmentScore]:
    """Build SegmentScores with population-normalized composites from raw metrics.

    Each tuple is (index, rms_db, band_energy_db, spectral_flatness, transient_density).
    Useful when metrics are collected during a streaming pass and scoring is deferred.
    """
    if not raw:
        return []

    arr = np.array([(r[1], r[2], r[3], r[4]) for r in raw], dtype=np.float64)

    def _normalize(col: np.ndarray) -> np.ndarray:
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < _EPS:
            return np.full_like(col, 0.5)
        return (col - lo) / (hi - lo)

    composites = (
        _W_RMS * _normalize(arr[:, 0])
        + _W_BAND * _normalize(arr[:, 1])
        + _W_FLATNESS * _normalize(arr[:, 2])
        + _W_TRANSIENT * _normalize(arr[:, 3])
    )

    scores = [
        SegmentScore(
            index=idx, rms_db=rms, band_energy_db=band,
            spectral_flatness=flat, transient_density=trans,
            composite=float(composites[i]),
        )
        for i, (idx, rms, band, flat, trans) in enumerate(raw)
    ]
    scores.sort(key=lambda s: s.index)
    return scores


def select_best(
    scores: list[SegmentScore],
    n_needed: int,
    min_score_db: float = DEFAULT_MIN_SCORE_DB,
) -> list[int]:
    """Return indices of the best n_needed segments for embedding.

    Segments below min_score_db (RMS) are excluded (silence avoidance).
    If fewer than n_needed segments pass the floor, returns all passing
    segments — RS error correction handles the shortfall as erasures.

    Output is sorted by segment index (preserves temporal order).
    """
    # Filter out near-silence
    passing = [s for s in scores if s.rms_db >= min_score_db]

    # Sort by composite score descending, pick top n_needed
    passing.sort(key=lambda s: s.composite, reverse=True)
    selected = passing[:n_needed]

    # Return sorted by original index for temporal order
    selected.sort(key=lambda s: s.index)
    return [s.index for s in selected]
