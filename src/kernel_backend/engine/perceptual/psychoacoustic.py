"""
Psychoacoustic masking gain for DWT-domain audio watermarking.

Implements Watson's energy-adaptive model: the just-noticeable distortion
threshold at each DWT coefficient is proportional to the local signal
envelope raised to a power (alpha).  Watermark energy is concentrated
where the host signal provides masking and reduced where it would be
audible.

The gain array is energy-normalised so that sqrt(mean(gain^2)) ≈ 1.0,
preserving the total watermark power budget set by target_snr_db.

References
----------
Watson, A. B. (1993). DCTune: A technique for visual optimization of DCT
quantization matrices for individual images. *SID Digest*, 24, 946-949.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft as scipy_stft


# ---------------------------------------------------------------------------
# MPEG Psychoacoustic Model 1 (simplified) — Sprint 2
# ---------------------------------------------------------------------------

_N_BARK = 24
_BARK_EDGES_HZ = np.array([
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
    1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
    5300, 6400, 7700, 9500, 12000, 15500,
], dtype=np.float64)  # 25 edges → 24 bands

# Table: DWT detail level → (f_low, f_high) Hz at 44100 Hz sample rate
_DWT_LEVEL_FREQ_RANGES: dict[int, tuple[float, float]] = {
    1: (11025.0, 22050.0),   # cD1
    2: (5512.5,  11025.0),   # cD2
    3: (2756.25,  5512.5),   # cD3
    4: (1378.125, 2756.25),  # cD4
}


def _hz_to_bark(f: np.ndarray) -> np.ndarray:
    """Zwicker formula: f in Hz → Bark."""
    return (
        13.0 * np.arctan(0.76 * f / 1000.0)
        + 3.5 * np.arctan((f / 7500.0) ** 2)
    )


def _ath_db(f: np.ndarray) -> np.ndarray:
    """Absolute Threshold of Hearing in dB SPL (ISO 226 approximation)."""
    f_khz = np.clip(f, 20.0, 20000.0) / 1000.0
    return (
        3.64 * f_khz ** (-0.8)
        - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
        + 1e-3 * f_khz ** 4
    )


def _spreading_matrix(bark_centers: np.ndarray) -> np.ndarray:
    """
    Precompute N×N spreading function matrix.
    sf[i, j] = spreading from masker band i to probe band j (linear scale).
    Schroeder formula: SF(Δz) = 15.81 + 7.5·(Δz+0.474) − 17.5·√(1+(Δz+0.474)²) dB
    """
    n = len(bark_centers)
    sf = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dz = bark_centers[j] - bark_centers[i]
            sf_db = 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1.0 + (dz + 0.474) ** 2)
            sf[i, j] = 10.0 ** (sf_db / 10.0)  # linear
    return sf


def _compute_bark_power_thresholds(
    segment: np.ndarray,
    sample_rate: int,
    safety_margin_db: float = 3.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Internal: compute per-Bark-band embedding threshold (linear power scale).

    Returns t_embed: np.ndarray[float64], shape (N_BARK,)
    Used by compute_masking_thresholds (public) and bark_amplitude_profile_for_dwt_level.
    """
    # 1. STFT — vectorized over all frames at once
    # Clamp n_fft to segment length so noverlap < nperseg is guaranteed.
    n_fft_eff = max(1, min(n_fft, len(segment)))
    noverlap = max(0, n_fft_eff - hop_length)
    _, _, Zxx = scipy_stft(
        segment.astype(np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=n_fft_eff,
        noverlap=noverlap,
        padded=True,
    )
    power = np.abs(Zxx) ** 2  # (n_fft_eff//2+1, n_frames)
    freqs = np.linspace(0, sample_rate / 2, n_fft_eff // 2 + 1)

    # 2. Map FFT bins to Bark bands
    bark_centers = _hz_to_bark((_BARK_EDGES_HZ[:-1] + _BARK_EDGES_HZ[1:]) / 2.0)
    bin_to_bark = np.searchsorted(_BARK_EDGES_HZ, freqs, side="right") - 1
    bin_to_bark = np.clip(bin_to_bark, 0, _N_BARK - 1)

    # 3. Aggregate power per Bark band (vectorized over frames)
    bark_power = np.zeros((_N_BARK, power.shape[1]), dtype=np.float64)
    for b in range(_N_BARK):
        mask = bin_to_bark == b
        if mask.any():
            bark_power[b] = power[mask].sum(axis=0)

    # 4. Spreading function matrix multiply
    sf_matrix = _spreading_matrix(bark_centers)  # (N_BARK, N_BARK)
    spread_power = sf_matrix @ bark_power         # (N_BARK, n_frames)

    # 5. ATH in linear power, broadcast to all frames
    ath_linear = 10.0 ** (_ath_db(bark_centers) / 10.0)  # (N_BARK,)
    t_global = spread_power + ath_linear[:, np.newaxis]   # (N_BARK, n_frames)

    # 6. Conservative: minimum over all frames
    t_conservative = t_global.min(axis=1)  # (N_BARK,)

    # 7. Apply safety margin (reduce threshold to stay below masking threshold)
    safety_factor = 10.0 ** (-safety_margin_db / 10.0)
    t_embed = t_conservative * safety_factor  # (N_BARK,) in linear power

    return t_embed


def compute_masking_thresholds(
    segment: np.ndarray,
    sample_rate: int,
    safety_margin_db: float = 3.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute per-sample masking threshold array for a segment.

    Returns an amplitude threshold array of length len(segment), suitable
    for element-wise multiplication with a DWT chip stream. The returned
    value represents the global (minimum across all Bark bands) embedding
    amplitude limit.

    For per-DWT-band amplitude profiles with frequency-resolved thresholds,
    use bark_amplitude_profile_for_dwt_level() with the internal
    _compute_bark_power_thresholds() helper.

    Args:
        segment: float32 array, mono, [-1, 1]
        sample_rate: Hz
        safety_margin_db: dB margin below masking threshold (default 3.0)
        n_fft: STFT window size (default 2048 ≈ 46 ms at 44100 Hz)
        hop_length: STFT hop (default 512 ≈ 11.6 ms)

    Returns:
        threshold_amplitude: np.ndarray[float64], shape (len(segment),)
    """
    t_embed = _compute_bark_power_thresholds(
        segment, sample_rate, safety_margin_db, n_fft, hop_length
    )
    global_threshold_amplitude = float(np.sqrt(max(t_embed.min(), 1e-20)))
    return np.full(len(segment), global_threshold_amplitude, dtype=np.float64)


def bark_amplitude_profile_for_dwt_level(
    masking_thresholds_by_bark: np.ndarray,  # (N_BARK,) linear power
    dwt_level: int,
    n_coefficients: int,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Produce a per-coefficient amplitude profile for one DWT detail band.

    Maps the Bark-domain masking thresholds (linear power) to the frequency
    range of dwt_level, linearly interpolates across n_coefficients positions
    in that band, and returns an amplitude array (sqrt of power thresholds).

    Args:
        masking_thresholds_by_bark: per-Bark masking threshold (linear power),
            e.g. from _compute_bark_power_thresholds()
        dwt_level: 1–4
        n_coefficients: number of DWT coefficients in this band
        sample_rate: Hz (default 44100)

    Returns:
        amplitude_profile: np.ndarray[float64], shape (n_coefficients,)
    """
    f_low, f_high = _DWT_LEVEL_FREQ_RANGES.get(dwt_level, (0.0, sample_rate / 2.0))

    bark_centers = _hz_to_bark((_BARK_EDGES_HZ[:-1] + _BARK_EDGES_HZ[1:]) / 2.0)

    bark_lo = float(_hz_to_bark(np.array([f_low]))[0])
    bark_hi = float(_hz_to_bark(np.array([f_high]))[0])

    relevant = (bark_centers >= bark_lo) & (bark_centers <= bark_hi)
    if not relevant.any():
        # Fallback: use minimum threshold across all bands
        threshold_power = float(masking_thresholds_by_bark.min())
        return np.full(n_coefficients, max(np.sqrt(threshold_power), 1e-10), dtype=np.float64)

    # Linearly interpolate threshold across the n_coefficients positions
    relevant_powers = masking_thresholds_by_bark[relevant]
    relevant_barks = bark_centers[relevant]

    # Coefficient index → Bark position (linear mapping within DWT band)
    coeff_barks = np.linspace(bark_lo, bark_hi, n_coefficients)
    interpolated_power = np.interp(coeff_barks, relevant_barks, relevant_powers)

    return np.sqrt(np.clip(interpolated_power, 1e-20, None))


def masking_gain(
    dwt_band: np.ndarray,
    sample_rate: int,
    dwt_level: int,
    alpha: float = 0.5,
    smooth_ms: float = 20.0,
    min_floor: float = 0.15,
    silence_gate: np.ndarray | None = None,
    temporal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-coefficient masking gain for a DWT band.

    Parameters
    ----------
    dwt_band : np.ndarray
        DWT coefficient array (approximation or detail band).
    sample_rate : int
        Original audio sample rate in Hz (e.g. 44100).
    dwt_level : int
        DWT decomposition level (1 or 2). Used to derive the effective
        coefficient rate: ``sample_rate / 2**dwt_level``.
    alpha : float, optional
        Watson's power-law exponent.  0 → flat gain (no adaptation);
        1 → fully proportional to envelope.  Default 0.5 (square-root
        compression, moderate adaptation for DWT coefficients).
    smooth_ms : float, optional
        Moving-average window in milliseconds.  Smooths the envelope to
        avoid per-coefficient gain fluctuations.  Default 20 ms.
    min_floor : float, optional
        Minimum gain as a fraction of the peak gain.  Ensures even silent
        passages retain some watermark energy for robustness.  Default 0.15.
        Pilot tone callers should use ~0.05 (tiling + silence gate);
        WID beacon callers should use ~0.12 (RS correction headroom).
    silence_gate : np.ndarray or None, optional
        Pre-computed silence gate array from ``jnd_model.silence_gate()``.
        Composed multiplicatively before energy normalisation.
    temporal_mask : np.ndarray or None, optional
        Pre-computed temporal masking array from ``jnd_model.temporal_masking()``.
        Composed multiplicatively before energy normalisation.

    Returns
    -------
    np.ndarray
        Gain array of same length as *dwt_band*, energy-normalised so
        that ``sqrt(mean(gain**2)) ≈ 1.0``.
    """
    n = len(dwt_band)
    if n == 0:
        return np.ones(0, dtype=np.float32)

    # 1. Absolute-value envelope
    envelope = np.abs(dwt_band).astype(np.float64)

    # 2. Smoothing window length (in DWT coefficients)
    coeff_rate = sample_rate / (2 ** dwt_level)
    window_len = max(1, int(smooth_ms * coeff_rate / 1000.0))
    # Clamp to band length to avoid scipy edge issues
    window_len = min(window_len, n)

    smoothed = uniform_filter1d(envelope, size=window_len, mode="reflect")

    # 3. Watson's power-law masking threshold
    #    alpha=0 → gain is flat (all ones); alpha=1 → gain tracks envelope
    if alpha == 0.0:
        gain = np.ones(n, dtype=np.float64)
    else:
        gain = np.power(smoothed, alpha)

    # 4. Floor clamp — ensure minimum gain for robustness
    peak = np.max(gain)
    if peak > 0.0:
        floor_value = min_floor * peak
        np.maximum(gain, floor_value, out=gain)

    # 4b. Compose with silence gate (multiplicative, Phase 10.B)
    if silence_gate is not None:
        gain *= silence_gate[:n].astype(np.float64)

    # 4c. Compose with temporal mask (multiplicative, Phase 10.B)
    if temporal_mask is not None:
        gain *= temporal_mask[:n].astype(np.float64)

    # 5. Energy normalisation: sqrt(mean(gain^2)) → 1.0
    rms = np.sqrt(np.mean(gain ** 2))
    if rms > 1e-10:
        gain /= rms
    else:
        gain = np.ones(n, dtype=np.float64)

    return gain.astype(np.float32)
