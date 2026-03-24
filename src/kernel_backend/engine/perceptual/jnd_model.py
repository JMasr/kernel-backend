"""
Temporal JND (Just Noticeable Difference) model — Phase 10.B.

Provides two perceptual gain functions that compose multiplicatively with
the spectral masking gain from ``psychoacoustic.masking_gain``:

``silence_gate``
    Suppresses watermark energy in silent / very-quiet passages where
    human hearing has zero masking.  Uses an adaptive percentile-based
    threshold with a smooth sigmoid transition to avoid click artifacts.

``temporal_masking``
    Models pre-masking suppression (~3 ms before transients) and
    post-masking boost (~30 ms after transients).  Pre-echoes are among
    the most audible watermark artifacts; forward masking after loud
    onsets provides free hiding capacity.

Both functions return per-coefficient gain arrays that are NOT
energy-normalised.  The caller (``masking_gain``) applies energy
normalisation after composing all gain layers.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Silence gate
# ---------------------------------------------------------------------------

def silence_gate(
    dwt_band: np.ndarray,
    sample_rate: int,
    dwt_level: int,
    gate_percentile: float = 10.0,
    threshold_scale: float = 3.0,
    gate_slope: float = 10.0,
    smooth_ms: float = 30.0,
) -> np.ndarray:
    """Compute a silence-suppression gate for a DWT band.

    Parameters
    ----------
    dwt_band : np.ndarray
        DWT coefficient array (approximation or detail band).
    sample_rate : int
        Original audio sample rate in Hz.
    dwt_level : int
        DWT decomposition level (1 or 2).
    gate_percentile : float
        Percentile of the smoothed envelope used to estimate the noise
        floor.  Default 10 — roughly the inter-word silence floor for
        typical speech.
    threshold_scale : float
        Multiplier applied to the percentile value to set the gate
        threshold above the noise floor.  Default 3.0 — ensures even
        the loudest coefficients in truly silent regions fall below
        threshold.
    gate_slope : float
        Steepness of the sigmoid transition.  Higher values give a
        sharper on/off boundary.  Default 10 (~10 dB transition zone).
    smooth_ms : float
        Envelope smoothing window in milliseconds.  Default 30 ms
        (slightly longer than Watson's 20 ms to avoid gate chatter).

    Returns
    -------
    np.ndarray
        Gate array in approximately [sigmoid_min, 1.0], same length as
        *dwt_band*.  NOT energy-normalised.
    """
    n = len(dwt_band)
    if n == 0:
        return np.ones(0, dtype=np.float32)

    # Smoothed absolute envelope
    coeff_rate = sample_rate / (2 ** dwt_level)
    window_len = max(1, int(smooth_ms * coeff_rate / 1000.0))
    window_len = min(window_len, n)

    envelope = np.abs(dwt_band).astype(np.float64)
    smoothed = uniform_filter1d(envelope, size=window_len, mode="reflect")

    # Adaptive threshold from percentile, scaled above noise floor
    p_low = float(np.percentile(smoothed, gate_percentile))
    p_high = float(np.percentile(smoothed, 100.0 - gate_percentile))

    # Dynamic range check: if the signal has little loudness variation
    # (e.g. broadband noise, sustained music), there is no meaningful
    # silence to gate.  Return transparent gate to avoid over-suppression.
    if p_high < 1e-10 or p_low > p_high * 0.3:
        return np.ones(n, dtype=np.float32)

    threshold = p_low * threshold_scale
    if threshold < 1e-10:
        return np.ones(n, dtype=np.float32)

    # Smooth sigmoid gate: 0 in silence → 1 in active signal
    exponent = -gate_slope * (smoothed - threshold) / threshold
    # Clip exponent to avoid overflow in exp()
    exponent = np.clip(exponent, -30.0, 30.0)
    gate = 1.0 / (1.0 + np.exp(exponent))

    return gate.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal masking (pre-echo suppression + forward masking boost)
# ---------------------------------------------------------------------------

def temporal_masking(
    dwt_band: np.ndarray,
    sample_rate: int,
    dwt_level: int,
    pre_mask_ms: float = 3.0,
    post_mask_ms: float = 30.0,
    transient_ratio: float = 3.0,
    pre_suppression: float = 0.10,
    post_boost: float = 0.3,
    smooth_ms: float = 5.0,
) -> np.ndarray:
    """Compute temporal masking gain for a DWT band.

    Detects transient onsets via short-time energy ratio, then:
    - Suppresses gain *before* transients (pre-echo removal).
    - Boosts gain *after* transients (exploits forward masking).

    Parameters
    ----------
    dwt_band : np.ndarray
        DWT coefficient array.
    sample_rate : int
        Original audio sample rate in Hz.
    dwt_level : int
        DWT decomposition level (1 or 2).
    pre_mask_ms : float
        Duration of pre-masking suppression before each transient.
    post_mask_ms : float
        Duration of post-masking boost after each transient.
    transient_ratio : float
        Minimum energy ratio to classify as a transient onset.
    pre_suppression : float
        Gain floor in pre-echo regions (0.10 = 10% of normal).
    post_boost : float
        Maximum boost factor added to gain=1.0 right after a transient.
        Decays exponentially over ``post_mask_ms``.
    smooth_ms : float
        Envelope smoothing window for onset detection.

    Returns
    -------
    np.ndarray
        Temporal gain array, same length as *dwt_band*.
        Values typically in [pre_suppression, 1.0 + post_boost].
        NOT energy-normalised.
    """
    n = len(dwt_band)
    if n == 0:
        return np.ones(0, dtype=np.float32)

    coeff_rate = sample_rate / (2 ** dwt_level)

    # 1. Short-time energy envelope
    frame_len = max(1, int(smooth_ms * coeff_rate / 1000.0))
    frame_len = min(frame_len, n)
    energy = uniform_filter1d(
        dwt_band.astype(np.float64) ** 2, size=frame_len, mode="reflect"
    )

    # 2. Energy ratio (forward-looking): how much energy jumps at each point
    shift = frame_len
    ratio = np.ones(n, dtype=np.float64)
    if shift < n:
        denominator = energy[:-shift].copy()
        denominator[denominator < 1e-20] = 1e-20
        ratio[shift:] = energy[shift:] / denominator

    # 3. Transient detection
    transient_mask = ratio > transient_ratio
    onset_indices = np.where(transient_mask)[0]

    if len(onset_indices) == 0:
        return np.ones(n, dtype=np.float32)

    # 4. Build gain envelope
    gain = np.ones(n, dtype=np.float64)

    pre_len = max(1, int(pre_mask_ms * coeff_rate / 1000.0))
    post_len = max(1, int(post_mask_ms * coeff_rate / 1000.0))
    decay_tau = max(1.0, post_len / 3.0)

    for idx in onset_indices:
        # Pre-masking: cosine ramp from pre_suppression to 1.0
        pre_start = max(0, idx - pre_len)
        if pre_start < idx:
            ramp_len = idx - pre_start
            # Cosine ramp: 0 at start → 1 at end
            t = np.linspace(0.0, 1.0, ramp_len)
            cosine_ramp = 0.5 * (1.0 - np.cos(np.pi * t))
            pre_gain = pre_suppression + (1.0 - pre_suppression) * cosine_ramp
            gain[pre_start:idx] = np.minimum(gain[pre_start:idx], pre_gain)

        # Post-masking: exponential decay boost
        post_end = min(n, idx + post_len)
        if idx < post_end:
            t_after = np.arange(post_end - idx, dtype=np.float64)
            boost = 1.0 + post_boost * np.exp(-t_after / decay_tau)
            gain[idx:post_end] = np.maximum(gain[idx:post_end], boost)

    return gain.astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostic utility
# ---------------------------------------------------------------------------

def compute_mean_rms_ratio(gain: np.ndarray) -> float:
    """Compute mean(gain) / RMS(gain) — the correlation efficiency factor.

    With RMS-normalised gain, the expected detection correlation is
    approximately ``(amplitude / band_rms) * mean_rms_ratio``.  For pilot
    tone at -14 dB, this means ``corr = 0.20 * ratio``.  The ratio must
    stay above 0.75 to keep pilot correlation above the 0.15 threshold.

    Parameters
    ----------
    gain : np.ndarray
        Gain array (before or after energy normalisation — the ratio is
        scale-invariant).

    Returns
    -------
    float
        The mean/RMS ratio, always in (0.0, 1.0] (1.0 for flat gain).
    """
    if len(gain) == 0:
        return 1.0
    mean_val = float(np.mean(np.abs(gain)))
    rms_val = float(np.sqrt(np.mean(gain.astype(np.float64) ** 2)))
    if rms_val < 1e-10:
        return 1.0
    return mean_val / rms_val
