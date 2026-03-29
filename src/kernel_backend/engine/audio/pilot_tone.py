from __future__ import annotations

import numpy as np
import pywt
from scipy.signal.windows import tukey

from kernel_backend.core.domain.dsp_manifest import PRODUCTION_MANIFEST as _M
from kernel_backend.engine.codec.spread_spectrum import (
    chip_stream,
    pn_sequence,
)

_N_BITS = 48


def embed_pilot(
    samples: np.ndarray,        # float32, mono, [-1.0, 1.0]
    sample_rate: int,
    hash_48: int,               # 48-bit int derived from content_id
    global_pn_seed: int,        # HMAC(pepper, b"global_pilot_seed")[:8] as int
    chips_per_bit: int = _M.audio_pilot.chips_per_bit,
    target_snr_db: float = _M.audio_pilot.target_snr_db,
    perceptual_shaping: bool = True,
    temporal_shaping: bool = True,
    use_psychoacoustic: bool = False,
) -> np.ndarray:
    """
    Embed hash_48 (48 bits) into DWT approximation band (coeffs[0]).
    Process the full audio as a single block — pilot is not segmented.
    Apply Tukey window (alpha=0.1) to chip stream before adding.
    Tile the chip stream across the full approximation band to maximise
    effective processing gain.  Amplitude is scaled relative to band RMS
    so detect_pilot's normalised correlation is consistent regardless of
    host signal frequency content.
    Return samples of identical length to input.
    """
    orig_len = len(samples)

    # Build bit array from hash_48 (MSB first)
    bits = np.array([(hash_48 >> (47 - i)) & 1 for i in range(_N_BITS)], dtype=np.float32)

    n_chips = _N_BITS * chips_per_bit
    chips = chip_stream(bits, chips_per_bit, global_pn_seed)
    window = tukey(n_chips, alpha=0.1).astype(np.float32)
    chips_windowed = chips * window

    # DWT decomposition
    coeffs = pywt.wavedec(samples.astype(np.float64), "db4", level=2, mode="periodization")
    band = coeffs[0].copy()

    # Amplitude relative to band RMS: makes the normalised correlation in
    # detect_pilot consistent at approximately 10^(target_snr_db/20).
    band_rms = float(np.sqrt(np.mean(band ** 2)))
    if band_rms < 1e-10:
        band_rms = 1.0
    amplitude = band_rms * (10.0 ** (target_snr_db / 20.0))

    # MPEG-1 psychoacoustic clamp: cap amplitude at the Bark-domain masking
    # threshold for the approximation band (0–SR/4 Hz at level 2).
    # Uses the more restrictive of the two constraints.
    if use_psychoacoustic:
        from kernel_backend.engine.perceptual.psychoacoustic import (
            _BARK_EDGES_HZ,
            _compute_bark_power_thresholds,
            _hz_to_bark,
        )
        t_by_bark = _compute_bark_power_thresholds(samples, sample_rate, safety_margin_db=9.0)
        f_high_approx = float(sample_rate) / 4.0  # approx band at level 2: 0–SR/4 Hz
        bark_centers = _hz_to_bark((_BARK_EDGES_HZ[:-1] + _BARK_EDGES_HZ[1:]) / 2.0)
        bark_hi = float(_hz_to_bark(np.array([f_high_approx]))[0])
        relevant = bark_centers <= bark_hi
        if relevant.any():
            bark_amp = float(np.sqrt(max(float(t_by_bark[relevant].min()), 1e-20)))
        else:
            bark_amp = float(np.sqrt(max(float(t_by_bark.min()), 1e-20)))
        amplitude = min(amplitude, bark_amp)

    # Perceptual masking gain — concentrate energy where host signal masks it.
    if perceptual_shaping:
        from kernel_backend.engine.perceptual import masking_gain

        sg, tm = None, None
        if temporal_shaping:
            from kernel_backend.engine.perceptual.jnd_model import (
                silence_gate as compute_silence_gate,
                temporal_masking as compute_temporal_mask,
            )
            sg = compute_silence_gate(band, sample_rate, dwt_level=2)
            tm = compute_temporal_mask(band, sample_rate, dwt_level=2)

        gain = masking_gain(
            band, sample_rate, dwt_level=2,
            alpha=0.65, min_floor=0.05,
            silence_gate=sg, temporal_mask=tm,
        )
    else:
        gain = None

    # Tile chip stream across the full band for extra processing gain.
    tile_count = max(1, len(band) // n_chips)
    for rep in range(tile_count):
        start = rep * n_chips
        end = start + n_chips
        if end > len(band):
            seg_len = len(band) - start
            if gain is not None:
                band[start:] += chips_windowed[:seg_len] * amplitude * gain[start:]
            else:
                band[start:] += chips_windowed[:seg_len] * amplitude
        else:
            if gain is not None:
                band[start:end] += chips_windowed * amplitude * gain[start:end]
            else:
                band[start:end] += chips_windowed * amplitude

    coeffs[0] = band
    reconstructed = pywt.waverec(coeffs, "db4", mode="periodization")
    return _trim_or_pad(reconstructed, orig_len).astype(np.float32)


def detect_pilot(
    samples: np.ndarray,
    sample_rate: int,
    global_pn_seed: int,
    chips_per_bit: int = _M.audio_pilot.chips_per_bit,
    threshold: float = 1.5,
) -> int | None:
    """
    Decode the 48-bit hash from the DWT approximation band and return it
    if the mean per-bit Z-score is >= threshold; else return None.

    Uses Z-score detection which preserves processing gain from tiling:
    Z grows as sqrt(n_tiles), allowing reliable detection at much lower
    embedding amplitude than normalized correlation (which cancels tiling gain).

    For random (no-pilot) data, mean Z converges to ~0.80 (half-normal mean).
    For embedded data at -26 dB over 34s (122 tiles), mean Z ≈ 4.1.
    Default threshold 1.5 gives effectively zero false positive probability.
    """
    coeffs = pywt.wavedec(samples.astype(np.float64), "db4", level=2, mode="periodization")
    band = coeffs[0].astype(np.float32)

    n_chips = _N_BITS * chips_per_bit
    pn = pn_sequence(n_chips, global_pn_seed)
    window = tukey(n_chips, alpha=0.1).astype(np.float32)
    pn_windowed = (pn * window).astype(np.float64)

    if len(band) < chips_per_bit:
        return None

    tile_count = max(1, len(band) // n_chips)

    # Accumulate per-bit raw dot-products over all complete repetitions.
    per_bit_raw = np.zeros(_N_BITS, dtype=np.float64)
    n_reps = 0
    for rep in range(tile_count):
        start = rep * n_chips
        end = start + n_chips
        if end > len(band):
            break
        band_seg = band[start:end].astype(np.float64)
        n_reps += 1
        for i in range(_N_BITS):
            bs = i * chips_per_bit
            be = bs + chips_per_bit
            per_bit_raw[i] += float(np.dot(band_seg[bs:be], pn_windowed[bs:be]))

    if n_reps == 0:
        return None

    # Decode bits from sign of accumulated dot-products (MSB first).
    bits_arr = np.array([1.0 if r > 0 else 0.0 for r in per_bit_raw], dtype=np.float32)

    # Z-score detection: measures how many standard deviations each bit's
    # accumulated dot product is from zero (the null hypothesis).
    # Uses band variance as noise estimator; Z grows as sqrt(n_reps).
    tiled_len = n_reps * n_chips
    band_variance = float(np.var(band[:tiled_len].astype(np.float64)))
    if band_variance < 1e-10:
        band_variance = 1.0

    # Per-bit noise std accounts for windowed PN: sum(window²) per bit slice
    z_scores = np.zeros(_N_BITS, dtype=np.float64)
    for i in range(_N_BITS):
        bs = i * chips_per_bit
        be = bs + chips_per_bit
        window_sq_sum = float(np.sum(pn_windowed[bs:be] ** 2))
        noise_std = np.sqrt(band_variance * n_reps * window_sq_sum)
        if noise_std > 1e-10:
            z_scores[i] = abs(per_bit_raw[i]) / noise_std

    mean_z = float(np.mean(z_scores))
    if mean_z < threshold:
        return None

    # Pack decoded bits MSB-first into 48-bit int.
    result = 0
    for b in bits_arr.astype(int):
        result = (result << 1) | int(b)
    return result


def _trim_or_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) > target_len:
        return arr[:target_len]
    if len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)))
    return arr
