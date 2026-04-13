from __future__ import annotations

import numpy as np
import pywt

from kernel_backend.core.domain.dsp_manifest import PRODUCTION_MANIFEST as _M
from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.codec.spread_spectrum import (
    accumulated_bit_decisions,
    chip_stream,
    normalized_correlation,  # noqa: F401 — kept for pilot_tone regression tests
    pn_sequence,
)

# Lazy-imported inside embed_segment when use_psychoacoustic=True to avoid
# pulling scipy.signal into the module-level import graph unconditionally.
# bark_amplitude_profile_for_dwt_level, _compute_bark_power_thresholds

# Z-score mean < ERASURE_THRESHOLD_Z → segment marked as erasure for RS decoder.
ERASURE_THRESHOLD_Z: float = _M.audio_wid.erasure_threshold_z


def embed_segment(
    segment: np.ndarray,        # float32, mono, one fixed-duration slice
    rs_symbol: int,             # single RS symbol, value 0–255
    band_config: BandConfig,    # from plan_audio_hopping
    pn_seed: int,               # HMAC(pepper, f"wid|{content_id}|{pubkey}|{i}")[:8]
    chips_per_bit: int = 256,
    target_snr_db: float = -14.0,
    sample_rate: int = 44100,
    perceptual_shaping: bool = True,
    temporal_shaping: bool = True,
    use_psychoacoustic: bool = False,
    safety_margin_db: float = 3.0,
) -> np.ndarray:
    """
    Spread rs_symbol (8 bits) as a tiled DSSS chip stream into coeffs[-2].

    Supports multi-band embedding via band_config.extra_dwt_levels.
    If extra_dwt_levels is non-empty, embeds the same symbol in each additional
    DWT level with amplitude scaled by 1/sqrt(n_bands) to keep total energy constant.

    Full tiling: uses all available DWT coefficients (not just one chip period).
    """
    orig_len = len(segment)

    # Build 8-bit array (MSB first)
    bits = np.array([(rs_symbol >> (7 - i)) & 1 for i in range(8)], dtype=np.float32)
    chips = chip_stream(bits, chips_per_bit, pn_seed)
    n_chips = len(chips)

    all_levels = (band_config.dwt_level,) + band_config.extra_dwt_levels
    n_bands = len(all_levels)

    # Pre-compute Bark-domain masking thresholds once per segment (Sprint 2).
    # This avoids running the STFT inside the per-level loop.
    t_by_bark = None
    if use_psychoacoustic:
        from kernel_backend.engine.perceptual.psychoacoustic import (
            _compute_bark_power_thresholds,
            bark_amplitude_profile_for_dwt_level,
        )
        t_by_bark = _compute_bark_power_thresholds(segment, sample_rate, safety_margin_db)

    # Sequential embedding: each level is embedded on the result of the previous
    # level's DWT→modify→IDWT cycle.  DWT is linear, so the watermark from level N
    # passes through the level N+1 decomposition/reconstruction unmodified (it
    # occupies frequency bands that the deeper decomposition doesn't resplit).
    # This avoids the previous accumulate-and-divide bug where dividing by n_bands
    # attenuated the watermark by an additional factor of 1/n_bands on top of the
    # intended 1/sqrt(n_bands) energy conservation scaling.
    result = segment.astype(np.float64)

    # Subband selection: "detail" → coeffs[-2] (legacy), "approximation" → coeffs[0]
    _band_idx = 0 if band_config.target_subband == "approximation" else -2

    for level in all_levels:
        coeffs = pywt.wavedec(result, "db4", level=level, mode="periodization")
        band = coeffs[_band_idx].copy()

        band_rms = float(np.sqrt(np.mean(band ** 2)))
        if band_rms < 1e-10:
            band_rms = 1.0
        amplitude = band_rms * (10.0 ** (target_snr_db / 20.0)) / np.sqrt(n_bands)

        tile_count = max(1, len(band) // n_chips)

        if use_psychoacoustic:
            # Per-coefficient amplitude profile from MPEG-1 psychoacoustic model.
            # Divided by sqrt(n_bands) to conserve total energy across multi-band embedding.
            amplitude_profile = (
                bark_amplitude_profile_for_dwt_level(
                    t_by_bark, level, len(band), sample_rate
                )
                / np.sqrt(n_bands)
            )
            # Robustness floor: the bark model sets a psychoacoustic ceiling (maximum
            # amplitude before the watermark becomes audible), but it can fall below
            # what AAC/MP3 codecs destroy.  Clamp from below at the target_snr_db
            # amplitude so codec survival is always guaranteed regardless of signal level.
            amplitude_profile = np.maximum(amplitude_profile, amplitude)

            # Compose temporal shaping: silence gate + temporal mask suppress the
            # watermark in silent/pre-transient passages even when bark profile is
            # active.  Without this, the codec-survival floor produces constant
            # energy through silence → audible tonal artefact.
            if temporal_shaping:
                from kernel_backend.engine.perceptual.jnd_model import (
                    silence_gate as compute_silence_gate,
                    temporal_masking as compute_temporal_mask,
                )
                sg = compute_silence_gate(band, sample_rate, dwt_level=level)
                tm = compute_temporal_mask(band, sample_rate, dwt_level=level)
                amplitude_profile *= sg[: len(amplitude_profile)]
                tm_capped = np.minimum(tm[: len(amplitude_profile)], 1.0)
                amplitude_profile *= tm_capped

            # tile_count * n_chips may be < len(band) when not evenly divisible;
            # use tile_count+1 and slice to exactly len(band).
            chips_tiled = np.tile(chips, tile_count + 1)[: len(band)]
            band += chips_tiled * amplitude_profile
        elif perceptual_shaping:
            from kernel_backend.engine.perceptual import masking_gain

            sg, tm = None, None
            if temporal_shaping:
                from kernel_backend.engine.perceptual.jnd_model import (
                    silence_gate as compute_silence_gate,
                    temporal_masking as compute_temporal_mask,
                )
                sg = compute_silence_gate(band, sample_rate, dwt_level=level)
                tm = compute_temporal_mask(band, sample_rate, dwt_level=level)

            gain = masking_gain(
                band, sample_rate, dwt_level=level,
                alpha=0.70, min_floor=0.12,
                silence_gate=sg, temporal_mask=tm,
                energy_floor=0.15,
            )
            for rep in range(tile_count + 1):
                start = rep * n_chips
                end = start + n_chips
                if end <= len(band):
                    band[start:end] += chips * amplitude * gain[start:end]
                else:
                    remainder = len(band) - start
                    if remainder > 0:
                        band[start:] += chips[:remainder] * amplitude * gain[start:]
        else:
            for rep in range(tile_count + 1):
                start = rep * n_chips
                end = start + n_chips
                if end <= len(band):
                    band[start:end] += chips * amplitude
                else:
                    remainder = len(band) - start
                    if remainder > 0:
                        band[start:] += chips[:remainder] * amplitude

        coeffs[_band_idx] = band
        result = _trim_or_pad(pywt.waverec(coeffs, "db4", mode="periodization"), orig_len)

    return result.astype(np.float32)


def extract_segment(
    segment: np.ndarray,
    band_config: BandConfig,
    pn_seed: int,
    chips_per_bit: int = 256,
) -> float:
    """
    Extract WID reliability for one segment using accumulated Z-score.

    Returns the mean Z-score across all 8 bits. The Z-score preserves
    processing gain from tiling: at -37 dB with full level-1 tiling
    (~172 tiles at 44100 Hz), expected Z ≈ 0.18 before psychoacoustic shaping.

    Erasure threshold must be calibrated in Z-score units (not [0, 1]).
    Recommended: erasure if mean_z < ERASURE_THRESHOLD_Z (1.0).
    """
    level = band_config.dwt_level
    coeffs = pywt.wavedec(segment.astype(np.float64), "db4", level=level, mode="periodization")
    band_idx = 0 if band_config.target_subband == "approximation" else -2
    band = coeffs[band_idx].astype(np.float64)

    n_chips = 8 * chips_per_bit
    pn = pn_sequence(n_chips, pn_seed)

    if len(band) < chips_per_bit:
        return 0.0

    _, z_scores, _ = accumulated_bit_decisions(band, pn, 8, chips_per_bit)
    return float(np.mean(z_scores))


def extract_symbol_segment(
    segment: np.ndarray,
    band_config: BandConfig,
    pn_seed: int,
    chips_per_bit: int = 256,
) -> tuple[int, float]:
    """
    Extract RS symbol and mean Z-score from one segment using EGC multi-band.

    Supports multi-band extraction via band_config.extra_dwt_levels.
    When extra_dwt_levels is non-empty, combines Z-score-weighted bit decisions
    across all levels using Equal Gain Combining (EGC).

    Returns:
        symbol: int in [0, 255] — decoded RS symbol (hard decision)
        mean_z: float — mean Z-score across 8 bits (reliability metric)

    Callers use mean_z to decide if this segment is an erasure:
        if mean_z < ERASURE_THRESHOLD_Z: mark as erasure (None for RS decoder)
    """
    all_levels = (band_config.dwt_level,) + band_config.extra_dwt_levels

    combined_raw = np.zeros(8, dtype=np.float64)
    combined_z = np.zeros(8, dtype=np.float64)

    n_chips = 8 * chips_per_bit
    pn = pn_sequence(n_chips, pn_seed)

    band_idx = 0 if band_config.target_subband == "approximation" else -2

    for level in all_levels:
        coeffs = pywt.wavedec(segment.astype(np.float64), "db4", level=level, mode="periodization")
        band = coeffs[band_idx].astype(np.float64)

        if len(band) < chips_per_bit:
            continue

        bits_l, z_l, _ = accumulated_bit_decisions(band, pn, 8, chips_per_bit)
        # EGC: accumulate Z-scores weighted by sign of bit decision
        for i in range(8):
            sign = 1.0 if bits_l[i] == 1 else -1.0
            combined_raw[i] += sign * z_l[i]
        combined_z += z_l

    bits_final = (combined_raw > 0).astype(np.uint8)
    symbol = int(sum(int(b) << (7 - i) for i, b in enumerate(bits_final)))
    mean_z = float(np.mean(combined_z)) / len(all_levels)
    return symbol, mean_z


def _trim_or_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) > target_len:
        return arr[:target_len]
    if len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)))
    return arr
