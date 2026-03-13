from __future__ import annotations

import numpy as np
import pywt

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.codec.spread_spectrum import (
    chip_stream,
    normalized_correlation,
    pn_sequence,
)


def embed_segment(
    segment: np.ndarray,        # float32, mono, one fixed-duration slice
    rs_symbol: int,             # single RS symbol, value 0–255
    band_config: BandConfig,    # from plan_audio_hopping
    pn_seed: int,               # HMAC(pepper, f"wid|{content_id}|{pubkey}|{i}")[:8]
    chips_per_bit: int = 32,
    target_snr_db: float = -14.0,
) -> np.ndarray:
    """
    Spread rs_symbol (8 bits) as a DSSS chip stream into coeffs[-2].
    Scale embedding strength by band RMS to achieve target_snr_db.
    Return segment of identical length.
    """
    orig_len = len(segment)

    # Build 8-bit array (MSB first)
    bits = np.array([(rs_symbol >> (7 - i)) & 1 for i in range(8)], dtype=np.float32)
    chips = chip_stream(bits, chips_per_bit, pn_seed)  # 8 * 32 = 256 chips

    level = band_config.dwt_level
    coeffs = pywt.wavedec(segment.astype(np.float64), "db4", level=level, mode="periodization")
    band = coeffs[-2].copy()

    band_rms = float(np.sqrt(np.mean(band ** 2)))
    if band_rms < 1e-10:
        band_rms = 1.0
    amplitude = band_rms * (10.0 ** (target_snr_db / 20.0))

    n = min(len(chips), len(band))
    band[:n] += chips[:n] * amplitude
    coeffs[-2] = band

    reconstructed = pywt.waverec(coeffs, "db4", mode="periodization")
    return _trim_or_pad(reconstructed, orig_len).astype(np.float32)


def extract_segment(
    segment: np.ndarray,
    band_config: BandConfig,
    pn_seed: int,
    chips_per_bit: int = 32,
) -> float:
    """
    Despread and return a soft correlation value in [0.0, 1.0].
    Computed as the mean of absolute per-bit normalised correlations.
    High value → valid RS symbol present; low value → erasure candidate.
    Erasure threshold decision belongs to signing_service, not here.
    """
    level = band_config.dwt_level
    coeffs = pywt.wavedec(segment.astype(np.float64), "db4", level=level, mode="periodization")
    band = coeffs[-2].astype(np.float32)

    n_chips = 8 * chips_per_bit
    pn = pn_sequence(n_chips, pn_seed)

    if len(band) < chips_per_bit:
        return 0.0

    per_bit_corrs: list[float] = []
    for i in range(8):
        bs = i * chips_per_bit
        be = bs + chips_per_bit
        if be > len(band):
            break
        corr = normalized_correlation(band[bs:be], pn[bs:be])
        per_bit_corrs.append(abs(corr))

    if not per_bit_corrs:
        return 0.0
    return float(np.mean(per_bit_corrs))


def _trim_or_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) > target_len:
        return arr[:target_len]
    if len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)))
    return arr
