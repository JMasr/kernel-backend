import numpy as np


def pn_sequence(length: int, seed: int) -> np.ndarray:
    """
    Deterministic PN sequence of {-1.0, +1.0} float32 values.
    np.random.default_rng(seed) — same seed always produces same sequence.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length)
    return np.where(bits == 1, 1.0, -1.0).astype(np.float32)


def chip_stream(bits: np.ndarray, chips_per_bit: int, seed: int) -> np.ndarray:
    """
    DSSS chip stream. Each bit is spread over chips_per_bit chips
    using the PN sequence: bit=1 → +pn, bit=0 → -pn.
    Output length = len(bits) * chips_per_bit.
    """
    pn = pn_sequence(len(bits) * chips_per_bit, seed)
    expanded = np.repeat(np.where(bits == 1, 1.0, -1.0).astype(np.float32), chips_per_bit)
    return expanded * pn


def accumulated_bit_decisions(
    band: np.ndarray,
    pn: np.ndarray,
    n_bits: int,
    chips_per_bit: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Tile PN template across band, accumulate per-bit dot products,
    return hard bit decisions and Z-scores.

    The Z-score denominator uses band variance (not band norm) so that
    processing gain from tiling is preserved: Z grows as sqrt(n_tiles).

    Args:
        band: DWT coefficient array (float64), length >= n_bits * chips_per_bit
        pn:   PN sequence, length == n_bits * chips_per_bit, values in {-1, +1}
        n_bits: number of bits to decode (typically 8 for WID, 48 for pilot)
        chips_per_bit: chips allocated to each bit

    Returns:
        bits:    np.ndarray[uint8], shape (n_bits,)  — hard decisions {0, 1}
        z_scores: np.ndarray[float64], shape (n_bits,) — reliability per bit
        n_tiles: int — number of complete tiles accumulated
    """
    n_chips = n_bits * chips_per_bit
    n_tiles = len(band) // n_chips

    if n_tiles == 0:
        # Band shorter than one chip period — fallback: single pass, no tiling
        n_tiles = 1
        effective_len = min(n_chips, len(band))
        band_view = band[:effective_len]
        pn_view = pn[:effective_len]
        per_bit_raw = np.zeros(n_bits, dtype=np.float64)
        for i in range(n_bits):
            bs = i * chips_per_bit
            be = min(bs + chips_per_bit, effective_len)
            if be > bs:
                per_bit_raw[i] = np.dot(band_view[bs:be], pn_view[bs:be])
    else:
        per_bit_raw = np.zeros(n_bits, dtype=np.float64)
        for tile in range(n_tiles):
            offset = tile * n_chips
            band_tile = band[offset: offset + n_chips]
            for i in range(n_bits):
                bs = i * chips_per_bit
                be = bs + chips_per_bit
                per_bit_raw[i] += np.dot(band_tile[bs:be], pn[bs:be])

    # Hard bit decisions from sign
    bits = (per_bit_raw > 0).astype(np.uint8)

    # Z-score: preserve processing gain by using band variance as noise estimator
    tiled_len = n_tiles * n_chips
    band_variance = float(np.var(band[:tiled_len])) if tiled_len > 0 else 1.0
    if band_variance < 1e-10:
        band_variance = 1.0
    noise_std_per_dot = np.sqrt(band_variance * chips_per_bit * n_tiles)

    if noise_std_per_dot < 1e-10:
        z_scores = np.zeros(n_bits, dtype=np.float64)
    else:
        z_scores = np.abs(per_bit_raw) / noise_std_per_dot

    return bits, z_scores, n_tiles


def normalized_correlation(window: np.ndarray, template: np.ndarray) -> float:
    """
    Normalized cross-correlation in [-1.0, 1.0].
    Returns 0.0 if either vector has zero norm (no division by zero).
    """
    norm_w = float(np.linalg.norm(window))
    norm_t = float(np.linalg.norm(template))
    if norm_w == 0.0 or norm_t == 0.0:
        return 0.0
    return float(np.dot(window.astype(np.float64), template.astype(np.float64)) / (norm_w * norm_t))
