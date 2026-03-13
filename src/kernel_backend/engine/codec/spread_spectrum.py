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
