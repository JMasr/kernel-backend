import numpy as np

from kernel_backend.engine.codec.spread_spectrum import (
    chip_stream,
    normalized_correlation,
    pn_sequence,
)


def test_pn_sequence_deterministic() -> None:
    assert np.array_equal(pn_sequence(100, 42), pn_sequence(100, 42))


def test_pn_sequence_different_seeds() -> None:
    assert not np.array_equal(pn_sequence(100, 42), pn_sequence(100, 43))


def test_pn_sequence_values() -> None:
    seq = pn_sequence(1000, 0)
    assert set(seq).issubset({-1.0, 1.0})


def test_pn_sequence_dtype() -> None:
    seq = pn_sequence(100, 0)
    assert seq.dtype == np.float32


def test_chip_stream_length() -> None:
    bits = np.array([0, 1, 0, 1])
    assert len(chip_stream(bits, chips_per_bit=32, seed=0)) == 128


def test_chip_stream_deterministic() -> None:
    bits = np.array([1, 0, 1, 1, 0])
    s1 = chip_stream(bits, chips_per_bit=16, seed=7)
    s2 = chip_stream(bits, chips_per_bit=16, seed=7)
    assert np.array_equal(s1, s2)


def test_normalized_correlation_self() -> None:
    x = pn_sequence(64, 7)
    assert abs(normalized_correlation(x, x) - 1.0) < 1e-5


def test_normalized_correlation_inverse() -> None:
    x = pn_sequence(64, 7)
    assert abs(normalized_correlation(x, -x) + 1.0) < 1e-5


def test_normalized_correlation_zero_vector() -> None:
    x = pn_sequence(64, 7)
    assert normalized_correlation(np.zeros(64), x) == 0.0


def test_normalized_correlation_both_zero() -> None:
    assert normalized_correlation(np.zeros(64), np.zeros(64)) == 0.0


def test_normalized_correlation_range() -> None:
    x = pn_sequence(128, 10)
    y = pn_sequence(128, 20)
    c = normalized_correlation(x, y)
    assert -1.0 <= c <= 1.0
