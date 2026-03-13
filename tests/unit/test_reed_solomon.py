import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from reedsolo import ReedSolomonError

from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec


@given(
    st.binary(min_size=16, max_size=16),
    st.integers(min_value=17, max_value=255),
)
@settings(max_examples=50)
def test_roundtrip(data: bytes, n_symbols: int) -> None:
    codec = ReedSolomonCodec(n_symbols=n_symbols)
    symbols = codec.encode(data)
    assert len(symbols) == n_symbols
    assert codec.decode(symbols) == data


@given(
    st.binary(min_size=16, max_size=16),
    st.integers(min_value=32, max_value=64),
)
@settings(max_examples=50)
def test_erasure_at_capacity(data: bytes, n_symbols: int) -> None:
    codec = ReedSolomonCodec(n_symbols=n_symbols)
    symbols = codec.encode(data)
    # Erase exactly (n_symbols - 16) positions — maximum correctable
    n_erasures = n_symbols - 16
    erased = [None if i < n_erasures else s for i, s in enumerate(symbols)]
    assert codec.decode(erased) == data


@given(
    st.binary(min_size=16, max_size=16),
    st.integers(min_value=32, max_value=64),
)
@settings(max_examples=50)
def test_erasure_over_capacity(data: bytes, n_symbols: int) -> None:
    codec = ReedSolomonCodec(n_symbols=n_symbols)
    symbols = codec.encode(data)
    # Erase (n_symbols - 16 + 1) positions — one beyond capacity
    n_erasures = n_symbols - 16 + 1
    erased = [None if i < n_erasures else s for i, s in enumerate(symbols)]
    with pytest.raises(ReedSolomonError):
        codec.decode(erased)


def test_constructor_rejects_invalid_n() -> None:
    # n_symbols=16 → ValueError (n must be > k)
    with pytest.raises(ValueError):
        ReedSolomonCodec(n_symbols=16)
    # n_symbols=256 → ValueError (exceeds GF field)
    with pytest.raises(ValueError):
        ReedSolomonCodec(n_symbols=256)


def test_encode_rejects_wrong_length() -> None:
    codec = ReedSolomonCodec(n_symbols=32)
    with pytest.raises(ValueError):
        codec.encode(b"\x00" * 15)
    with pytest.raises(ValueError):
        codec.encode(b"\x00" * 17)


def test_encode_output_length() -> None:
    codec = ReedSolomonCodec(n_symbols=32)
    symbols = codec.encode(b"\xAB" * 16)
    assert len(symbols) == 32
    assert all(0 <= s <= 255 for s in symbols)


def test_leading_zero_bytes_preserved() -> None:
    # nostrip=True must prevent stripping leading zero bytes
    codec = ReedSolomonCodec(n_symbols=32)
    data = b"\x00\x00\x00\x00" + b"\x01\x02\x03\x04" * 3
    assert codec.decode(codec.encode(data)) == data
