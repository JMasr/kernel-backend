from reedsolo import RSCodec, ReedSolomonError  # noqa: F401 (re-exported for callers)


class ReedSolomonCodec:
    def __init__(self, n_symbols: int, k_data: int = 16) -> None:
        """
        n_symbols: total symbols (data + parity). Max 255.
        k_data: data symbols. Fixed at 16 (= WID byte count).
        Raises ValueError if n_symbols <= k_data or n_symbols > 255.
        """
        if n_symbols <= k_data:
            raise ValueError(
                f"n_symbols ({n_symbols}) must be > k_data ({k_data})"
            )
        if n_symbols > 255:
            raise ValueError(
                f"n_symbols ({n_symbols}) exceeds GF(2^8) field limit of 255"
            )
        self._k = k_data
        self._n = n_symbols
        self._codec = RSCodec(n_symbols - k_data)

    def encode(self, data: bytes) -> list[int]:
        """
        Encode k_data bytes → n_symbols ints (GF(2^8), values 0–255).
        Raises ValueError if len(data) != k_data.
        """
        if len(data) != self._k:
            raise ValueError(
                f"data must be exactly {self._k} bytes, got {len(data)}"
            )
        encoded = self._codec.encode(data)
        return list(encoded)

    def decode(self, symbols: list[int | None]) -> bytes:
        """
        Decode n_symbols ints → k_data bytes.
        None in the list = erasure (position known, value lost).
        Raises reedsolo.ReedSolomonError if uncorrectable.
        Uses nostrip=True to prevent stripping leading zero bytes from binary WID data.
        """
        erasures = [i for i, s in enumerate(symbols) if s is None]
        filled = bytes(0 if s is None else s for s in symbols)
        decoded, _, _ = self._codec.decode(filled, erase_pos=erasures)
        # Pad to exactly k_data bytes — reedsolo may strip leading zero bytes
        return bytes(decoded).rjust(self._k, b"\x00")
