from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pywt
import pytest

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.audio.wid_beacon import embed_segment, extract_segment
from kernel_backend.engine.codec.hopping import plan_audio_hopping
from kernel_backend.engine.codec.reed_solomon import ReedSolomonCodec
from kernel_backend.engine.codec.spread_spectrum import normalized_correlation, pn_sequence

SR = 44100
SEGMENT_S = 2.0
SEG_LEN = int(SR * SEGMENT_S)
PEPPER = b"test-pepper-bytes-padded-to-32b!"
CONTENT_ID = "test-content-wid"
PUBKEY = "-----BEGIN PUBLIC KEY-----\ntest\n-----END PUBLIC KEY-----\n"
# Use -6 dB for tests that need reliable bit-level detection.
# The -14 dB API default is for perceptual quality; tests that probe
# bit-accurate recovery must use enough SNR to keep BER well below 1%.
_TEST_SNR_DB = -6.0


def _pn_seed(i: int) -> int:
    msg = f"wid|{CONTENT_ID}|{PUBKEY}|{i}".encode()
    return int.from_bytes(hmac.new(PEPPER, msg, hashlib.sha256).digest()[:8], "big")


def _noise_segment(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(SEG_LEN).astype(np.float32) * 0.3


def _default_bc(i: int = 0) -> BandConfig:
    return BandConfig(segment_index=i, coeff_positions=[], dwt_level=2)


def test_embed_extract_roundtrip_correlation() -> None:
    """
    embed_segment → extract_segment → correlation > 0.3 on synthetic noise.
    Uses -6 dB SNR to achieve reliable per-bit despread (≈0.5 per-bit corr).
    """
    seg = _noise_segment(0)
    bc = _default_bc(0)
    seed = _pn_seed(0)
    embedded = embed_segment(seg, rs_symbol=0b10101010, band_config=bc,
                              pn_seed=seed, target_snr_db=_TEST_SNR_DB)
    corr = extract_segment(embedded, bc, seed)
    assert corr > 0.3, f"Correlation {corr:.4f} too low (expected > 0.3)"


def test_rs_symbols_survive_roundtrip() -> None:
    """
    32 segments, embed all RS symbols at -6 dB, decode per-bit, RS decode.
    decoded bytes == original WID.
    """
    wid = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10'
    n_segments = 32
    codec = ReedSolomonCodec(n_symbols=n_segments)
    rs_symbols = codec.encode(wid)

    band_configs = plan_audio_hopping(n_segments, CONTENT_ID, PUBKEY, PEPPER)
    chips_per_bit = 32
    seeds = [_pn_seed(i) for i in range(n_segments)]

    embedded_segs = []
    for i in range(n_segments):
        seg = _noise_segment(i)
        emb = embed_segment(seg, rs_symbols[i], band_configs[i], seeds[i],
                            chips_per_bit, target_snr_db=_TEST_SNR_DB)
        embedded_segs.append(emb)

    # Recover each RS symbol via per-bit demodulation in the DWT detail band.
    recovered_symbols: list[int] = []
    for emb, bc, seed in zip(embedded_segs, band_configs, seeds):
        level = bc.dwt_level
        coeffs = pywt.wavedec(emb.astype(np.float64), "db4", level=level, mode="periodization")
        band = coeffs[-2].astype(np.float32)
        pn = pn_sequence(8 * chips_per_bit, seed)
        symbol = 0
        for bit_i in range(8):
            bs = bit_i * chips_per_bit
            be = bs + chips_per_bit
            corr = normalized_correlation(band[bs:be], pn[bs:be])
            symbol = (symbol << 1) | (1 if corr > 0 else 0)
        recovered_symbols.append(symbol)

    decoded = codec.decode(recovered_symbols)
    assert decoded == wid, f"WID mismatch: {decoded.hex()} != {wid.hex()}"


def test_output_length_preserved() -> None:
    """Output length == input length for various segment sizes."""
    for length in [SR, SEG_LEN, int(SR * 0.5)]:
        seg = np.random.default_rng(99).standard_normal(length).astype(np.float32) * 0.3
        bc = _default_bc()
        embedded = embed_segment(seg, 0b11001100, bc, _pn_seed(0))
        assert len(embedded) == length


def test_wrong_seed_gives_low_correlation() -> None:
    """
    Different pn_seed on extraction → mean |correlation| < 0.25.
    With correct seed at -6 dB the correlation is ~0.5; wrong seed gives ~0.14.
    """
    seg = _noise_segment(0)
    bc = _default_bc(0)
    embed_seed = _pn_seed(0)
    wrong_seed = _pn_seed(99)

    corrs = []
    for sym in [0, 85, 170, 255, 42, 128, 200, 15]:
        embedded = embed_segment(seg, sym, bc, embed_seed, target_snr_db=_TEST_SNR_DB)
        corr = extract_segment(embedded, bc, wrong_seed)
        corrs.append(corr)

    assert float(np.mean(corrs)) < 0.25, (
        f"Mean |corr| with wrong seed: {np.mean(corrs):.4f} (expected < 0.25)"
    )
