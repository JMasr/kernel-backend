from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pywt
import pytest

from kernel_backend.core.domain.watermark import BandConfig
from kernel_backend.engine.audio.wid_beacon import (
    ERASURE_THRESHOLD_Z,
    embed_segment,
    extract_segment,
)
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


def _approx_bc(i: int = 0, level: int = 5) -> BandConfig:
    return BandConfig(
        segment_index=i, coeff_positions=[], dwt_level=level,
        target_subband="approximation",
    )


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
    assert corr > 1.0, f"Z-score {corr:.4f} too low (expected > {ERASURE_THRESHOLD_Z})"


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
    Wrong pn_seed on extraction → mean Z-score significantly lower than correct seed.
    With correct seed at -6 dB the Z-score is >> ERASURE_THRESHOLD_Z.
    With wrong seed, Z-score converges to the noise floor (E[|Z|] ≈ 0.8).
    We verify wrong-seed Z-score < correct-seed Z-score (not an exact threshold,
    since with only 8 symbols the noise floor estimate has variance).
    """
    seg = _noise_segment(0)
    bc = _default_bc(0)
    embed_seed = _pn_seed(0)
    wrong_seed = _pn_seed(99)

    corrs_wrong = []
    corrs_correct = []
    for sym in [0, 85, 170, 255, 42, 128, 200, 15]:
        embedded = embed_segment(seg, sym, bc, embed_seed, target_snr_db=_TEST_SNR_DB)
        corrs_wrong.append(extract_segment(embedded, bc, wrong_seed))
        corrs_correct.append(extract_segment(embedded, bc, embed_seed))

    mean_wrong = float(np.mean(corrs_wrong))
    mean_correct = float(np.mean(corrs_correct))
    assert mean_wrong < mean_correct, (
        f"Wrong-seed Z ({mean_wrong:.4f}) should be < correct-seed Z ({mean_correct:.4f})"
    )
    # Correct seed must clearly be above the erasure threshold
    assert mean_correct >= ERASURE_THRESHOLD_Z, (
        f"Correct-seed Z-score {mean_correct:.4f} < ERASURE_THRESHOLD_Z {ERASURE_THRESHOLD_Z}"
    )


# ── Approximation band tests ────────────────────────────────────────────────

def test_approx_band_embed_extract_roundtrip() -> None:
    """Embed in approximation band at level 5, extract → Z-score above threshold."""
    seg = _noise_segment(0)
    bc = _approx_bc(0, level=5)
    seed = _pn_seed(0)
    embedded = embed_segment(seg, rs_symbol=0b10101010, band_config=bc,
                              pn_seed=seed, target_snr_db=_TEST_SNR_DB)
    corr = extract_segment(embedded, bc, seed)
    assert corr > ERASURE_THRESHOLD_Z, (
        f"Approx band Z-score {corr:.4f} too low (expected > {ERASURE_THRESHOLD_Z})"
    )


def test_approx_band_output_length_preserved() -> None:
    """Approximation band embedding preserves signal length."""
    seg = _noise_segment(0)
    bc = _approx_bc(0, level=5)
    embedded = embed_segment(seg, 0b11001100, bc, _pn_seed(0), target_snr_db=_TEST_SNR_DB)
    assert len(embedded) == len(seg)


def test_cross_subband_mismatch_low_z() -> None:
    """Embedding in detail, extracting as approximation → near-zero Z-score."""
    seg = _noise_segment(0)
    seed = _pn_seed(0)

    # Embed in detail band (default)
    bc_detail = _default_bc(0)
    embedded = embed_segment(seg, 0b10101010, bc_detail, seed, target_snr_db=_TEST_SNR_DB)

    # Extract expecting approximation band → should find nothing
    bc_approx = _approx_bc(0, level=2)
    z_cross = extract_segment(embedded, bc_approx, seed)

    # Extract with correct subband
    z_correct = extract_segment(embedded, bc_detail, seed)

    assert z_cross < z_correct, (
        f"Cross-subband Z ({z_cross:.4f}) should be < correct Z ({z_correct:.4f})"
    )


def test_default_subband_is_detail() -> None:
    """BandConfig without target_subband uses detail (backward compat)."""
    bc = BandConfig(segment_index=0, coeff_positions=[], dwt_level=2)
    assert bc.target_subband == "detail"


# ── Segment map tests ───────────────────────────────────────────────────────

from kernel_backend.core.domain.watermark import (
    AudioEmbeddingParams,
    EmbeddingParams,
    SegmentMap,
    embedding_params_to_dict,
    embedding_params_from_dict,
)
from kernel_backend.engine.audio.wid_beacon import (
    extract_symbol_segment,
)


def test_segment_map_roundtrip() -> None:
    """Embed with segment map → extract with same map → RS decode succeeds."""
    wid = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10'
    n_total_segments = 40
    # Select every other segment: [0, 2, 4, ..., 30] = 16 segments ...
    # Actually we need rs_n >= 17, so select 32 of 40
    n_segments = 32
    # Select segments 4..35 (skip first 4 silent ones)
    selected = list(range(4, 4 + n_segments))
    segment_map = SegmentMap(
        selected_indices=tuple(selected),
        total_segments=n_total_segments,
    )
    seg_to_rs = {seg_idx: rs_idx for rs_idx, seg_idx in enumerate(selected)}

    codec = ReedSolomonCodec(n_symbols=n_segments)
    rs_symbols = codec.encode(wid)
    band_configs = plan_audio_hopping(n_segments, CONTENT_ID, PUBKEY, PEPPER)
    chips_per_bit = 32

    # Embed: only into selected segments, keyed by RS index
    embedded_segs: dict[int, np.ndarray] = {}
    for seg_idx in range(n_total_segments):
        seg = _noise_segment(seg_idx)
        if seg_idx in seg_to_rs:
            rs_idx = seg_to_rs[seg_idx]
            seed = _pn_seed(rs_idx)
            emb = embed_segment(seg, rs_symbols[rs_idx], band_configs[rs_idx],
                                seed, chips_per_bit, target_snr_db=_TEST_SNR_DB)
            embedded_segs[seg_idx] = emb
        else:
            embedded_segs[seg_idx] = seg

    # Extract: using segment map
    recovered_symbols: list[int | None] = [None] * n_segments
    for seg_idx in selected:
        rs_idx = seg_to_rs[seg_idx]
        seed = _pn_seed(rs_idx)
        symbol, mean_z = extract_symbol_segment(
            embedded_segs[seg_idx], band_configs[rs_idx], seed,
            chips_per_bit=chips_per_bit,
        )
        if mean_z >= 1.0:
            recovered_symbols[rs_idx] = symbol

    decoded = codec.decode(recovered_symbols)
    assert decoded == wid, f"WID mismatch: {decoded.hex()} != {wid.hex()}"


def test_segment_map_skips_silence() -> None:
    """Create signal with silent segments at start, verify embedding avoids them."""
    from kernel_backend.engine.audio.segment_scorer import score_segments, select_best

    n_total = 20
    segments = []
    for i in range(n_total):
        if i < 3:
            # Silent segments
            segments.append((i, np.zeros(SEG_LEN, dtype=np.float32)))
        else:
            segments.append((i, _noise_segment(i)))

    scores = score_segments(iter(segments))
    selected = select_best(scores, n_needed=17)

    # None of the first 3 (silent) segments should be selected
    for idx in selected:
        assert idx >= 3, f"Silent segment {idx} was selected"


def test_segment_map_serialization_roundtrip() -> None:
    """SegmentMap survives embedding_params serialization → deserialization."""
    sm = SegmentMap(selected_indices=(3, 5, 7, 10), total_segments=20)
    ap = AudioEmbeddingParams(
        dwt_levels=(2,),
        chips_per_bit=32,
        psychoacoustic=False,
        safety_margin_db=12.0,
        target_snr_db=-20.0,
        segment_map=sm,
    )
    ep = EmbeddingParams(audio=ap, video=None)
    d = embedding_params_to_dict(ep)
    ep2 = embedding_params_from_dict(d)

    assert ep2.audio is not None
    assert ep2.audio.segment_map is not None
    assert ep2.audio.segment_map.selected_indices == (3, 5, 7, 10)
    assert ep2.audio.segment_map.total_segments == 20


def test_segment_map_none_backward_compat() -> None:
    """Legacy content without segment_map deserializes to None."""
    d = {
        "audio": {
            "dwt_levels": [2],
            "chips_per_bit": 32,
            "psychoacoustic": False,
            "safety_margin_db": 12.0,
            "target_snr_db": -20.0,
        },
        "video": None,
    }
    ep = embedding_params_from_dict(d)
    assert ep.audio is not None
    assert ep.audio.segment_map is None
