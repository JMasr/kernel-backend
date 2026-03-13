"""
engine/video/wid_watermark.py

Layer 1 — WID embedding.
Embeds Reed-Solomon symbols into video frames using QIM on 4×4 DCT coefficients.

Coefficient selection per segment:
  Mandatory (always): (0,1), (1,0)    — AC coefficients, robust to DC drift
  Optional (per HMAC seed): (1,1), (0,2) — extends redundancy when available

Each frame in the segment embeds the same bit pattern.
Detection uses majority vote across all frames in the segment.
Agreement score = fraction of bits that match the embedded pattern.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import cv2
import numpy as np

BLOCK_SIZE = 4
QIM_STEP_WID = 64.0
N_WID_BLOCKS_PER_SEGMENT = 128
MANDATORY_COEFFS = [(0, 1), (1, 0)]
OPTIONAL_COEFFS = [(1, 1), (0, 2)]
WID_AGREEMENT_THRESHOLD = 0.52


@dataclass(frozen=True)
class SegmentWIDResult:
    segment_idx: int
    agreement: float
    extracted_bits: bytes
    erasure: bool


def embed_segment(
    frames: list[np.ndarray],
    symbol_bits: np.ndarray,
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
) -> list[np.ndarray]:
    """
    Embeds 8-bit RS symbol into all frames of a segment.
    Returns modified frames (same dtype and shape as input).
    """
    coeffs = _coeff_set(content_id, author_public_key, segment_idx, pepper)
    result = []
    for frame in frames:
        h, w = frame.shape[:2]
        blocks = _select_blocks(h, w, content_id, author_public_key, segment_idx, pepper)
        if not blocks:
            result.append(frame.copy())
            continue

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_float = ycrcb[:, :, 0].astype(np.float32)

        for block_idx, (y0, x0) in enumerate(blocks):
            if y0 + BLOCK_SIZE > h or x0 + BLOCK_SIZE > w:
                continue
            block = y_float[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE].copy()
            dct_block = cv2.dct(block)
            bit = int(symbol_bits[block_idx % 8])
            for cr, cc in coeffs:
                dct_block[cr, cc] = _qim_embed(dct_block[cr, cc], bit, QIM_STEP_WID)
            y_float[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE] = cv2.idct(dct_block)

        ycrcb[:, :, 0] = np.clip(y_float, 0, 255).astype(np.uint8)
        result.append(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR))
    return result


def extract_segment(
    frames: list[np.ndarray],
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
) -> SegmentWIDResult:
    """
    Extracts 8-bit RS symbol from a segment using majority vote across frames.
    """
    coeffs = _coeff_set(content_id, author_public_key, segment_idx, pepper)

    # Accumulate votes per bit position: votes[bit_pos][0/1]
    votes = np.zeros((8, 2), dtype=np.int32)

    for frame in frames:
        h, w = frame.shape[:2]
        blocks = _select_blocks(h, w, content_id, author_public_key, segment_idx, pepper)
        if not blocks:
            continue

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_float = ycrcb[:, :, 0].astype(np.float32)

        for block_idx, (y0, x0) in enumerate(blocks):
            if y0 + BLOCK_SIZE > h or x0 + BLOCK_SIZE > w:
                continue
            bit_pos = block_idx % 8
            block = y_float[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE].copy()
            dct_block = cv2.dct(block)
            for cr, cc in coeffs:
                extracted = _qim_extract(dct_block[cr, cc], QIM_STEP_WID)
                votes[bit_pos, extracted] += 1

    # Majority vote per bit
    decoded_bits = []
    total_votes = 0
    matching_votes = 0
    for i in range(8):
        bit = 1 if votes[i, 1] >= votes[i, 0] else 0
        decoded_bits.append(bit)
        total_votes += votes[i, 0] + votes[i, 1]
        matching_votes += max(votes[i, 0], votes[i, 1])

    agreement = matching_votes / total_votes if total_votes > 0 else 0.0
    symbol_byte = 0
    for b in decoded_bits:
        symbol_byte = (symbol_byte << 1) | b

    return SegmentWIDResult(
        segment_idx=segment_idx,
        agreement=agreement,
        extracted_bits=bytes([symbol_byte]),
        erasure=agreement < WID_AGREEMENT_THRESHOLD,
    )


def _coeff_set(
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
) -> list[tuple[int, int]]:
    """
    Returns coefficient list for this segment.
    Always includes MANDATORY_COEFFS. May include 0, 1, or 2 OPTIONAL_COEFFS.
    """
    msg = f"wid_coeff|{content_id}|{author_public_key}|{segment_idx}".encode()
    digest = hmac.new(pepper, msg, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    n_extra = int(rng.integers(0, 3))  # 0, 1, or 2 optional coefficients
    result = list(MANDATORY_COEFFS)
    if n_extra > 0:
        indices = rng.choice(len(OPTIONAL_COEFFS), size=min(n_extra, len(OPTIONAL_COEFFS)), replace=False)
        for idx in indices:
            result.append(OPTIONAL_COEFFS[int(idx)])
    return result


def _select_blocks(
    height: int,
    width: int,
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
    n_blocks: int = N_WID_BLOCKS_PER_SEGMENT,
) -> list[tuple[int, int]]:
    """
    Deterministic block selection (normalized coordinates → pixel coords).
    Same normalization invariant as pilot_tone._select_blocks.
    """
    n_rows = height // BLOCK_SIZE
    n_cols = width // BLOCK_SIZE
    total = n_rows * n_cols

    if total < n_blocks:
        n_blocks = total
    if n_blocks == 0:
        return []

    msg = f"wid_blocks|{content_id}|{author_public_key}|{segment_idx}".encode()
    digest = hmac.new(pepper, msg, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)

    norm_positions = rng.random((n_blocks, 2))
    result = []
    for ny, nx in norm_positions:
        row = min(int(ny * n_rows), n_rows - 1)
        col = min(int(nx * n_cols), n_cols - 1)
        result.append((row * BLOCK_SIZE, col * BLOCK_SIZE))
    return result


def _qim_embed(value: float, bit: int, step: float) -> float:
    """Quantization Index Modulation — embed one bit."""
    half = step / 2.0
    if bit == 0:
        return step * np.round(value / step)
    else:
        return step * np.round((value - half) / step) + half


def _qim_extract(value: float, step: float) -> int:
    """Quantization Index Modulation — extract one bit."""
    half = step / 2.0
    q0 = step * np.round(value / step)
    q1 = step * np.round((value - half) / step) + half
    if abs(value - q0) <= abs(value - q1):
        return 0
    return 1
