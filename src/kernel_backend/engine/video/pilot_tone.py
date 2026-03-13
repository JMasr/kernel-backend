"""
engine/video/pilot_tone.py

Layer 0 — fast content identification.
Embeds a 48-bit hash of content_id into the DC coefficient of 4×4 luma blocks.
The DC coefficient (0,0) is the most energy-dominant and survives even aggressive
H.264 quantization (CRF up to ~35).

Embedding:
  1. Decompose frame into 4×4 luma blocks.
  2. For each selected block, apply QIM to dct[0,0] with step = QIM_STEP_PILOT.
  3. Block selection is deterministic: HMAC(pepper, content_id + "pilot") → seed →
     np.random.default_rng(seed).choice(total_blocks, n_pilot_blocks, replace=False)
  4. Bits cycle through the 48-bit hash for all selected blocks.

Detection:
  1. Reconstruct block selection with same HMAC seed.
  2. Extract QIM bit from each block's dct[0,0].
  3. Majority vote per bit position across all blocks that encode that bit.
  4. pilot_hash_48 = int.from_bytes(SHA256(content_id.encode())[:6], "big")
  5. Agreement = fraction of 48 bits that match.
  6. Threshold: agreement >= 0.75 → pilot detected.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import cv2
import numpy as np

QIM_STEP_PILOT = 28.0          # calibrated to survive uint8 quantization + H.264 CRF 28
N_PILOT_BLOCKS_PER_FRAME = 256  # more blocks → more votes per bit for robustness
PILOT_AGREEMENT_THRESHOLD = 0.75


@dataclass(frozen=True)
class PilotDetection:
    agreement: float
    detected: bool
    pilot_hash_48: int | None


def pilot_hash_48(content_id: str) -> int:
    """Canonical formula — must match signing_service.py exactly."""
    return int.from_bytes(hashlib.sha256(content_id.encode()).digest()[:6], "big")


def embed_pilot(
    frame: np.ndarray,
    content_id: str,
    pepper: bytes,
) -> np.ndarray:
    """
    Embeds 48-bit pilot hash into 4×4 luma blocks of a single frame (BGR input).
    Returns modified frame (BGR, same dtype as input).
    """
    h, w = frame.shape[:2]

    blocks = _select_blocks(h, w, content_id, pepper)
    if not blocks:
        return frame.copy()

    hash_val = pilot_hash_48(content_id)
    bits = [(hash_val >> (47 - i)) & 1 for i in range(48)]

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_float = ycrcb[:, :, 0].astype(np.float32)

    for idx, (y0, x0) in enumerate(blocks):
        if y0 + 4 > h or x0 + 4 > w:
            continue
        bit = bits[idx % 48]
        block = y_float[y0:y0 + 4, x0:x0 + 4].copy()
        dct_block = cv2.dct(block)
        dct_block[0, 0] = _qim_embed(dct_block[0, 0], bit, QIM_STEP_PILOT)
        y_float[y0:y0 + 4, x0:x0 + 4] = cv2.idct(dct_block)

    ycrcb[:, :, 0] = np.clip(y_float, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def detect_pilot(
    frame: np.ndarray,
    content_id: str,
    pepper: bytes,
) -> PilotDetection:
    """
    Extracts and validates pilot hash from a single frame (BGR input).
    """
    h, w = frame.shape[:2]
    blocks = _select_blocks(h, w, content_id, pepper)
    if not blocks:
        return PilotDetection(agreement=0.0, detected=False, pilot_hash_48=None)

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_float = ycrcb[:, :, 0].astype(np.float32)

    # Accumulate votes per bit position
    votes = np.zeros((48, 2), dtype=np.int32)  # [bit_pos][0=zero, 1=one]

    for idx, (y0, x0) in enumerate(blocks):
        if y0 + 4 > h or x0 + 4 > w:
            continue
        bit_pos = idx % 48
        block = y_float[y0:y0 + 4, x0:x0 + 4].copy()
        dct_block = cv2.dct(block)
        extracted_bit = _qim_extract(dct_block[0, 0], QIM_STEP_PILOT)
        votes[bit_pos, extracted_bit] += 1

    # Majority vote per bit
    decoded_bits = []
    for i in range(48):
        decoded_bits.append(1 if votes[i, 1] >= votes[i, 0] else 0)

    decoded_hash = 0
    for b in decoded_bits:
        decoded_hash = (decoded_hash << 1) | b

    expected_hash = pilot_hash_48(content_id)
    expected_bits = [(expected_hash >> (47 - i)) & 1 for i in range(48)]

    matching = sum(1 for d, e in zip(decoded_bits, expected_bits) if d == e)
    agreement = matching / 48.0

    detected = agreement >= PILOT_AGREEMENT_THRESHOLD
    return PilotDetection(
        agreement=agreement,
        detected=detected,
        pilot_hash_48=decoded_hash if detected else None,
    )


def _select_blocks(
    height: int,
    width: int,
    content_id: str,
    pepper: bytes,
    block_size: int = 4,
    n_blocks: int = N_PILOT_BLOCKS_PER_FRAME,
) -> list[tuple[int, int]]:
    """
    Deterministic block selection via HMAC seed.
    Generates positions in normalized [0,1] space (resolution-independent),
    then converts to pixel coordinates for the given frame dimensions.
    """
    n_rows = height // block_size
    n_cols = width // block_size
    total = n_rows * n_cols

    if total < n_blocks:
        return []

    seed_bytes = hmac.new(
        pepper,
        (content_id + "pilot").encode(),
        hashlib.sha256,
    ).digest()
    seed = int.from_bytes(seed_bytes[:8], "big")
    rng = np.random.default_rng(seed)

    # Generate positions in normalized [0,1] space — resolution-independent
    norm_positions = rng.random((n_blocks, 2))  # (y_frac, x_frac)

    result = []
    for ny, nx in norm_positions:
        row = int(ny * n_rows)
        col = int(nx * n_cols)
        row = min(row, n_rows - 1)
        col = min(col, n_cols - 1)
        y0 = row * block_size
        x0 = col * block_size
        result.append((y0, x0))

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
