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
from functools import lru_cache

import cv2
import numpy as np

from kernel_backend.core.domain.dsp_manifest import PRODUCTION_MANIFEST as _M
from kernel_backend.core.domain.watermark import VideoEmbeddingParams

BLOCK_SIZE = _M.video_wid.block_size
QIM_STEP_WID = _M.video_wid.qim_step_wid
N_WID_BLOCKS_PER_SEGMENT = _M.video_wid.n_blocks_per_segment
MANDATORY_COEFFS = [(0, 1), (1, 0)]
OPTIONAL_COEFFS = [(1, 1), (0, 2)]
WID_AGREEMENT_THRESHOLD = _M.video_wid.agreement_threshold

# Adaptive QIM step parameters (Chou-Li JND model)
JND_BASE_LUMINANCE = 127.0
QIM_STEP_ADAPTIVE_BASE = _M.video_wid.qim_step_base
QIM_STEP_ADAPTIVE_MIN = _M.video_wid.qim_step_min
QIM_STEP_ADAPTIVE_MAX = _M.video_wid.qim_step_max
QIM_STEP_QUANTIZE_TO = _M.video_wid.qim_quantize_to


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


def _embed_ycrcb_inplace(
    ycrcb: np.ndarray,
    symbol_bits: np.ndarray,
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
    use_jnd_adaptive: bool = False,
    jnd_params: VideoEmbeddingParams | None = None,
) -> None:
    """Modify the Y channel of a uint8 YCrCb frame in place. Core DCT/QIM loop."""
    coeffs_list = _coeff_set(content_id, author_public_key, segment_idx, pepper)
    h, w = ycrcb.shape[:2]

    oversample = jnd_params.block_oversample if jnd_params else 1
    min_var = jnd_params.min_block_variance if jnd_params else 0.0

    candidates = _select_blocks(
        h, w, content_id, author_public_key, segment_idx, pepper,
        oversample=oversample,
    )
    if not candidates:
        return

    y_float = ycrcb[:, :, 0].astype(np.float32)

    if use_jnd_adaptive and jnd_params is not None:
        step_base = jnd_params.qim_step_base
        step_min = jnd_params.qim_step_min
        step_max = jnd_params.qim_step_max
        quantize_to = jnd_params.qim_quantize_to
    else:
        step_base = step_min = step_max = QIM_STEP_WID
        quantize_to = 1.0

    for cand_idx, (y0, x0) in enumerate(candidates):
        if y0 + BLOCK_SIZE > h or x0 + BLOCK_SIZE > w:
            continue

        block = y_float[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE].copy()

        if min_var > 0 and float(np.var(block)) < min_var:
            continue

        if use_jnd_adaptive:
            block_mean = float(np.mean(block))
            step = _compute_adaptive_step(
                block_mean, step_base, step_min, step_max, quantize_to
            )
        else:
            step = QIM_STEP_WID

        dct_block = cv2.dct(block)
        bit = int(symbol_bits[cand_idx % 8])
        for cr, cc in coeffs_list:
            dct_block[cr, cc] = _qim_embed(dct_block[cr, cc], bit, step)
        y_float[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE] = cv2.idct(dct_block)

    ycrcb[:, :, 0] = np.clip(y_float, 0, 255).astype(np.uint8)


def embed_video_frame(
    frame: np.ndarray,
    symbol_bits: np.ndarray,
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
    use_jnd_adaptive: bool = False,
    jnd_params: VideoEmbeddingParams | None = None,
) -> np.ndarray:
    """
    Embed one RS symbol into a single BGR frame.

    When use_jnd_adaptive=True, QIM step varies per 4×4 block based on
    the block's mean luminance (Chou-Li JND model). The step is derived
    from the unmodified block before any coefficient change, ensuring
    the extractor can reproduce the same step from the received frame.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    _embed_ycrcb_inplace(
        ycrcb, symbol_bits, content_id, author_public_key,
        segment_idx, pepper, use_jnd_adaptive, jnd_params,
    )
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def embed_video_frame_yuvj420_planes(
    frame: np.ndarray,
    symbol_bits: np.ndarray,
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
    use_jnd_adaptive: bool = False,
    jnd_params: VideoEmbeddingParams | None = None,
) -> tuple[bytes, bytes, bytes]:
    """
    Fused embed + YUV420 plane extraction for the signing encoder pipe.

    Equivalent to `embed_video_frame` followed by BGR→yuvj420p conversion,
    but drops the YCrCb→BGR→YUV round-trip. One cvtColor per frame instead
    of three. Full-range YCrCb planes match ffmpeg's yuvj420p pipe.

    Returns (Y, U, V) as bytes — caller writes each to the ffmpeg stdin.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    _embed_ycrcb_inplace(
        ycrcb, symbol_bits, content_id, author_public_key,
        segment_idx, pepper, use_jnd_adaptive, jnd_params,
    )
    return (
        ycrcb[:, :, 0].tobytes(),
        ycrcb[::2, ::2, 2].tobytes(),  # U = Cb
        ycrcb[::2, ::2, 1].tobytes(),  # V = Cr
    )


def frame_to_yuvj420_planes(frame: np.ndarray) -> tuple[bytes, bytes, bytes]:
    """
    BGR frame → (Y, U, V) bytes for an ffmpeg yuvj420p pipe.

    Full-range YCrCb (matches yuvj420p) via a single cv2 color conversion,
    then plane subsampling in numpy. Use this for frames that don't need
    watermark embedding — companion to `embed_video_frame_yuvj420_planes`.
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return (
        ycrcb[:, :, 0].tobytes(),
        ycrcb[::2, ::2, 2].tobytes(),
        ycrcb[::2, ::2, 1].tobytes(),
    )


def frame_to_yuv420(frame: np.ndarray) -> bytes:
    """
    [Deprecated] BGR → YUV I420 bytes via cv2.COLOR_BGR2YUV_I420.

    Retained for callers outside the signing hot loop. Produces BT.601
    limited-range Y (16-235), which mismatches ffmpeg's yuvj420p full-range
    pipe — prefer `frame_to_yuvj420_planes` for new code.
    """
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    return yuv.tobytes()


def extract_segment(
    frames: list[np.ndarray],
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
    use_jnd_adaptive: bool = False,
    jnd_params: VideoEmbeddingParams | None = None,
) -> SegmentWIDResult:
    """
    Extracts 8-bit RS symbol from a segment using majority vote across frames.

    When use_jnd_adaptive=True, reproduces the same per-block QIM step that
    embed_video_frame used, derived from the received block's mean luminance.
    Since embedding only modifies AC coefficients, the DC-based mean is stable
    under H.264 (error < ±2 px, absorbed by quantize_to=4.0).
    """
    coeffs = _coeff_set(content_id, author_public_key, segment_idx, pepper)

    # Unpack JND params if adaptive
    if use_jnd_adaptive and jnd_params is not None:
        step_base = jnd_params.qim_step_base
        step_min = jnd_params.qim_step_min
        step_max = jnd_params.qim_step_max
        quantize_to = jnd_params.qim_quantize_to
    else:
        step_base = step_min = step_max = QIM_STEP_WID
        quantize_to = 1.0

    # Extract block selection params
    oversample = jnd_params.block_oversample if jnd_params else 1
    min_var = jnd_params.min_block_variance if jnd_params else 0.0

    # Accumulate votes per bit position: votes[bit_pos][0/1]
    votes = np.zeros((8, 2), dtype=np.int32)

    for frame in frames:
        h, w = frame.shape[:2]
        candidates = _select_blocks(
            h, w, content_id, author_public_key, segment_idx, pepper,
            oversample=oversample,
        )
        if not candidates:
            continue

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_float = ycrcb[:, :, 0].astype(np.float32)

        # Iterate ALL candidates; use cand_idx for bit mapping (stable
        # across embed/extract even when borderline blocks disagree).
        for cand_idx, (y0, x0) in enumerate(candidates):
            if y0 + BLOCK_SIZE > h or x0 + BLOCK_SIZE > w:
                continue
            block = y_float[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE].copy()

            # Skip low-variance blocks inline
            if min_var > 0 and float(np.var(block)) < min_var:
                continue

            bit_pos = cand_idx % 8
            dct_block = cv2.dct(block)

            if use_jnd_adaptive:
                block_mean = float(np.mean(block))
                step = _compute_adaptive_step(
                    block_mean, step_base, step_min, step_max, quantize_to
                )
            else:
                step = QIM_STEP_WID

            for cr, cc in coeffs:
                extracted = _qim_extract(dct_block[cr, cc], step)
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


@lru_cache(maxsize=512)
def _coeff_set(
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
) -> tuple[tuple[int, int], ...]:
    """
    Returns coefficient list for this segment.
    Always includes MANDATORY_COEFFS. May include 0, 1, or 2 OPTIONAL_COEFFS.

    Cached per (content_id, pubkey, segment_idx, pepper): when the embed /
    extract loops call this once per frame, the LRU cache collapses those
    calls back to one HMAC+RNG per segment.
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
    return tuple(result)


@lru_cache(maxsize=512)
def _select_blocks(
    height: int,
    width: int,
    content_id: str,
    author_public_key: str,
    segment_idx: int,
    pepper: bytes,
    n_blocks: int = N_WID_BLOCKS_PER_SEGMENT,
    oversample: int = 1,
) -> tuple[tuple[int, int], ...]:
    """
    Deterministic block selection (normalized coordinates → pixel coords).
    Same normalization invariant as pilot_tone._select_blocks.

    When oversample > 1, generates n_blocks * oversample candidates.
    Caller is responsible for filtering down to n_blocks.

    Cached: the block set is a pure function of (h, w, content_id, pubkey,
    segment_idx, pepper, n_blocks, oversample).  Within a signing job the
    same segment's blocks are re-requested once per frame — the LRU cache
    turns that into one RNG per segment instead of one per frame.
    """
    n_rows = height // BLOCK_SIZE
    n_cols = width // BLOCK_SIZE
    total = n_rows * n_cols

    n_candidates = min(n_blocks * oversample, total)
    if n_candidates == 0:
        return ()

    msg = f"wid_blocks|{content_id}|{author_public_key}|{segment_idx}".encode()
    digest = hmac.new(pepper, msg, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)

    norm_positions = rng.random((n_candidates, 2))
    result = []
    for ny, nx in norm_positions:
        row = min(int(ny * n_rows), n_rows - 1)
        col = min(int(nx * n_cols), n_cols - 1)
        result.append((row * BLOCK_SIZE, col * BLOCK_SIZE))
    return tuple(result)


_MIN_USABLE_BLOCKS = 16


def _filter_blocks_by_variance(
    candidates: list[tuple[int, int]],
    y_channel: np.ndarray,
    min_variance: float,
    n_blocks: int,
) -> list[tuple[int, int]]:
    """Filter candidate blocks by luma variance, preserving HMAC order.

    Blocks with variance >= min_variance are kept (in original order).
    If fewer than _MIN_USABLE_BLOCKS pass, falls back to top-N by variance
    (re-sorted into original HMAC order).

    HMAC order preservation is critical: block_idx % 8 determines bit position.

    Args:
        candidates: block positions from _select_blocks (HMAC-ordered)
        y_channel: Y channel as float32, shape (H, W)
        min_variance: minimum block variance threshold
        n_blocks: maximum number of blocks to return

    Returns:
        Filtered block list, subsequence of candidates, len <= n_blocks.
    """
    if min_variance <= 0:
        return candidates[:n_blocks]

    # Compute variance for each candidate
    variances = []
    for y0, x0 in candidates:
        block = y_channel[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
        variances.append(float(np.var(block)))

    # Filter by threshold, preserving order
    passed = [
        (i, candidates[i])
        for i, v in enumerate(variances)
        if v >= min_variance
    ]

    if len(passed) >= _MIN_USABLE_BLOCKS:
        return [pos for _, pos in passed[:n_blocks]]

    # Fallback: top-N by variance, re-sorted into HMAC order
    indexed = sorted(
        range(len(candidates)),
        key=lambda i: variances[i],
        reverse=True,
    )
    top_n = sorted(indexed[:max(n_blocks, _MIN_USABLE_BLOCKS)])
    return [candidates[i] for i in top_n[:n_blocks]]


def _compute_adaptive_step(
    block_mean_luminance: float,
    step_base: float = QIM_STEP_ADAPTIVE_BASE,
    step_min: float = QIM_STEP_ADAPTIVE_MIN,
    step_max: float = QIM_STEP_ADAPTIVE_MAX,
    quantize_to: float = QIM_STEP_QUANTIZE_TO,
) -> float:
    """
    Compute adaptive QIM step using Chou-Li JND luminance model.

    JND formula:
      bg <= 127: jnd = 17 * (1 - sqrt(bg / 127)) + 3
      bg >  127: jnd = (3 / 128) * (bg - 127) + 3

    The step is normalized to mid-gray baseline (jnd=3 at bg=127),
    then quantized to multiples of quantize_to for H.264 robustness.

    Args:
        block_mean_luminance: mean Y-channel value of the 4×4 block [0, 255]
        step_base: QIM step at mid-gray (jnd=3)
        step_min:  minimum allowed step
        step_max:  maximum allowed step
        quantize_to: step is rounded to nearest multiple of this value

    Returns:
        Adaptive QIM step (float, multiple of quantize_to)
    """
    bg = float(np.clip(block_mean_luminance, 0.0, 255.0))

    if bg <= JND_BASE_LUMINANCE:
        jnd = 17.0 * (1.0 - np.sqrt(bg / JND_BASE_LUMINANCE)) + 3.0
    else:
        jnd = (3.0 / 128.0) * (bg - JND_BASE_LUMINANCE) + 3.0

    # Normalize: jnd=3 → step_base, jnd>3 → larger step
    step_raw = step_base * (jnd / 3.0)

    # Clamp to valid range
    step_clamped = float(np.clip(step_raw, step_min, step_max))

    # Quantize to nearest multiple of quantize_to
    return round(step_clamped / quantize_to) * quantize_to


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
