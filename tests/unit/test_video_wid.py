"""
Unit tests for engine/video/wid_watermark.py — RS symbol embedding via QIM on 4×4 DCT.

Uses random-noise frames (never uniform-color frames).
"""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.core.domain.watermark import VideoEmbeddingParams
from kernel_backend.engine.video.wid_watermark import (
    BLOCK_SIZE,
    SegmentWIDResult,
    _coeff_set,
    _filter_blocks_by_variance,
    _select_blocks,
    _MIN_USABLE_BLOCKS,
    embed_segment,
    embed_video_frame,
    extract_segment,
    MANDATORY_COEFFS,
    N_WID_BLOCKS_PER_SEGMENT,
    WID_AGREEMENT_THRESHOLD,
)

CONTENT_ID_A = "test-content-id-alpha"
CONTENT_ID_B = "test-content-id-beta"
PUBKEY_A = "test-pubkey-material-AAA"
PUBKEY_B = "test-pubkey-material-BBB"
PEPPER = b"test-pepper-32-bytes-padded-AAA!"
SYMBOL_BITS = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)


def _random_frames(n: int = 5, height: int = 540, width: int = 960, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(n)]


def test_embed_extract_roundtrip_single_segment():
    """5 random frames, embed 8-bit symbol, extract. Agreement >= 0.90 on clean."""
    frames = _random_frames()
    embedded = embed_segment(frames, SYMBOL_BITS, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    result = extract_segment(embedded, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    assert result.agreement >= 0.90, f"agreement={result.agreement:.3f} < 0.90"
    assert not result.erasure


def test_agreement_threshold():
    """Unsigned frames → agreement near 0.50 (random baseline)."""
    frames = _random_frames()
    result = extract_segment(frames, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    assert result.agreement < 0.60, f"unsigned agreement={result.agreement:.3f} too high"


def test_wrong_seed_scores_near_random():
    """Embed with (A, A), extract with (B, B) → agreement near 0.50."""
    frames = _random_frames()
    embedded = embed_segment(frames, SYMBOL_BITS, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    result = extract_segment(embedded, CONTENT_ID_B, PUBKEY_B, 0, PEPPER)
    assert result.agreement < 0.60, f"wrong-seed agreement={result.agreement:.3f} too high"


def test_mandatory_coefficients_always_present():
    """_coeff_set always includes (0,1) and (1,0) for 100 different segment indices."""
    for seg_idx in range(100):
        coeffs = _coeff_set(CONTENT_ID_A, PUBKEY_A, seg_idx, PEPPER)
        for mc in MANDATORY_COEFFS:
            assert mc in coeffs, f"segment {seg_idx}: mandatory coeff {mc} missing"


def test_unsigned_agreement_well_below_embedded():
    """
    Extract from unembedded frames → agreement well below embedded level.
    Random baseline is ~0.50 + O(1/sqrt(votes)), so with finite votes per bit
    the agreement can be 0.52–0.58 even for random data. The key property is
    the separation from embedded agreement (~0.95).
    """
    frames = _random_frames(seed=42)
    # Embedded: should be high
    embedded = embed_segment(frames, SYMBOL_BITS, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    result_emb = extract_segment(embedded, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    # Unsigned: should be near random
    result_raw = extract_segment(frames, CONTENT_ID_A, PUBKEY_A, 0, PEPPER)
    # At least 30 percentage points separation
    gap = result_emb.agreement - result_raw.agreement
    assert gap >= 0.30, (
        f"insufficient gap: embedded={result_emb.agreement:.3f} "
        f"unsigned={result_raw.agreement:.3f} gap={gap:.3f}"
    )


# -- Block filtering tests ---------------------------------------------------

def _dark_frame_with_patches(
    height: int = 540, width: int = 960, seed: int = 42,
) -> np.ndarray:
    """Dark frame (luma ~10) with bright textured patches."""
    rng = np.random.default_rng(seed)
    # Dark base
    frame = np.full((height, width, 3), 10, dtype=np.uint8)
    # Add textured patches every 120px (high variance)
    for y in range(0, height - 32, 120):
        for x in range(0, width - 32, 120):
            patch = rng.integers(60, 200, (32, 32, 3), dtype=np.uint8)
            frame[y:y + 32, x:x + 32] = patch
    return frame


class TestFilterBlocksNoFiltering:
    """min_variance=0 produces same behavior as unfiltered."""

    def test_no_filtering_returns_first_n(self):
        candidates = [(i * 4, 0) for i in range(200)]
        y = np.zeros((800, 4), dtype=np.float32)
        result = _filter_blocks_by_variance(candidates, y, 0.0, 128)
        assert result == candidates[:128]

    def test_negative_variance_no_filtering(self):
        candidates = [(i * 4, 0) for i in range(50)]
        y = np.zeros((200, 4), dtype=np.float32)
        result = _filter_blocks_by_variance(candidates, y, -1.0, 50)
        assert result == candidates[:50]


class TestFilterBlocksRemovesFlatBlocks:
    """Blocks with low variance are removed."""

    def test_mixed_frame(self):
        import cv2
        frame = _dark_frame_with_patches()
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_float = ycrcb[:, :, 0].astype(np.float32)

        # Generate candidates
        candidates = _select_blocks(
            frame.shape[0], frame.shape[1],
            CONTENT_ID_A, PUBKEY_A, 0, PEPPER,
            oversample=4,
        )
        filtered = _filter_blocks_by_variance(
            candidates, y_float, 100.0, N_WID_BLOCKS_PER_SEGMENT,
        )
        # Filtered should be fewer than candidates
        assert len(filtered) <= len(candidates)
        # All filtered blocks should have variance >= 100
        for y0, x0 in filtered:
            block = y_float[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
            assert np.var(block) >= 100.0


class TestFilterBlocksPreservesHmacOrder:
    """Filtered list must be a subsequence of candidates."""

    def test_subsequence(self):
        import cv2
        frame = _dark_frame_with_patches()
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_float = ycrcb[:, :, 0].astype(np.float32)

        candidates = _select_blocks(
            frame.shape[0], frame.shape[1],
            CONTENT_ID_A, PUBKEY_A, 0, PEPPER,
            oversample=4,
        )
        filtered = _filter_blocks_by_variance(
            candidates, y_float, 100.0, N_WID_BLOCKS_PER_SEGMENT,
        )
        # filtered must be a subsequence of candidates
        it = iter(candidates)
        for pos in filtered:
            while True:
                c = next(it, None)
                assert c is not None, f"{pos} not found in candidates order"
                if c == pos:
                    break


class TestFilterBlocksFallback:
    """When too few blocks pass, fallback to top-N by variance."""

    def test_flat_frame_fallback(self):
        # Nearly uniform frame — very few blocks will have high variance
        y = np.full((540, 960), 10.0, dtype=np.float32)
        candidates = _select_blocks(
            540, 960, CONTENT_ID_A, PUBKEY_A, 0, PEPPER,
            oversample=4,
        )
        filtered = _filter_blocks_by_variance(
            candidates, y, 100.0, N_WID_BLOCKS_PER_SEGMENT,
        )
        # Should still get at least _MIN_USABLE_BLOCKS via fallback
        assert len(filtered) >= min(_MIN_USABLE_BLOCKS, len(candidates))


class TestEmbedExtractWithFiltering:
    """Round-trip on dark frame with bright patches using variance filtering."""

    def test_roundtrip_dark_with_patches(self):
        import cv2
        frame = _dark_frame_with_patches()
        params = VideoEmbeddingParams(
            jnd_adaptive=True,
            qim_step_base=80.0,
            qim_step_min=64.0,
            qim_step_max=96.0,
            qim_quantize_to=4.0,
            min_block_variance=100.0,
            block_oversample=4,
        )
        embedded = embed_video_frame(
            frame, SYMBOL_BITS, CONTENT_ID_A, PUBKEY_A, 0, PEPPER,
            use_jnd_adaptive=True, jnd_params=params,
        )
        result = extract_segment(
            [embedded], CONTENT_ID_A, PUBKEY_A, 0, PEPPER,
            use_jnd_adaptive=True, jnd_params=params,
        )
        assert result.agreement >= 0.85, f"agreement={result.agreement:.3f}"
        assert not result.erasure
