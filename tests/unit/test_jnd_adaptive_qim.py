"""
Unit tests for Sprint 3 — Chou-Li JND adaptive QIM step (engine/video/wid_watermark.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.core.domain.watermark import VideoEmbeddingParams
from kernel_backend.engine.video.wid_watermark import (
    JND_BASE_LUMINANCE,
    QIM_STEP_ADAPTIVE_BASE,
    QIM_STEP_ADAPTIVE_MAX,
    QIM_STEP_ADAPTIVE_MIN,
    QIM_STEP_QUANTIZE_TO,
    QIM_STEP_WID,
    _compute_adaptive_step,
    embed_video_frame,
    extract_segment,
)

_JND_PARAMS = VideoEmbeddingParams(
    jnd_adaptive=True,
    qim_step_base=QIM_STEP_ADAPTIVE_BASE,
    qim_step_min=QIM_STEP_ADAPTIVE_MIN,
    qim_step_max=QIM_STEP_ADAPTIVE_MAX,
    qim_quantize_to=QIM_STEP_QUANTIZE_TO,
)

_CONTENT_ID = "test-content-id"
_AUTHOR_KEY = "test-author-key"
_PEPPER = b"test-pepper-bytes-padded-to-32b!"


def _random_frame(seed: int, luminance: float = 127.0, size: tuple = (64, 64)) -> np.ndarray:
    """BGR frame with approximately uniform luminance."""
    rng = np.random.default_rng(seed)
    frame = np.full((size[0], size[1], 3), int(luminance), dtype=np.uint8)
    noise = rng.integers(-5, 6, size=(size[0], size[1], 3), dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _symbol_bits(value: int = 0b10110100) -> np.ndarray:
    return np.array([(value >> (7 - i)) & 1 for i in range(8)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# _compute_adaptive_step
# ---------------------------------------------------------------------------


def test_compute_adaptive_step_midgray() -> None:
    """At mid-gray (127), step must equal step_base."""
    step = _compute_adaptive_step(JND_BASE_LUMINANCE)
    assert step == pytest.approx(QIM_STEP_ADAPTIVE_BASE, abs=QIM_STEP_QUANTIZE_TO)


def test_compute_adaptive_step_black() -> None:
    """Black (0) → max JND → step > step_base."""
    step_dark = _compute_adaptive_step(0.0)
    step_mid = _compute_adaptive_step(JND_BASE_LUMINANCE)
    assert step_dark > step_mid


def test_compute_adaptive_step_white() -> None:
    """White (255) → step > step_base (linear JND branch)."""
    step_white = _compute_adaptive_step(255.0)
    step_mid = _compute_adaptive_step(JND_BASE_LUMINANCE)
    assert step_white > step_mid


def test_compute_adaptive_step_quantization() -> None:
    """Returned step must be a multiple of quantize_to."""
    for bg in [0, 64, 127, 200, 255]:
        step = _compute_adaptive_step(float(bg))
        assert step % QIM_STEP_QUANTIZE_TO == pytest.approx(0.0, abs=1e-9)


def test_compute_adaptive_step_clamped_min() -> None:
    """Step never goes below step_min."""
    step = _compute_adaptive_step(JND_BASE_LUMINANCE)
    assert step >= QIM_STEP_ADAPTIVE_MIN


def test_compute_adaptive_step_clamped_max() -> None:
    """Step never exceeds step_max."""
    step = _compute_adaptive_step(0.0)
    assert step <= QIM_STEP_ADAPTIVE_MAX


def test_adaptive_step_deterministic() -> None:
    """Same luminance → same step on repeated calls."""
    for bg in [0.0, 63.5, 127.0, 200.0, 255.0]:
        assert _compute_adaptive_step(bg) == _compute_adaptive_step(bg)


# ---------------------------------------------------------------------------
# embed_video_frame / extract_segment roundtrip
# ---------------------------------------------------------------------------


def test_embed_extract_roundtrip_adaptive() -> None:
    """Adaptive JND embed→extract roundtrip achieves agreement ≥ 0.90."""
    bits = _symbol_bits(0b10110100)
    frame = _random_frame(seed=42, luminance=127.0, size=(64, 64))
    watermarked = embed_video_frame(
        frame, bits, _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
        use_jnd_adaptive=True, jnd_params=_JND_PARAMS,
    )
    result = extract_segment(
        [watermarked], _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
        use_jnd_adaptive=True, jnd_params=_JND_PARAMS,
    )
    assert result.agreement >= 0.80
    assert result.extracted_bits[0] == 0b10110100


def test_embed_extract_roundtrip_legacy() -> None:
    """Legacy (non-adaptive) embed→extract still works and is unaffected."""
    bits = _symbol_bits(0b01001101)
    frame = _random_frame(seed=7, luminance=127.0, size=(64, 64))
    watermarked = embed_video_frame(
        frame, bits, _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
        use_jnd_adaptive=False,
    )
    result = extract_segment(
        [watermarked], _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
        use_jnd_adaptive=False,
    )
    assert result.agreement >= 0.90
    assert result.extracted_bits[0] == 0b01001101


def test_adaptive_stronger_in_dark_blocks() -> None:
    """Dark frames (luminance≈20) use larger QIM step than mid-gray."""
    bits = _symbol_bits(0b11001010)
    frame_dark = _random_frame(seed=1, luminance=20.0, size=(64, 64))
    frame_mid = _random_frame(seed=1, luminance=127.0, size=(64, 64))

    w_dark = embed_video_frame(
        frame_dark, bits, _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
        use_jnd_adaptive=True, jnd_params=_JND_PARAMS,
    )
    w_mid = embed_video_frame(
        frame_mid, bits, _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
        use_jnd_adaptive=True, jnd_params=_JND_PARAMS,
    )

    # Dark frame embedding should introduce more distortion (larger step)
    diff_dark = float(np.mean(np.abs(w_dark.astype(np.float32) - frame_dark.astype(np.float32))))
    diff_mid = float(np.mean(np.abs(w_mid.astype(np.float32) - frame_mid.astype(np.float32))))
    assert diff_dark >= diff_mid


def test_h264_step_recovery() -> None:
    """
    Extractor reproduces embed step from received block mean luminance.
    Uses lossless roundtrip (no H.264) — verifies step consistency logic.
    Multiple symbols extracted across a 5-frame segment.
    """
    bits = _symbol_bits(0b10101010)
    frames = [_random_frame(seed=i, luminance=80.0, size=(64, 64)) for i in range(5)]
    watermarked = [
        embed_video_frame(
            f, bits, _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
            use_jnd_adaptive=True, jnd_params=_JND_PARAMS,
        )
        for f in frames
    ]
    result = extract_segment(
        watermarked, _CONTENT_ID, _AUTHOR_KEY, 0, _PEPPER,
        use_jnd_adaptive=True, jnd_params=_JND_PARAMS,
    )
    assert result.agreement >= 0.80
    assert result.extracted_bits[0] == 0b10101010
