"""Unit tests for engine/audio/segment_scorer.py."""
from __future__ import annotations

import numpy as np
import pytest

from kernel_backend.engine.audio.segment_scorer import (
    SegmentScore,
    score_segment,
    score_segments,
    select_best,
    DEFAULT_MIN_SCORE_DB,
)

SR = 44100
SEG_LEN = int(SR * 2.0)  # 2-second segments


def _silence(seed: int = 0) -> np.ndarray:
    return np.zeros(SEG_LEN, dtype=np.float32)


def _noise(seed: int = 42, amplitude: float = 0.3) -> np.ndarray:
    return (np.random.default_rng(seed)
            .standard_normal(SEG_LEN).astype(np.float32) * amplitude)


def _quiet_noise(seed: int = 99) -> np.ndarray:
    """Very quiet noise — below the -40 dBFS floor."""
    return _noise(seed, amplitude=1e-4)


# ── score_segment ────────────────────────────────────────────────────────────

class TestScoreSegment:

    def test_silence_scores_lowest_rms(self) -> None:
        sc = score_segment(_silence(), SR, dwt_level=2, target_subband="detail")
        assert sc.rms_db < -100  # effectively -inf dBFS

    def test_noise_has_reasonable_rms(self) -> None:
        sc = score_segment(_noise(), SR, dwt_level=2, target_subband="detail")
        assert -20 < sc.rms_db < 0

    def test_noise_has_high_spectral_flatness(self) -> None:
        sc = score_segment(_noise(), SR, dwt_level=2, target_subband="detail")
        assert sc.spectral_flatness > 0.1

    def test_silence_has_zero_spectral_flatness(self) -> None:
        sc = score_segment(_silence(), SR, dwt_level=2, target_subband="detail")
        assert sc.spectral_flatness < 0.01

    def test_band_energy_differs_by_subband(self) -> None:
        seg = _noise(42)
        sc_detail = score_segment(seg, SR, dwt_level=2, target_subband="detail")
        sc_approx = score_segment(seg, SR, dwt_level=2, target_subband="approximation")
        # Different subbands should yield different band energy values
        assert sc_detail.band_energy_db != sc_approx.band_energy_db


# ── score_segments (population normalization) ────────────────────────────────

class TestScoreSegments:

    def test_empty_input(self) -> None:
        assert score_segments(iter([])) == []

    def test_composite_range(self) -> None:
        """Composites should be in [0, 1] after normalization."""
        segs = [
            (0, _silence()),
            (1, _noise(1)),
            (2, _noise(2, amplitude=0.5)),
            (3, _quiet_noise()),
        ]
        scores = score_segments(iter(segs))
        for sc in scores:
            assert 0.0 <= sc.composite <= 1.0, f"seg {sc.index}: {sc.composite}"

    def test_noise_scores_higher_than_silence(self) -> None:
        """White noise segment should score higher than silence."""
        segs = [(0, _silence()), (1, _noise(42))]
        scores = score_segments(iter(segs))
        assert scores[1].composite > scores[0].composite

    def test_ordered_by_index(self) -> None:
        segs = [(5, _noise(5)), (2, _noise(2)), (0, _noise(0))]
        scores = score_segments(iter(segs))
        indices = [sc.index for sc in scores]
        assert indices == sorted(indices)

    def test_single_segment_gets_half_composite(self) -> None:
        """With one segment, all dimensions normalize to 0.5."""
        scores = score_segments(iter([(0, _noise(0))]))
        assert len(scores) == 1
        assert abs(scores[0].composite - 0.5) < 0.01


# ── select_best ──────────────────────────────────────────────────────────────

class TestSelectBest:

    def _make_scores(self, rms_values: list[float]) -> list[SegmentScore]:
        """Build SegmentScores with given RMS values, rest zeroed."""
        return [
            SegmentScore(
                index=i,
                rms_db=rms,
                band_energy_db=rms,  # same for simplicity
                spectral_flatness=0.5,
                transient_density=0.1,
                composite=(rms + 100) / 100,  # simple linear mapping
            )
            for i, rms in enumerate(rms_values)
        ]

    def test_excludes_silence(self) -> None:
        """Segments below min_score_db are excluded."""
        scores = self._make_scores([-200, -30, -10, -5])
        selected = select_best(scores, n_needed=3, min_score_db=-40)
        assert 0 not in selected  # -200 dB is below floor
        assert len(selected) == 3

    def test_returns_sorted_indices(self) -> None:
        scores = self._make_scores([-10, -5, -15, -3, -8])
        selected = select_best(scores, n_needed=3)
        assert selected == sorted(selected)

    def test_fewer_than_needed(self) -> None:
        """When most segments are silent, returns all passing segments."""
        scores = self._make_scores([-200, -200, -200, -5, -200])
        selected = select_best(scores, n_needed=3, min_score_db=-40)
        assert selected == [3]

    def test_all_segments_selected_when_enough(self) -> None:
        """When n_needed == len(scores) and all pass, all are selected."""
        scores = self._make_scores([-10, -5, -15, -3])
        selected = select_best(scores, n_needed=4)
        assert selected == [0, 1, 2, 3]

    def test_picks_highest_composite(self) -> None:
        """Selects segments with highest composite scores."""
        scores = [
            SegmentScore(i, rms_db=-10, band_energy_db=-10,
                         spectral_flatness=0.5, transient_density=0.1,
                         composite=c)
            for i, c in enumerate([0.1, 0.9, 0.3, 0.8, 0.2])
        ]
        selected = select_best(scores, n_needed=2)
        # Should pick indices 1 (0.9) and 3 (0.8), sorted
        assert selected == [1, 3]
