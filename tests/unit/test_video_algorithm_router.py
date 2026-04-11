"""Unit tests for the video algorithm router engine module."""
from __future__ import annotations

import pytest

from kernel_backend.core.domain.video_content_profile import (
    VideoContentProfile,
    VideoRoutingDecision,
    video_routing_decision_from_dict,
    video_routing_decision_to_dict,
)
from kernel_backend.engine.video.algorithm_router import (
    route,
    video_routing_decision_to_video_params,
    _ROUTING_TABLE,
)


def _make_profile(
    content_type: str = "normal",
    confidence: float = 0.95,
) -> VideoContentProfile:
    """Create a minimal VideoContentProfile for testing."""
    return VideoContentProfile(
        content_type=content_type,
        confidence=confidence,
        features={"mean_luminance": 128.0},
        descriptor_version="1.0.0",
        code_hash="sha256:test1234test1234",
    )


class TestRouting:
    """Each content type maps to expected parameters."""

    def test_normal_routes_to_production_defaults(self):
        rd = route(_make_profile("normal", 0.95))
        assert rd.content_type == "normal"
        assert rd.parameters["qim_step_base"] == 64.0
        assert rd.parameters["qim_step_min"] == 44.0
        assert rd.parameters["qim_step_max"] == 128.0

    def test_dark_routes_to_bounded_jnd(self):
        rd = route(_make_profile("dark", 0.90))
        assert rd.content_type == "dark"
        assert rd.parameters["qim_step_base"] == 80.0
        assert rd.parameters["qim_step_min"] == 64.0
        assert rd.parameters["qim_step_max"] == 96.0
        assert rd.parameters["min_block_variance"] == 100.0
        assert rd.parameters["block_oversample"] == 16

    def test_bright_routes_to_smaller_steps(self):
        rd = route(_make_profile("bright", 0.90))
        assert rd.content_type == "bright"
        assert rd.parameters["qim_step_base"] == 56.0
        assert rd.parameters["qim_step_min"] == 40.0

    def test_high_motion_routes_to_larger_base(self):
        rd = route(_make_profile("high_motion", 0.90))
        assert rd.content_type == "high_motion"
        assert rd.parameters["qim_step_base"] == 72.0

    def test_static_routes_to_smaller_steps(self):
        rd = route(_make_profile("static", 0.90))
        assert rd.content_type == "static"
        assert rd.parameters["qim_step_base"] == 56.0
        assert rd.parameters["qim_step_max"] == 100.0

    def test_all_content_types_have_routing_entries(self):
        for ct in ("normal", "dark", "bright", "high_motion", "static"):
            assert ct in _ROUTING_TABLE, f"Missing routing entry for {ct}"

    def test_all_entries_have_jnd_adaptive_true(self):
        for ct, entry in _ROUTING_TABLE.items():
            assert entry["jnd_adaptive"] is True, f"{ct} should have jnd_adaptive=True"

    def test_non_dark_types_have_default_block_selection(self):
        for ct in ("normal", "bright", "high_motion", "static"):
            rd = route(_make_profile(ct, 0.95))
            vp = video_routing_decision_to_video_params(rd)
            assert vp.min_block_variance == 0.0, f"{ct} should have default min_block_variance"
            assert vp.block_oversample == 1, f"{ct} should have default block_oversample"


class TestFallback:
    """Low confidence or unknown types fall back to normal."""

    def test_low_confidence_falls_back_to_normal(self):
        rd = route(_make_profile("dark", 0.5))
        assert rd.content_type == "normal"
        assert rd.parameters["qim_step_base"] == 64.0
        assert "low confidence" in rd.reason

    def test_confidence_boundary_at_0_7(self):
        """Exactly 0.7 should NOT trigger fallback."""
        rd = route(_make_profile("dark", 0.7))
        assert rd.content_type == "dark"

    def test_confidence_just_below_0_7(self):
        rd = route(_make_profile("dark", 0.69))
        assert rd.content_type == "normal"


class TestVideoParams:
    """Convert VideoRoutingDecision to VideoEmbeddingParams."""

    def test_normal_params(self):
        rd = route(_make_profile("normal", 0.95))
        vp = video_routing_decision_to_video_params(rd)
        assert vp.jnd_adaptive is True
        assert vp.qim_step_base == 64.0
        assert vp.qim_step_min == 44.0
        assert vp.qim_step_max == 128.0
        assert vp.qim_quantize_to == 4.0

    def test_dark_params(self):
        rd = route(_make_profile("dark", 0.90))
        vp = video_routing_decision_to_video_params(rd)
        assert vp.qim_step_base == 80.0
        assert vp.qim_step_min == 64.0
        assert vp.qim_step_max == 96.0
        assert vp.min_block_variance == 100.0
        assert vp.block_oversample == 16

    def test_round_trip_all_types(self):
        for ct in ("normal", "dark", "bright", "high_motion", "static"):
            rd = route(_make_profile(ct, 0.95))
            vp = video_routing_decision_to_video_params(rd)
            assert vp.jnd_adaptive is True
            assert vp.qim_step_base > 0
            assert vp.qim_step_min < vp.qim_step_max


class TestSerialization:
    """Round-trip VideoRoutingDecision through dict serialization."""

    def test_round_trip(self):
        rd = route(_make_profile("dark", 0.90))
        d = video_routing_decision_to_dict(rd)
        rd2 = video_routing_decision_from_dict(d)
        assert rd2.algorithm_id == rd.algorithm_id
        assert rd2.content_type == rd.content_type
        assert rd2.confidence == rd.confidence
        assert rd2.parameters == rd.parameters

    def test_reason_preserved(self):
        rd = route(_make_profile("dark", 0.90))
        d = video_routing_decision_to_dict(rd)
        rd2 = video_routing_decision_from_dict(d)
        assert rd2.reason == rd.reason


class TestRoutingDecisionFields:
    def test_algorithm_id_present(self):
        rd = route(_make_profile("dark", 0.90))
        assert rd.algorithm_id == "dct_qim_v1"

    def test_reason_is_descriptive(self):
        rd = route(_make_profile("dark", 0.90))
        assert "dark" in rd.reason
        assert "qim_step_base=80" in rd.reason
