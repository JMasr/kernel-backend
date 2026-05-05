"""Unit tests for the algorithm router engine module."""
from __future__ import annotations

import pytest

from kernel_backend.core.domain.content_profile import (
    ContentProfile,
    RoutingDecision,
    routing_decision_from_dict,
    routing_decision_to_dict,
)
from kernel_backend.engine.audio.algorithm_router import (
    route,
    routing_decision_to_audio_params,
    _ROUTING_TABLE,
)


def _make_profile(
    content_type: str = "speech",
    confidence: float = 0.95,
) -> ContentProfile:
    """Create a minimal ContentProfile for testing."""
    return ContentProfile(
        content_type=content_type,
        confidence=confidence,
        features={"spectral_flatness_mean": 0.05},
        descriptor_version="1.0.0",
        code_hash="sha256:test1234test1234",
    )


class TestRouting:
    """Each content type maps to expected parameters."""

    def test_speech_routes_to_detail_band(self):
        rd = route(_make_profile("speech", 0.95))
        assert rd.content_type == "speech"
        assert rd.parameters["target_subband"] == "detail"
        assert rd.parameters["dwt_levels"] == (2,)

    def test_music_routes_to_approximation_band(self):
        rd = route(_make_profile("music", 0.90))
        assert rd.content_type == "music"
        assert rd.parameters["target_subband"] == "approximation"
        assert rd.parameters["dwt_levels"] == (5,)
        assert rd.parameters["target_snr_db"] == -16.0
        assert rd.parameters["pn_sequence_length"] == 2048

    def test_classical_routes_to_approximation_band(self):
        rd = route(_make_profile("classical", 0.85))
        assert rd.content_type == "classical"
        assert rd.parameters["target_subband"] == "approximation"
        assert rd.parameters["dwt_levels"] == (5,)

    def test_ambient_routes_to_level_4_approximation(self):
        rd = route(_make_profile("ambient", 0.92))
        assert rd.content_type == "ambient"
        assert rd.parameters["target_subband"] == "approximation"
        assert rd.parameters["dwt_levels"] == (4,)

    def test_silence_routes_to_level_4_approximation(self):
        rd = route(_make_profile("silence", 0.99))
        assert rd.content_type == "silence"
        assert rd.parameters["target_subband"] == "approximation"
        assert rd.parameters["dwt_levels"] == (4,)

    def test_all_content_types_have_routing_entries(self):
        """Every valid content type must have a routing table entry."""
        for ct in ("speech", "music", "classical", "ambient", "silence"):
            assert ct in _ROUTING_TABLE, f"Missing routing entry for {ct}"


class TestFallback:
    """Low confidence or unknown types fall back to speech."""

    def test_low_confidence_falls_back_to_speech(self):
        rd = route(_make_profile("music", 0.3))
        assert rd.content_type == "speech"
        assert rd.parameters["target_subband"] == "detail"
        assert "low confidence" in rd.reason

    def test_confidence_boundary_at_0_5(self):
        """Exactly 0.5 should NOT trigger fallback."""
        rd = route(_make_profile("music", 0.5))
        assert rd.content_type == "music"

    def test_confidence_just_below_0_5(self):
        rd = route(_make_profile("music", 0.49))
        assert rd.content_type == "speech"


class TestRoutingDecisionToAudioParams:
    """Convert RoutingDecision to AudioEmbeddingParams."""

    def test_speech_params(self):
        rd = route(_make_profile("speech", 0.95))
        ap = routing_decision_to_audio_params(rd)
        assert ap.dwt_levels == (2,)
        assert ap.target_subband == "detail"
        assert ap.chips_per_bit == 32
        assert ap.target_snr_db == -12.0
        assert ap.psychoacoustic is False

    def test_music_params(self):
        rd = route(_make_profile("music", 0.90))
        ap = routing_decision_to_audio_params(rd)
        assert ap.dwt_levels == (5,)
        assert ap.target_subband == "approximation"
        assert ap.target_snr_db == -16.0
        assert ap.pn_sequence_length == 2048

    def test_round_trip_all_types(self):
        """route() -> routing_decision_to_audio_params() should work for all types."""
        for ct in ("speech", "music", "classical", "ambient", "silence"):
            rd = route(_make_profile(ct, 0.95))
            ap = routing_decision_to_audio_params(rd)
            assert ap.dwt_levels is not None
            assert ap.target_subband in ("detail", "approximation")


class TestSerialization:
    """Round-trip RoutingDecision through dict serialization."""

    def test_round_trip(self):
        rd = route(_make_profile("music", 0.90))
        d = routing_decision_to_dict(rd)
        rd2 = routing_decision_from_dict(d)
        assert rd2.algorithm_id == rd.algorithm_id
        assert rd2.content_type == rd.content_type
        assert rd2.confidence == rd.confidence
        assert rd2.parameters == rd.parameters

    def test_reason_preserved(self):
        rd = route(_make_profile("speech", 0.95))
        d = routing_decision_to_dict(rd)
        rd2 = routing_decision_from_dict(d)
        assert rd2.reason == rd.reason


class TestRoutingDecisionFields:
    """RoutingDecision has expected structure."""

    def test_algorithm_id_present(self):
        rd = route(_make_profile("music", 0.90))
        assert rd.algorithm_id == "dwt_dsss_v2"

    def test_reason_is_descriptive(self):
        rd = route(_make_profile("music", 0.90))
        assert "music" in rd.reason
        assert "approximation" in rd.reason
