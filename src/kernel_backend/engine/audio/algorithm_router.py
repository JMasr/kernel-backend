"""
Content-adaptive algorithm router for audio watermarking.

Maps a ContentProfile to optimized DWT-DSSS parameters based on a
deterministic routing table. No file I/O, no ML dependencies.

The routing table is a Python dict constant — fully auditable, versionable,
and free of file I/O in the engine layer.
"""
from __future__ import annotations

from kernel_backend.core.domain.content_profile import ContentProfile, RoutingDecision
from kernel_backend.core.domain.watermark import AudioEmbeddingParams

# ── Routing table ───────────────────────────────────────────────────────────
# Maps content_type → optimized DWT-DSSS parameters.
#
# Key parameter changes vs speech baseline:
#   - Music/classical: approximation band at level 5 (0-690 Hz at 44.1 kHz)
#     for robustness against MP3/AAC compression
#   - Ambient/silence: approximation band at level 4 (0-1380 Hz)
#   - Speech: detail band at level 2 (5.5-11 kHz) — unchanged from current
#
# Frame length changes (100-200ms for music) are deferred — they require
# cross-cutting changes to segment duration, RS layout, and fingerprinting.

_ROUTING_TABLE: dict[str, dict] = {
    "speech": {
        "algorithm_id": "dwt_dsss_v2",
        "dwt_levels": (2,),
        "target_subband": "detail",
        "chips_per_bit": 32,
        "target_snr_db": -20.0,
        "frame_length_ms": 0.0,
        "pn_sequence_length": 0,
        "psychoacoustic": False,
        "safety_margin_db": 12.0,
    },
    "music": {
        "algorithm_id": "dwt_dsss_v2",
        "dwt_levels": (5,),
        "target_subband": "approximation",
        "chips_per_bit": 32,
        "target_snr_db": -16.0,
        "frame_length_ms": 0.0,
        "pn_sequence_length": 2048,
        "psychoacoustic": False,
        "safety_margin_db": 12.0,
    },
    "classical": {
        "algorithm_id": "dwt_dsss_v2",
        "dwt_levels": (5,),
        "target_subband": "approximation",
        "chips_per_bit": 32,
        "target_snr_db": -18.0,
        "frame_length_ms": 0.0,
        "pn_sequence_length": 2048,
        "psychoacoustic": False,
        "safety_margin_db": 12.0,
    },
    "ambient": {
        "algorithm_id": "dwt_dsss_v2",
        "dwt_levels": (4,),
        "target_subband": "approximation",
        "chips_per_bit": 32,
        "target_snr_db": -18.0,
        "frame_length_ms": 0.0,
        "pn_sequence_length": 1024,
        "psychoacoustic": False,
        "safety_margin_db": 12.0,
    },
    "silence": {
        "algorithm_id": "dwt_dsss_v2",
        "dwt_levels": (4,),
        "target_subband": "approximation",
        "chips_per_bit": 32,
        "target_snr_db": -18.0,
        "frame_length_ms": 0.0,
        "pn_sequence_length": 1024,
        "psychoacoustic": False,
        "safety_margin_db": 12.0,
    },
}

_FALLBACK_CONTENT_TYPE = "speech"
_CONFIDENCE_THRESHOLD = 0.5


def route(profile: ContentProfile) -> RoutingDecision:
    """Map a ContentProfile to a RoutingDecision.

    Falls back to speech parameters if confidence is below threshold
    or if content_type is not in the routing table.

    Args:
        profile: acoustic content profile from ContentProfiler

    Returns:
        RoutingDecision with algorithm_id, parameters, and reason.
    """
    content_type = profile.content_type
    confidence = profile.confidence

    # Low-confidence fallback
    if confidence < _CONFIDENCE_THRESHOLD:
        content_type = _FALLBACK_CONTENT_TYPE
        reason = (
            f"low confidence ({confidence:.2f} < {_CONFIDENCE_THRESHOLD}) "
            f"for detected type '{profile.content_type}'; "
            f"falling back to speech parameters"
        )
    elif content_type not in _ROUTING_TABLE:
        content_type = _FALLBACK_CONTENT_TYPE
        reason = (
            f"unknown content type '{profile.content_type}'; "
            f"falling back to speech parameters"
        )
    else:
        entry = _ROUTING_TABLE[content_type]
        reason = (
            f"{content_type} detected (confidence={confidence:.2f}); "
            f"routing to {entry['target_subband']} band "
            f"level {entry['dwt_levels']}"
        )

    params = dict(_ROUTING_TABLE[content_type])
    algorithm_id = params.pop("algorithm_id")

    return RoutingDecision(
        algorithm_id=algorithm_id,
        parameters=params,
        content_type=content_type,
        confidence=confidence,
        reason=reason,
    )


def routing_decision_to_audio_params(rd: RoutingDecision) -> AudioEmbeddingParams:
    """Convert a RoutingDecision to concrete AudioEmbeddingParams.

    Args:
        rd: routing decision from route()

    Returns:
        AudioEmbeddingParams ready for use in the signing pipeline.
    """
    p = rd.parameters
    return AudioEmbeddingParams(
        dwt_levels=tuple(p["dwt_levels"]),
        chips_per_bit=p["chips_per_bit"],
        psychoacoustic=p["psychoacoustic"],
        safety_margin_db=p["safety_margin_db"],
        target_snr_db=p["target_snr_db"],
        target_subband=p["target_subband"],
        frame_length_ms=p["frame_length_ms"],
        pn_sequence_length=p["pn_sequence_length"],
    )
