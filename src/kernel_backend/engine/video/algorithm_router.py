"""
Content-adaptive algorithm router for video watermarking.

Maps a VideoContentProfile to optimized DCT-QIM parameters based on a
deterministic routing table. No file I/O, no ML dependencies.

The routing table is a Python dict constant -- fully auditable, versionable,
and free of file I/O in the engine layer.
"""
from __future__ import annotations

from kernel_backend.core.domain.video_content_profile import (
    VideoContentProfile,
    VideoRoutingDecision,
)
from kernel_backend.core.domain.watermark import VideoEmbeddingParams

# -- Routing table -----------------------------------------------------------
# Maps content_type -> optimized DCT-QIM (Chou-Li JND) parameters.
#
# Key parameter changes vs normal baseline:
#   - Dark: higher qim_step_base (80 vs 64), tighter max (96 vs 128).
#     Prevents JND model from creating oversized bins on low-luminance blocks
#     that collapse under H.264 quantization.
#   - Bright: smaller steps -- DCT coefficients have more energy, tolerate less.
#   - High motion: motion masking allows slightly larger base step.
#   - Static: artifacts more visible on static content, use smaller steps.

_ROUTING_TABLE: dict[str, dict] = {
    "normal": {
        "algorithm_id": "dct_qim_v1",
        "jnd_adaptive": True,
        "qim_step_base": 64.0,
        "qim_step_min": 44.0,
        "qim_step_max": 128.0,
        "qim_quantize_to": 4.0,
    },
    "dark": {
        "algorithm_id": "dct_qim_v1",
        "jnd_adaptive": True,
        "qim_step_base": 80.0,
        "qim_step_min": 64.0,
        "qim_step_max": 96.0,
        "qim_quantize_to": 4.0,
        "min_block_variance": 100.0,
        "block_oversample": 16,
    },
    "bright": {
        "algorithm_id": "dct_qim_v1",
        "jnd_adaptive": True,
        "qim_step_base": 56.0,
        "qim_step_min": 40.0,
        "qim_step_max": 96.0,
        "qim_quantize_to": 4.0,
    },
    "high_motion": {
        "algorithm_id": "dct_qim_v1",
        "jnd_adaptive": True,
        "qim_step_base": 72.0,
        "qim_step_min": 48.0,
        "qim_step_max": 128.0,
        "qim_quantize_to": 4.0,
    },
    "static": {
        "algorithm_id": "dct_qim_v1",
        "jnd_adaptive": True,
        "qim_step_base": 56.0,
        "qim_step_min": 40.0,
        "qim_step_max": 100.0,
        "qim_quantize_to": 4.0,
    },
}

_FALLBACK_CONTENT_TYPE = "normal"
_CONFIDENCE_THRESHOLD = 0.7


def route(profile: VideoContentProfile) -> VideoRoutingDecision:
    """Map a VideoContentProfile to a VideoRoutingDecision.

    Falls back to normal parameters if confidence is below threshold
    or if content_type is not in the routing table.

    Args:
        profile: visual content profile from video ContentProfiler

    Returns:
        VideoRoutingDecision with algorithm_id, parameters, and reason.
    """
    content_type = profile.content_type
    confidence = profile.confidence

    # Low-confidence fallback
    if confidence < _CONFIDENCE_THRESHOLD:
        content_type = _FALLBACK_CONTENT_TYPE
        reason = (
            f"low confidence ({confidence:.2f} < {_CONFIDENCE_THRESHOLD}) "
            f"for detected type '{profile.content_type}'; "
            f"falling back to normal parameters"
        )
    elif content_type not in _ROUTING_TABLE:
        content_type = _FALLBACK_CONTENT_TYPE
        reason = (
            f"unknown content type '{profile.content_type}'; "
            f"falling back to normal parameters"
        )
    else:
        entry = _ROUTING_TABLE[content_type]
        reason = (
            f"{content_type} detected (confidence={confidence:.2f}); "
            f"routing to qim_step_base={entry['qim_step_base']}, "
            f"range=[{entry['qim_step_min']}, {entry['qim_step_max']}]"
        )

    params = dict(_ROUTING_TABLE[content_type])
    algorithm_id = params.pop("algorithm_id")

    return VideoRoutingDecision(
        algorithm_id=algorithm_id,
        parameters=params,
        content_type=content_type,
        confidence=confidence,
        reason=reason,
    )


def video_routing_decision_to_video_params(
    rd: VideoRoutingDecision,
) -> VideoEmbeddingParams:
    """Convert a VideoRoutingDecision to concrete VideoEmbeddingParams.

    Args:
        rd: routing decision from route()

    Returns:
        VideoEmbeddingParams ready for use in the signing pipeline.
    """
    p = rd.parameters
    return VideoEmbeddingParams(
        jnd_adaptive=p["jnd_adaptive"],
        qim_step_base=p["qim_step_base"],
        qim_step_min=p["qim_step_min"],
        qim_step_max=p["qim_step_max"],
        qim_quantize_to=p["qim_quantize_to"],
        min_block_variance=p.get("min_block_variance", 0.0),
        block_oversample=p.get("block_oversample", 1),
    )
