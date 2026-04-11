"""
Domain models for video content-adaptive watermark routing.

VideoContentProfile captures deterministic visual descriptors of video content.
VideoRoutingDecision maps a video profile to optimized VideoEmbeddingParams.
Both are frozen dataclasses -- immutable at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


VideoContentType = Literal["normal", "dark", "bright", "high_motion", "static"]


@dataclass(frozen=True)
class VideoContentProfile:
    """Immutable visual content descriptor produced by video ContentProfiler."""
    content_type: VideoContentType
    confidence: float
    features: dict[str, float]
    descriptor_version: str  # e.g. "1.0.0"
    code_hash: str           # "sha256:<hex prefix>"


@dataclass(frozen=True)
class VideoRoutingDecision:
    """Immutable routing decision mapping video profile to watermark params."""
    algorithm_id: str        # e.g. "dct_qim_v1"
    parameters: dict         # overrides for VideoEmbeddingParams fields
    content_type: str
    confidence: float
    reason: str              # human-readable explanation


def video_content_profile_to_dict(profile: VideoContentProfile) -> dict:
    """Serialize VideoContentProfile to a plain dict."""
    return asdict(profile)


def video_routing_decision_to_dict(rd: VideoRoutingDecision) -> dict:
    """Serialize VideoRoutingDecision to a plain dict for JSON storage."""
    return asdict(rd)


def video_routing_decision_from_dict(d: dict) -> VideoRoutingDecision:
    """Deserialize VideoRoutingDecision from a dict (read from JSONB)."""
    return VideoRoutingDecision(
        algorithm_id=d["algorithm_id"],
        parameters=d["parameters"],
        content_type=d["content_type"],
        confidence=d["confidence"],
        reason=d.get("reason", ""),
    )
