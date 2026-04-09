"""
Domain models for content-adaptive watermark routing.

ContentProfile captures deterministic acoustic descriptors of audio content.
RoutingDecision maps a content profile to optimized watermarking parameters.
Both are frozen dataclasses — immutable at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal


ContentType = Literal["speech", "music", "classical", "ambient", "silence"]


@dataclass(frozen=True)
class ContentProfile:
    """Immutable acoustic content descriptor produced by ContentProfiler."""
    content_type: ContentType
    confidence: float
    features: dict[str, float]
    descriptor_version: str  # e.g. "1.0.0"
    code_hash: str           # "sha256:<hex prefix>"


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable routing decision mapping content profile to watermark params."""
    algorithm_id: str        # e.g. "dwt_dsss_v2"
    parameters: dict         # overrides for AudioEmbeddingParams fields
    content_type: str
    confidence: float
    reason: str              # human-readable explanation


def content_profile_to_dict(profile: ContentProfile) -> dict:
    """Serialize ContentProfile to a plain dict."""
    return asdict(profile)


def routing_decision_to_dict(rd: RoutingDecision) -> dict:
    """Serialize RoutingDecision to a plain dict for JSON storage."""
    return asdict(rd)


def routing_decision_from_dict(d: dict) -> RoutingDecision:
    """Deserialize RoutingDecision from a dict (read from JSONB)."""
    return RoutingDecision(
        algorithm_id=d["algorithm_id"],
        parameters=d["parameters"],
        content_type=d["content_type"],
        confidence=d["confidence"],
        reason=d.get("reason", ""),
    )
