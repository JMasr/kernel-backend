"""
Phase 4 — Verification API schemas.

Design note: there is deliberately no top-level `confidence` or `score` field.
Including such a field would suggest to API consumers that the verdict is
score-based. `fingerprint_confidence` is a sub-field, clearly labelled diagnostic.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class VerificationResponse(BaseModel):
    verdict: Literal["VERIFIED", "RED"]
    content_id: str | None                        = None
    author_id: str | None                         = None
    red_reason: str | None                        = None   # RedReason value, or None on VERIFIED
    wid_match: bool                               = False
    signature_valid: bool                         = False
    n_segments_total: int                         = 0
    n_segments_decoded: int                       = 0
    n_erasures: int                               = 0
    fingerprint_confidence: float                 = 0.0    # diagnostic only, never drives verdict

    # AV-only fields — None for video-only or audio-only containers
    audio_verdict: Literal["VERIFIED", "RED"] | None = None
    video_verdict: Literal["VERIFIED", "RED"] | None = None
