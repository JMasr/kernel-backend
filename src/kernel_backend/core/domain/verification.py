"""
Phase 4 — Verification domain model.

The central invariant:
  VERIFIED means extracted_WID == stored_WID AND Ed25519 signature valid.
  Nothing else can produce VERIFIED. No score, no threshold, no heuristic.

fingerprint_confidence exists ONLY for diagnostics. It must never appear in
any conditional that changes the verdict. A code review must reject any
`if fingerprint_confidence > threshold` that affects verdict logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    RED      = "RED"


class RedReason(str, Enum):
    CANDIDATE_NOT_FOUND = "candidate_not_found"
    # fingerprint lookup found no registry entry matching the submitted file

    WID_MISMATCH        = "wid_mismatch"
    # RS decode succeeded, but decoded WID != stored WID — primary tamper indicator

    SIGNATURE_INVALID   = "signature_invalid"
    # Ed25519 verify failed — manifest was altered after signing

    WID_UNDECODABLE     = "wid_undecodable"
    # RS decode failed completely (too many erasures) — quality signal, NOT tampering

    WATERMARK_DEGRADED  = "watermark_degraded"
    # partial watermark detected but WID could not be recovered
    # (signed but too damaged to verify) — quality signal, NOT tampering


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict

    # Populated when a candidate is found (all non-CANDIDATE_NOT_FOUND cases)
    content_id: str | None        = None
    author_id: str | None         = None
    author_public_key: str | None = None

    # Populated only when verdict == RED
    red_reason: RedReason | None  = None

    # Both True only when verdict == VERIFIED
    wid_match: bool               = False
    signature_valid: bool         = False

    # Watermark extraction statistics — always populated when Phase B runs
    n_segments_total: int         = 0
    n_segments_decoded: int       = 0   # segments where WID symbol was recoverable
    n_erasures: int               = 0   # segments marked as erasure by RS

    # DIAGNOSTIC ONLY — never drives the verdict
    fingerprint_confidence: float = 0.0
