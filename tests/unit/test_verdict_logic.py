"""
Pure-logic tests for the verdict decision table (_compose_verdict).

No DSP, no media, no mocking. Tests the combinatorial logic that maps
(audio_wid_ok, video_wid_ok, decodable, signature_ok) → (Verdict, RedReason).

Extracted from test_verify_av_service.py during pipeline testing refactor.
"""
from __future__ import annotations

import pytest

from kernel_backend.core.domain.verification import RedReason, Verdict
from kernel_backend.core.services.verification_service import _compose_verdict


@pytest.mark.parametrize(
    "audio_wid_ok,video_wid_ok,sig_ok,audio_dec,video_dec,expected_verdict,expected_reason",
    [
        # Both undecodable
        (False, False, True, False, False, Verdict.RED, RedReason.WID_UNDECODABLE),
        # Only audio undecodable
        (False, True, True, False, True, Verdict.RED, RedReason.AUDIO_WID_UNDECODABLE),
        # Only video undecodable
        (True, False, True, True, False, Verdict.RED, RedReason.VIDEO_WID_UNDECODABLE),
        # Both decodable, both mismatch
        (False, False, True, True, True, Verdict.RED, RedReason.WID_MISMATCH),
        # Audio mismatch only
        (False, True, True, True, True, Verdict.RED, RedReason.AUDIO_WID_MISMATCH),
        # Video mismatch only
        (True, False, True, True, True, Verdict.RED, RedReason.VIDEO_WID_MISMATCH),
        # Both match, signature invalid
        (True, True, False, True, True, Verdict.RED, RedReason.SIGNATURE_INVALID),
        # All pass — VERIFIED
        (True, True, True, True, True, Verdict.VERIFIED, None),
    ],
)
def test_av_verdict_table_exhaustive(
    audio_wid_ok,
    video_wid_ok,
    sig_ok,
    audio_dec,
    video_dec,
    expected_verdict,
    expected_reason,
) -> None:
    """
    Machine-readable decision table for _compose_verdict.
    Every combination must map to the expected (verdict, red_reason).
    Any failure here is a logic error in the verdict composition.
    """
    verdict, reason = _compose_verdict(
        audio_wid_ok=audio_wid_ok,
        video_wid_ok=video_wid_ok,
        audio_decodable=audio_dec,
        video_decodable=video_dec,
        signature_ok=sig_ok,
    )
    assert verdict == expected_verdict
    assert reason == expected_reason


def test_wid_mismatch_precedes_signature_invalid() -> None:
    """WID_MISMATCH is more specific than SIGNATURE_INVALID — must take precedence."""
    verdict, reason = _compose_verdict(
        audio_wid_ok=False,
        video_wid_ok=True,
        audio_decodable=True,
        video_decodable=True,
        signature_ok=False,
    )
    assert reason == RedReason.AUDIO_WID_MISMATCH
