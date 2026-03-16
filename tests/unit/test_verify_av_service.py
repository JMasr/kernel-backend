"""
Unit tests for verify_av() — Phase 5 AV verification pipeline.

All tests use mocks / patching. No real media files are required.
The explicit decision table (_compose_verdict) is tested exhaustively via
test_av_verdict_table_exhaustive.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import json
import pytest

from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.verification import AVVerificationResult, RedReason, Verdict
from kernel_backend.core.domain.watermark import VideoEntry
from kernel_backend.core.services.crypto_service import derive_wid, generate_keypair, sign_manifest
from kernel_backend.core.services.verification_service import VerificationService, _compose_verdict


# ── Shared helpers ──────────────────────────────────────────────────────────────

CONTENT_ID = "test-content-id-phase5-av"
AUTHOR_ID  = "test-author-id"
PEPPER     = b"unit-test-pepper-bytes-padded32!"

_private_pem, _public_pem = generate_keypair()


def _manifest() -> CryptographicManifest:
    return CryptographicManifest(
        content_id=CONTENT_ID,
        content_hash_sha256="a" * 64,
        fingerprints_audio=["0102030405060708"],
        fingerprints_video=["0102030405060708"],
        author_id=AUTHOR_ID,
        author_public_key=_public_pem,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _signed_manifest():
    m = _manifest()
    sig = sign_manifest(m, _private_pem)
    wid = derive_wid(sig, CONTENT_ID)
    return m, sig, wid


def _manifest_json_str(m: CryptographicManifest) -> str:
    return json.dumps({
        "content_id": m.content_id,
        "content_hash_sha256": m.content_hash_sha256,
        "fingerprints_audio": m.fingerprints_audio,
        "fingerprints_video": m.fingerprints_video,
        "author_id": m.author_id,
        "author_public_key": m.author_public_key,
        "created_at": m.created_at,
        "schema_version": m.schema_version,
    })


def _make_entry(
    rs_n: int = 20,
    sig: bytes | None = None,
    m: CryptographicManifest | None = None,
) -> VideoEntry:
    if sig is None or m is None:
        m, sig, _ = _signed_manifest()
    return VideoEntry(
        content_id=CONTENT_ID,
        author_id=AUTHOR_ID,
        author_public_key=_public_pem,
        active_signals=[
            "audio_pilot", "audio_wid", "audio_fingerprint",
            "video_pilot", "video_wid", "video_fingerprint",
        ],
        rs_n=rs_n,
        pilot_hash_48=0,
        manifest_signature=sig,
        manifest_json=_manifest_json_str(m),
    )


def _patch_identify(content_id=CONTENT_ID, pubkey=None, confidence=1.0):
    """Patch _identify_candidate to return a fixed candidate."""
    return patch.object(
        VerificationService,
        "_identify_candidate",
        new=AsyncMock(return_value=(content_id, pubkey or _public_pem, confidence)),
    )


def _patch_audio_wid(wid: bytes | None, decodable: bool = True,
                     n_seg: int = 10, n_dec: int = 10, n_era: int = 0):
    """Patch _extract_audio_wid to return a fixed result."""
    return patch.object(
        VerificationService,
        "_extract_audio_wid",
        return_value=(wid, decodable, n_seg, n_dec, n_era),
    )


def _patch_video_wid(wid: bytes | None, decodable: bool = True,
                     n_seg: int = 10, n_dec: int = 10, n_era: int = 0):
    """Patch _extract_video_wid to return a fixed result."""
    return patch.object(
        VerificationService,
        "_extract_video_wid",
        return_value=(wid, decodable, n_seg, n_dec, n_era),
    )


# ── Tests ───────────────────────────────────────────────────────────────────────

async def test_av_verified_when_both_channels_match() -> None:
    """VERIFIED when audio_wid matches, video_wid matches, and signature is valid."""
    m, sig, wid = _signed_manifest()
    entry = _make_entry(sig=sig, m=m)
    stored_wid_bytes = wid.data

    media = MagicMock()
    storage = MagicMock()
    registry = AsyncMock()
    registry.get_by_content_id.return_value = entry

    with (
        _patch_identify(),
        _patch_audio_wid(stored_wid_bytes),
        _patch_video_wid(stored_wid_bytes),
        patch(
            "kernel_backend.core.services.verification_service._manifest_from_json",
            return_value=m,
        ),
    ):
        result: AVVerificationResult = await VerificationService().verify_av(
            media_path=Path("fake.mp4"),
            media=media,
            storage=storage,
            registry=registry,
            pepper=PEPPER,
        )

    assert result.verdict == Verdict.VERIFIED
    assert result.audio_verdict == Verdict.VERIFIED
    assert result.video_verdict == Verdict.VERIFIED
    assert result.wid_match is True
    assert result.signature_valid is True
    assert result.red_reason is None


async def test_av_red_audio_wid_mismatch() -> None:
    """RED(AUDIO_WID_MISMATCH) when audio WID differs; video_verdict stays VERIFIED."""
    m, sig, wid = _signed_manifest()
    entry = _make_entry(sig=sig, m=m)
    stored_wid_bytes = wid.data
    wrong_wid = bytes(b ^ 0xFF for b in stored_wid_bytes)

    media, storage, registry = MagicMock(), MagicMock(), AsyncMock()
    registry.get_by_content_id.return_value = entry

    with (
        _patch_identify(),
        _patch_audio_wid(wrong_wid),          # audio WID is wrong
        _patch_video_wid(stored_wid_bytes),    # video WID is correct
        patch(
            "kernel_backend.core.services.verification_service._manifest_from_json",
            return_value=m,
        ),
    ):
        result = await VerificationService().verify_av(
            media_path=Path("fake.mp4"),
            media=media, storage=storage, registry=registry, pepper=PEPPER,
        )

    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.AUDIO_WID_MISMATCH
    assert result.audio_verdict == Verdict.RED
    assert result.video_verdict == Verdict.VERIFIED   # channel-level resolution preserved


async def test_av_red_video_wid_mismatch() -> None:
    """RED(VIDEO_WID_MISMATCH) when video WID differs; audio_verdict stays VERIFIED."""
    m, sig, wid = _signed_manifest()
    entry = _make_entry(sig=sig, m=m)
    stored_wid_bytes = wid.data
    wrong_wid = bytes(b ^ 0xFF for b in stored_wid_bytes)

    media, storage, registry = MagicMock(), MagicMock(), AsyncMock()
    registry.get_by_content_id.return_value = entry

    with (
        _patch_identify(),
        _patch_audio_wid(stored_wid_bytes),
        _patch_video_wid(wrong_wid),
        patch(
            "kernel_backend.core.services.verification_service._manifest_from_json",
            return_value=m,
        ),
    ):
        result = await VerificationService().verify_av(
            media_path=Path("fake.mp4"),
            media=media, storage=storage, registry=registry, pepper=PEPPER,
        )

    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.VIDEO_WID_MISMATCH
    assert result.audio_verdict == Verdict.VERIFIED
    assert result.video_verdict == Verdict.RED


async def test_av_red_video_undecodable_audio_verified() -> None:
    """
    RED(VIDEO_WID_UNDECODABLE) when video RS decode fails; audio remains verified.
    This is the expected result for content redistributed at CRF 35.
    """
    m, sig, wid = _signed_manifest()
    entry = _make_entry(sig=sig, m=m)
    stored_wid_bytes = wid.data

    media, storage, registry = MagicMock(), MagicMock(), AsyncMock()
    registry.get_by_content_id.return_value = entry

    with (
        _patch_identify(),
        _patch_audio_wid(stored_wid_bytes, decodable=True),
        _patch_video_wid(None, decodable=False),  # RS decode failed
        patch(
            "kernel_backend.core.services.verification_service._manifest_from_json",
            return_value=m,
        ),
    ):
        result = await VerificationService().verify_av(
            media_path=Path("fake.mp4"),
            media=media, storage=storage, registry=registry, pepper=PEPPER,
        )

    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.VIDEO_WID_UNDECODABLE
    assert result.audio_verdict == Verdict.VERIFIED
    assert result.video_verdict == Verdict.RED


@pytest.mark.parametrize("audio_wid_ok,video_wid_ok,sig_ok,audio_dec,video_dec,expected_verdict,expected_reason", [
    # Both undecodable
    (False, False, True,  False, False, Verdict.RED, RedReason.WID_UNDECODABLE),
    # Only audio undecodable
    (False, True,  True,  False, True,  Verdict.RED, RedReason.AUDIO_WID_UNDECODABLE),
    # Only video undecodable
    (True,  False, True,  True,  False, Verdict.RED, RedReason.VIDEO_WID_UNDECODABLE),
    # Both decodable, both mismatch
    (False, False, True,  True,  True,  Verdict.RED, RedReason.WID_MISMATCH),
    # Audio mismatch only
    (False, True,  True,  True,  True,  Verdict.RED, RedReason.AUDIO_WID_MISMATCH),
    # Video mismatch only
    (True,  False, True,  True,  True,  Verdict.RED, RedReason.VIDEO_WID_MISMATCH),
    # Both match, signature invalid
    (True,  True,  False, True,  True,  Verdict.RED, RedReason.SIGNATURE_INVALID),
    # All pass — VERIFIED
    (True,  True,  True,  True,  True,  Verdict.VERIFIED, None),
])
def test_av_verdict_table_exhaustive(
    audio_wid_ok, video_wid_ok, sig_ok, audio_dec, video_dec,
    expected_verdict, expected_reason,
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


async def test_fingerprint_confidence_never_produces_verified() -> None:
    """High fingerprint confidence combined with WID mismatch must still return RED."""
    m, sig, wid = _signed_manifest()
    entry = _make_entry(sig=sig, m=m)
    stored_wid_bytes = wid.data
    wrong_wid = bytes(b ^ 0xFF for b in stored_wid_bytes)

    media, storage, registry = MagicMock(), MagicMock(), AsyncMock()
    registry.get_by_content_id.return_value = entry

    with (
        patch.object(
            VerificationService,
            "_identify_candidate",
            new=AsyncMock(return_value=(CONTENT_ID, _public_pem, 0.99)),  # very high confidence
        ),
        _patch_audio_wid(wrong_wid),
        _patch_video_wid(wrong_wid),
        patch(
            "kernel_backend.core.services.verification_service._manifest_from_json",
            return_value=m,
        ),
    ):
        result = await VerificationService().verify_av(
            media_path=Path("fake.mp4"),
            media=media, storage=storage, registry=registry, pepper=PEPPER,
        )

    assert result.verdict == Verdict.RED
    assert result.fingerprint_confidence == pytest.approx(0.99)


async def test_wid_mismatch_precedes_signature_invalid() -> None:
    """When audio WID mismatches AND signature is invalid, red_reason == AUDIO_WID_MISMATCH."""
    m, sig, wid = _signed_manifest()
    entry = _make_entry(sig=sig, m=m)
    stored_wid_bytes = wid.data
    wrong_wid = bytes(b ^ 0xFF for b in stored_wid_bytes)

    media, storage, registry = MagicMock(), MagicMock(), AsyncMock()
    registry.get_by_content_id.return_value = entry

    with (
        _patch_identify(),
        _patch_audio_wid(wrong_wid),           # WID mismatch
        _patch_video_wid(stored_wid_bytes),
        patch(
            "kernel_backend.core.services.verification_service.verify_manifest",
            return_value=False,                # signature also invalid
        ),
        patch(
            "kernel_backend.core.services.verification_service._manifest_from_json",
            return_value=m,
        ),
    ):
        result = await VerificationService().verify_av(
            media_path=Path("fake.mp4"),
            media=media, storage=storage, registry=registry, pepper=PEPPER,
        )

    # WID mismatch is more specific — must take precedence over signature invalid
    assert result.red_reason == RedReason.AUDIO_WID_MISMATCH
