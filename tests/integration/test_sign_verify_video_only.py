"""
tests/integration/test_sign_verify_video_only.py

Phase 4 integration tests — full sign → verify roundtrip on real polygon clips.
Uses SQLite in-memory registry (no Neon connection required).
Marked @pytest.mark.integration @pytest.mark.slow.

Prerequisite: polygon clips must be present in data/video/.
Tests skip automatically if clips are missing.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).parents[2] / "data"
SPEECH_01 = DATA_ROOT / "video" / "speech" / "speech.mp4"
CAMPING_01 = DATA_ROOT / "video" / "outdoor" / "camping_01.mp4"


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Polygon clip not found: {path}")


async def _build_infra(db_session):
    """Return (storage, registry) backed by SQLite in-memory session."""
    from kernel_backend.infrastructure.database.repositories import VideoRepository
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter
    import tempfile

    storage_dir = Path(tempfile.mkdtemp())
    storage = LocalStorageAdapter(base_path=storage_dir)
    registry = VideoRepository(session=db_session)
    return storage, registry


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_sign_verify_roundtrip_speech(db_session):
    """
    [BLOCKING] Full roundtrip on speech_01 (960×540, 25fps, 206s).
    Sign → store → verify → assert VERIFIED.
    Uses SQLite in-memory registry.
    """
    _skip_if_missing(SPEECH_01)

    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.domain.verification import Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService

    priv_pem, pub_pem = generate_keypair()
    cert = Certificate(
        author_id="integration-test-author",
        name="Integration Test",
        institution="Test Suite",
        public_key_pem=pub_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    pepper = b"integration-test-pepper-32bytes!"
    media = MediaService()
    storage, registry = await _build_infra(db_session)

    # Sign
    sign_result = await sign_video(
        media_path=SPEECH_01,
        certificate=cert,
        private_key_pem=priv_pem,
        storage=storage,
        registry=registry,
        pepper=pepper,
        media=media,
    )

    assert sign_result.content_id

    # Get the signed output path from storage
    signed_bytes = await storage.get(sign_result.signed_media_key)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        signed_path = Path(tmp.name)
        signed_path.write_bytes(signed_bytes)

    try:
        service = VerificationService()
        result = await service.verify(
            media_path=signed_path,
            media=media,
            storage=storage,
            registry=registry,
            pepper=pepper,
        )
    finally:
        signed_path.unlink(missing_ok=True)

    assert result.verdict == Verdict.VERIFIED, (
        f"Expected VERIFIED, got {result.verdict} / {result.red_reason}"
    )
    assert result.content_id == sign_result.content_id
    assert result.n_segments_decoded > 0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_sign_verify_roundtrip_h264_crf23(db_session, tmp_path):
    """
    [BLOCKING] Sign → H.264 CRF 23 recompress → verify.
    Simulates the most common real-world distribution scenario.
    Assert verdict == VERIFIED.
    """
    _skip_if_missing(SPEECH_01)

    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.domain.verification import Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService

    priv_pem, pub_pem = generate_keypair()
    cert = Certificate(
        author_id="integration-crf23-author",
        name="CRF23 Test",
        institution="Test Suite",
        public_key_pem=pub_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    pepper = b"integration-test-pepper-32bytes!"
    media = MediaService()
    storage, registry = await _build_infra(db_session)

    sign_result = await sign_video(
        media_path=SPEECH_01,
        certificate=cert,
        private_key_pem=priv_pem,
        storage=storage,
        registry=registry,
        pepper=pepper,
        media=media,
    )

    signed_bytes = await storage.get(sign_result.signed_media_key)
    signed_path = tmp_path / "signed.mp4"
    signed_path.write_bytes(signed_bytes)

    # Recompress at CRF 23
    recomp_path = tmp_path / "recomp_crf23.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(signed_path),
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-loglevel", "quiet", str(recomp_path)],
        check=True,
    )

    service = VerificationService()
    result = await service.verify(
        media_path=recomp_path,
        media=media,
        storage=storage,
        registry=registry,
        pepper=pepper,
    )

    assert result.verdict == Verdict.VERIFIED, (
        f"Expected VERIFIED after CRF 23, got {result.verdict} / {result.red_reason}"
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_sign_verify_roundtrip_h264_crf28(db_session, tmp_path):
    """[BLOCKING] Sign → H.264 CRF 28 recompress → verify → VERIFIED."""
    _skip_if_missing(SPEECH_01)

    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.domain.verification import Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService

    priv_pem, pub_pem = generate_keypair()
    cert = Certificate(
        author_id="integration-crf28-author",
        name="CRF28 Test",
        institution="Test Suite",
        public_key_pem=pub_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    pepper = b"integration-test-pepper-32bytes!"
    media = MediaService()
    storage, registry = await _build_infra(db_session)

    sign_result = await sign_video(
        media_path=SPEECH_01,
        certificate=cert,
        private_key_pem=priv_pem,
        storage=storage,
        registry=registry,
        pepper=pepper,
        media=media,
    )

    signed_bytes = await storage.get(sign_result.signed_media_key)
    signed_path = tmp_path / "signed.mp4"
    signed_path.write_bytes(signed_bytes)

    recomp_path = tmp_path / "recomp_crf28.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(signed_path),
         "-vcodec", "libx264", "-crf", "28", "-preset", "fast",
         "-loglevel", "quiet", str(recomp_path)],
        check=True,
    )

    service = VerificationService()
    result = await service.verify(
        media_path=recomp_path,
        media=media,
        storage=storage,
        registry=registry,
        pepper=pepper,
    )

    assert result.verdict == Verdict.VERIFIED, (
        f"Expected VERIFIED after CRF 28, got {result.verdict} / {result.red_reason}"
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="CRF 35 may exceed RS erasure tolerance")
@pytest.mark.asyncio
async def test_sign_verify_roundtrip_h264_crf35(db_session, tmp_path):
    """
    [INFORMATIONAL] CRF 35 — acceptable outcomes are VERIFIED or RED(WID_UNDECODABLE).
    RED(WID_MISMATCH) is never acceptable — that would be a tampering false-positive.
    """
    _skip_if_missing(SPEECH_01)

    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.domain.verification import RedReason, Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService

    priv_pem, pub_pem = generate_keypair()
    cert = Certificate(
        author_id="integration-crf35-author",
        name="CRF35 Test",
        institution="Test Suite",
        public_key_pem=pub_pem,
        created_at="2026-01-01T00:00:00+00:00",
    )
    pepper = b"integration-test-pepper-32bytes!"
    media = MediaService()
    storage, registry = await _build_infra(db_session)

    sign_result = await sign_video(
        media_path=SPEECH_01,
        certificate=cert,
        private_key_pem=priv_pem,
        storage=storage,
        registry=registry,
        pepper=pepper,
        media=media,
    )

    signed_bytes = await storage.get(sign_result.signed_media_key)
    signed_path = tmp_path / "signed.mp4"
    signed_path.write_bytes(signed_bytes)

    recomp_path = tmp_path / "recomp_crf35.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(signed_path),
         "-vcodec", "libx264", "-crf", "35", "-preset", "fast",
         "-loglevel", "quiet", str(recomp_path)],
        check=True,
    )

    service = VerificationService()
    result = await service.verify(
        media_path=recomp_path,
        media=media,
        storage=storage,
        registry=registry,
        pepper=pepper,
    )

    # WID_MISMATCH would be a false tamper accusation — must never happen
    assert result.red_reason != RedReason.WID_MISMATCH, (
        "CRF 35 produced WID_MISMATCH — this is a false tamper accusation, not a quality issue"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unsigned_video_returns_red_candidate_not_found(db_session):
    """
    [BLOCKING — security property] Unsigned content must return RED.
    Uses camping_01 (unsigned) against an empty registry.
    """
    _skip_if_missing(CAMPING_01)

    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.verification import RedReason, Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService

    media = MediaService()
    storage, registry = await _build_infra(db_session)
    # Registry is empty — no candidates can match

    service = VerificationService()
    result = await service.verify(
        media_path=CAMPING_01,
        media=media,
        storage=storage,
        registry=registry,
        pepper=b"integration-test-pepper-32bytes!",
    )

    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.CANDIDATE_NOT_FOUND


@pytest.mark.integration
@pytest.mark.asyncio
async def test_different_content_id_returns_red_wid_mismatch(db_session, tmp_path):
    """
    [BLOCKING — security property] Transplant attack: sign content A, verify as
    if it is content B. Fingerprints will match but WID will not.
    Assert verdict == RED, red_reason == WID_MISMATCH.
    """
    _skip_if_missing(SPEECH_01)

    from kernel_backend.core.services.crypto_service import generate_keypair
    from kernel_backend.core.services.signing_service import sign_video
    from kernel_backend.core.services.verification_service import VerificationService
    from kernel_backend.core.domain.identity import Certificate
    from kernel_backend.core.domain.verification import RedReason, Verdict
    from kernel_backend.infrastructure.media.media_service import MediaService
    from kernel_backend.core.domain.watermark import VideoEntry
    from kernel_backend.core.services.crypto_service import derive_wid, sign_manifest
    from kernel_backend.core.domain.manifest import CryptographicManifest
    from datetime import datetime, timezone

    pepper = b"integration-test-pepper-32bytes!"
    media = MediaService()
    storage, registry = await _build_infra(db_session)

    # Sign content A
    priv_a, pub_a = generate_keypair()
    cert_a = Certificate(
        author_id="transplant-test-author-A",
        name="Author A",
        institution="Test",
        public_key_pem=pub_a,
        created_at="2026-01-01T00:00:00+00:00",
    )
    result_a = await sign_video(
        media_path=SPEECH_01,
        certificate=cert_a,
        private_key_pem=priv_a,
        storage=storage,
        registry=registry,
        pepper=pepper,
        media=media,
    )

    # Retrieve signed content A
    signed_bytes = await storage.get(result_a.signed_media_key)
    signed_path = tmp_path / "signed_a.mp4"
    signed_path.write_bytes(signed_bytes)

    # Verify: the registry has content A's entry with WID_A.
    # The fingerprints will match (same visual content), but the registered WID
    # is WID_A, while the file is correctly watermarked with WID_A — so this
    # test validates the round-trip is correct, then we perturb the stored WID.

    # Manually inject a DIFFERENT WID into the stored entry to simulate transplant
    entry_a = await registry.get_by_content_id(result_a.content_id)
    assert entry_a is not None

    # Create a fake manifest with a different signature (different WID)
    priv_b, pub_b = generate_keypair()
    fake_manifest = CryptographicManifest(
        content_id=result_a.content_id,
        content_hash_sha256="d" * 64,
        fingerprints_audio=[],
        fingerprints_video=[],
        author_id="faked",
        author_public_key=pub_b,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    fake_sig = sign_manifest(fake_manifest, priv_b)

    # Overwrite the stored entry with the fake signature (different WID)
    from kernel_backend.core.domain.watermark import VideoEntry as VE
    tampered_entry = VE(
        content_id=entry_a.content_id,
        author_id=entry_a.author_id,
        author_public_key=pub_b,  # different key
        active_signals=entry_a.active_signals,
        rs_n=entry_a.rs_n,
        pilot_hash_48=entry_a.pilot_hash_48,
        manifest_signature=fake_sig,  # different signature → different WID
    )
    await registry.save_video(tampered_entry)

    service = VerificationService()
    result = await service.verify(
        media_path=signed_path,
        media=media,
        storage=storage,
        registry=registry,
        pepper=pepper,
    )

    # The file contains WID_A but registry now has WID_B → WID mismatch
    assert result.verdict == Verdict.RED
    assert result.red_reason == RedReason.WID_MISMATCH
