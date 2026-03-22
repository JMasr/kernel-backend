"""Public verification endpoint — no authentication required.

POST /verify/public — verify media globally across all organisations.

Pepper handling: each org has a unique pepper_v1 used during signing to seed
the fingerprint projection matrix.  Since the caller does not know which org
signed the file, this endpoint tries every org pepper until it finds a
candidate match (or exhausts them all → CANDIDATE_NOT_FOUND).  Wrong peppers
fail fast in Phase A (fingerprint matching) without entering Phase B (WID
extraction), so the overhead is bounded by N × fingerprint-extraction time.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

logger = logging.getLogger("kernel.verification.public")

from kernel_backend.api.verification.router import _resolve_author_and_org, _to_response
from kernel_backend.config import Settings
from kernel_backend.core.domain.verification import RedReason
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService

router = APIRouter(prefix="/verify", tags=["public"])

_SYSTEM_PEPPER = Settings().system_pepper_bytes
_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


class PublicVerifyResponse(BaseModel):
    verdict: str
    content_id: str | None = None
    author_id: str | None = None
    author_name: str | None = None
    org_name: str | None = None
    red_reason: str | None = None
    wid_match: bool = False
    signature_valid: bool = False
    fingerprint_confidence: float = 0.0


async def _get_all_peppers(session_factory) -> list[bytes]:
    """Return all distinct org peppers + system fallback.

    Each organization has a unique pepper_v1 (hex string) that seeds the
    fingerprint projection matrix at sign time.  For public verification we
    must try every known pepper because the caller doesn't know the org.
    """
    peppers: list[bytes] = []
    try:
        from sqlalchemy import select

        from kernel_backend.infrastructure.database.models import OrgRecord

        async with session_factory() as session:
            result = await session.execute(select(OrgRecord.pepper_v1))
            for (pepper_hex,) in result.all():
                if pepper_hex:
                    peppers.append(bytes.fromhex(pepper_hex))
    except Exception:
        pass
    # Always include system default as fallback (content signed without org pepper)
    if _SYSTEM_PEPPER not in peppers:
        peppers.append(_SYSTEM_PEPPER)
    return peppers


@router.post("/public", response_model=PublicVerifyResponse, status_code=200)
async def verify_public(
    request: Request,
    file: UploadFile = File(..., description="Media file to verify (no auth required)"),
) -> PublicVerifyResponse:
    """Verify a media file globally across ALL organisations.

    Does not require authentication. Searches fingerprints across every org
    by trying each org's pepper until a candidate is found.
    Returns org_name if a signed match is found.
    """
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 GB)")

    suffix = Path(file.filename or "upload").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(content)

    try:
        media = MediaService()
        service = VerificationService()

        storage = request.app.state.storage
        registry = request.app.state.registry

        try:
            profile = media.probe(tmp_path)
        except ValueError as exc:
            logger.warning("verify/public: media probe failed for %s: %s", file.filename, exc)
            raise HTTPException(status_code=400, detail="Could not read media file — unsupported format or corrupted")

        if not profile.has_video and not profile.has_audio:
            raise HTTPException(status_code=400, detail="No audio or video stream detected")

        logger.debug("verify/public: file=%s, has_video=%s, has_audio=%s",
                      file.filename, profile.has_video, profile.has_audio)

        # Try every org pepper — wrong peppers fail fast at Phase A (fingerprint matching)
        all_peppers = await _get_all_peppers(request.app.state.db_session_factory)
        logger.debug("verify/public: trying %d peppers", len(all_peppers))

        result = None
        for idx, pepper in enumerate(all_peppers):
            if profile.has_video and profile.has_audio:
                attempt = await service.verify_av(
                    media_path=tmp_path, media=media, storage=storage,
                    registry=registry, pepper=pepper, org_id=None,
                )
            elif profile.has_video:
                attempt = await service.verify(
                    media_path=tmp_path, media=media, storage=storage,
                    registry=registry, pepper=pepper, org_id=None,
                )
            else:
                attempt = await service.verify_audio(
                    media_path=tmp_path, media=media, storage=storage,
                    registry=registry, pepper=pepper, org_id=None,
                )

            # Any result other than CANDIDATE_NOT_FOUND is authoritative
            if attempt.red_reason != RedReason.CANDIDATE_NOT_FOUND:
                logger.info("verify/public: found candidate with pepper #%d", idx + 1)
                result = attempt
                break

            result = attempt  # keep last CANDIDATE_NOT_FOUND as fallback

    finally:
        tmp_path.unlink(missing_ok=True)

    # Build base response from standard fields
    base = _to_response(result)

    # Look up human-readable names when we have a candidate
    author_name: str | None = None
    org_name: str | None = None
    if base.content_id or base.author_id:
        author_name, org_name = await _resolve_author_and_org(
            base.author_id,
            base.content_id,
            request.app.state.db_session_factory,
        )

    return PublicVerifyResponse(
        verdict=base.verdict,
        content_id=base.content_id,
        author_id=base.author_id,
        author_name=author_name,
        org_name=org_name,
        red_reason=base.red_reason,
        wid_match=base.wid_match,
        signature_valid=base.signature_valid,
        fingerprint_confidence=base.fingerprint_confidence,
    )
