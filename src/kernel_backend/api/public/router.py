"""Public verification endpoint — no authentication required.

POST /verify/public — verify media globally across all organisations.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.api.verification.router import _to_response
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService

router = APIRouter(prefix="/verify", tags=["public"])

_PEPPER = os.environ.get("WATERMARK_PEPPER", "kernel-default-pepper-32b!").encode()
_MAX_BYTES = 150 * 1024 * 1024  # 150 MB


class PublicVerifyResponse(BaseModel):
    verdict: str
    content_id: str | None = None
    author_id: str | None = None
    org_name: str | None = None
    red_reason: str | None = None
    wid_match: bool = False
    signature_valid: bool = False
    fingerprint_confidence: float = 0.0


async def _lookup_org_name(
    content_id: str,
    session_factory,
) -> str | None:
    """Return org name for the video identified by content_id, or None."""
    try:
        from kernel_backend.infrastructure.database.models import OrgRecord, Video

        async with session_factory() as session:
            result = await session.execute(
                select(OrgRecord.name)
                .join(Video, Video.org_id == OrgRecord.id)
                .where(Video.content_id == content_id)
            )
            return result.scalar_one_or_none()
    except Exception:
        return None


@router.post("/public", response_model=PublicVerifyResponse, status_code=200)
async def verify_public(
    request: Request,
    file: UploadFile = File(..., description="Media file to verify (no auth required)"),
) -> PublicVerifyResponse:
    """Verify a media file globally across ALL organisations.

    Does not require authentication. Searches fingerprints across every org.
    Returns org_name if a signed match is found.
    """
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 150 MB)")

    suffix = Path(file.filename or "upload").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(content)

    try:
        media = MediaService()
        service = VerificationService()

        storage = request.app.state.storage
        registry = request.app.state.registry

        profile = media.probe(tmp_path)

        # org_id=None → global search across all orgs
        if profile.has_video and profile.has_audio:
            result = await service.verify_av(
                media_path=tmp_path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=_PEPPER,
                org_id=None,
            )
        elif profile.has_video:
            result = await service.verify(
                media_path=tmp_path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=_PEPPER,
                org_id=None,
            )
        else:
            raise HTTPException(status_code=400, detail="No audio or video stream detected")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Build base response from standard fields
    base = _to_response(result)

    # Look up org name when we have a content_id
    org_name: str | None = None
    if base.content_id and result.verdict.value == "VERIFIED":
        org_name = await _lookup_org_name(
            base.content_id,
            request.app.state.db_session_factory,
        )

    return PublicVerifyResponse(
        verdict=base.verdict,
        content_id=base.content_id,
        author_id=base.author_id,
        org_name=org_name,
        red_reason=base.red_reason,
        wid_match=base.wid_match,
        signature_valid=base.signature_valid,
        fingerprint_confidence=base.fingerprint_confidence,
    )
