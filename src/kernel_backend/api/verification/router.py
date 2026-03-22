"""Phase 4 + Phase 5 — POST /verify router."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from kernel_backend.api.verification.schemas import VerificationResponse
from kernel_backend.config import Settings
from kernel_backend.core.domain.verification import AVVerificationResult, VerificationResult
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService

logger = logging.getLogger("kernel.verification")

router = APIRouter(tags=["verification"])

_settings = Settings()
_FALLBACK_PEPPER = _settings.system_pepper_bytes


async def _resolve_pepper(org_id, session_factory) -> bytes:
    """Return org.pepper_v1 bytes if set, else fall back to system pepper."""
    if org_id is None or session_factory is None:
        return _FALLBACK_PEPPER
    try:
        from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository
        async with session_factory() as session:
            repo = OrganizationRepository(session)
            org = await repo.get_organization_by_id(org_id)
            if org and org.pepper_v1:
                return bytes.fromhex(org.pepper_v1)
    except Exception:
        pass
    return _FALLBACK_PEPPER
_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


async def _resolve_author_and_org(
    author_id: str | None,
    content_id: str | None,
    session_factory,
) -> tuple[str | None, str | None]:
    """Return (author_name, org_name) from Identity + OrgRecord tables."""
    if session_factory is None or (author_id is None and content_id is None):
        return None, None
    try:
        from kernel_backend.infrastructure.database.models import Identity, OrgRecord, Video
        from sqlalchemy import select

        async with session_factory() as session:
            author_name: str | None = None
            org_name: str | None = None

            if author_id:
                row = await session.execute(
                    select(Identity.name).where(Identity.author_id == author_id)
                )
                author_name = row.scalar_one_or_none()

            if content_id:
                row = await session.execute(
                    select(OrgRecord.name)
                    .join(Video, Video.org_id == OrgRecord.id)
                    .where(Video.content_id == content_id)
                )
                org_name = row.scalar_one_or_none()

            return author_name, org_name
    except Exception:
        return None, None


def _to_response(result: VerificationResult | AVVerificationResult) -> VerificationResponse:
    """Convert a domain result (single-channel or AV) to the API response schema."""
    if isinstance(result, AVVerificationResult):
        return VerificationResponse(
            verdict=result.verdict.value,
            content_id=result.content_id,
            author_id=result.author_id,
            red_reason=result.red_reason.value if result.red_reason else None,
            wid_match=result.wid_match,
            signature_valid=result.signature_valid,
            n_segments_total=result.audio_n_segments + result.video_n_segments,
            n_segments_decoded=result.audio_n_decoded + result.video_n_decoded,
            n_erasures=result.audio_n_erasures + result.video_n_erasures,
            fingerprint_confidence=result.fingerprint_confidence,
            audio_verdict=result.audio_verdict.value,
            video_verdict=result.video_verdict.value,
        )
    return VerificationResponse(
        verdict=result.verdict.value,
        content_id=result.content_id,
        author_id=result.author_id,
        red_reason=result.red_reason.value if result.red_reason else None,
        wid_match=result.wid_match,
        signature_valid=result.signature_valid,
        n_segments_total=result.n_segments_total,
        n_segments_decoded=result.n_segments_decoded,
        n_erasures=result.n_erasures,
        fingerprint_confidence=result.fingerprint_confidence,
    )


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=200,
    summary="Verify watermark authenticity",
    description=(
        "Submit a media file for watermark verification. "
        "Routes to AV pipeline (verify_av) when both audio and video are present. "
        "Routes to video-only pipeline (verify) for video-without-audio containers. "
        "Routes to audio-only pipeline (verify_audio) for audio-only containers. "
        "Returns 200 for all verification outcomes including RED verdicts — "
        "a RED verdict is a valid result, not an HTTP error. "
        "Returns 400 only for malformed requests (missing file, no streams detected). "
        "Returns 500 only for unexpected infrastructure failures."
    ),
)
async def verify_media(
    request: Request,
    file: UploadFile = File(..., description="Audio or video media file to verify"),
) -> VerificationResponse:
    """POST /verify — submit media for cryptographic watermark verification."""
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 GB)")

    # Save upload to a temp file — MediaService needs a real path
    suffix = Path(file.filename or "upload").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(content)

    try:
        media = MediaService()
        service = VerificationService()

        # Retrieve infrastructure dependencies from app state
        storage = request.app.state.storage
        registry = request.app.state.registry

        # Probe container type and route accordingly
        try:
            profile = media.probe(tmp_path)
        except ValueError as exc:
            logger.warning("verify: media probe failed for %s: %s", file.filename, exc)
            raise HTTPException(status_code=400, detail="Could not read media file — unsupported format or corrupted")

        org_id = getattr(request.state, "org_id", None)
        logger.debug("verify: file=%s, has_video=%s, has_audio=%s, org_id=%s",
                      file.filename, profile.has_video, profile.has_audio, org_id)

        # Use org-specific pepper for cryptographic isolation (Phase C)
        pepper = await _resolve_pepper(org_id, getattr(request.app.state, "db_session_factory", None))

        if profile.has_video and profile.has_audio:
            # AV container — use combined pipeline (Phase 5)
            result = await service.verify_av(
                media_path=tmp_path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=pepper,
                org_id=org_id,
            )
        elif profile.has_video:
            # Video-only container — use Phase 4 pipeline
            result = await service.verify(
                media_path=tmp_path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=pepper,
                org_id=org_id,
            )
        elif profile.has_audio:
            # Audio-only container
            result = await service.verify_audio(
                media_path=tmp_path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=pepper,
                org_id=org_id,
            )
        else:
            raise HTTPException(status_code=400, detail="No audio or video stream detected")

    finally:
        tmp_path.unlink(missing_ok=True)

    response = _to_response(result)

    # Enrich with human-readable names when verification found a candidate
    if response.content_id or response.author_id:
        author_name, org_name = await _resolve_author_and_org(
            response.author_id,
            response.content_id,
            getattr(request.app.state, "db_session_factory", None),
        )
        response.author_name = author_name
        response.org_name = org_name

    return response
