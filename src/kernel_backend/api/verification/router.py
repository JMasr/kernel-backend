"""Phase 4 + Phase 5 — POST /verify router."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from kernel_backend.api.verification.schemas import VerificationResponse
from kernel_backend.core.domain.verification import AVVerificationResult, VerificationResult
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService

router = APIRouter(tags=["verification"])

_PEPPER = os.environ.get("WATERMARK_PEPPER", "kernel-default-pepper-32b!").encode()
_MAX_BYTES = 150 * 1024 * 1024  # 150 MB


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
        raise HTTPException(status_code=400, detail="File too large (max 150 MB)")

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
        profile = media.probe(tmp_path)
        org_id = getattr(request.state, "org_id", None)

        if profile.has_video and profile.has_audio:
            # AV container — use combined pipeline (Phase 5)
            result = await service.verify_av(
                media_path=tmp_path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=_PEPPER,
                org_id=org_id,
            )
        elif profile.has_video:
            # Video-only container — use Phase 4 pipeline
            result = await service.verify(
                media_path=tmp_path,
                media=media,
                storage=storage,
                registry=registry,
                pepper=_PEPPER,
                org_id=org_id,
            )
        else:
            raise HTTPException(status_code=400, detail="No audio or video stream detected")

    finally:
        tmp_path.unlink(missing_ok=True)

    return _to_response(result)
