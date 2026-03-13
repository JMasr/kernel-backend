"""Phase 4 — POST /verify router."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from kernel_backend.api.verification.schemas import VerificationResponse
from kernel_backend.core.services.verification_service import VerificationService
from kernel_backend.infrastructure.media.media_service import MediaService

router = APIRouter(tags=["verification"])

_PEPPER = os.environ.get("WATERMARK_PEPPER", "kernel-default-pepper-32b!").encode()
_MAX_BYTES = 150 * 1024 * 1024  # 150 MB


@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=200,
    summary="Verify watermark authenticity",
    description=(
        "Submit a media file for watermark verification. "
        "Returns 200 for all verification outcomes including RED verdicts — "
        "a RED verdict is a valid result, not an HTTP error. "
        "Returns 400 only for malformed requests (missing file, wrong content-type). "
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

        result = await service.verify(
            media_path=tmp_path,
            media=media,
            storage=storage,
            registry=registry,
            pepper=_PEPPER,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

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
