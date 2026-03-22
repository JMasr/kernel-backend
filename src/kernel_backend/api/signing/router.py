from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

logger = logging.getLogger("kernel.signing")
from arq.jobs import Job, JobStatus

from kernel_backend.api.signing.schemas import (
    SignJobResponse,
    SignJobResult,
    SignJobStatusResponse,
)

router = APIRouter(tags=["signing"])

_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


async def _get_org_pepper_hex(org_id, session_factory) -> str | None:
    """Return hex-encoded pepper_v1 for the org, or None if not set."""
    if org_id is None:
        return None
    try:
        from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository
        async with session_factory() as session:
            repo = OrganizationRepository(session)
            org = await repo.get_organization_by_id(org_id)
            if org and org.pepper_v1:
                return org.pepper_v1
    except Exception:
        pass
    return None


@router.post("/sign", status_code=202, response_model=SignJobResponse)
async def sign(
    request: Request,
    file: UploadFile = File(..., description="Audio or AV media file"),
    certificate_json: str = Form(..., description="Certificate JSON from POST /identity/generate"),
    private_key_pem: str = Form(..., description="Ed25519 private key PEM"),
) -> SignJobResponse:
    """Enqueue a signing job and return immediately with job_id."""
    logger.debug("sign: received form fields — file=%s, cert_json_len=%d, pkey_len=%d",
                 file.filename, len(certificate_json), len(private_key_pem))
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 GB)")

    # Ownership check — cert.author_id must match the authenticated user (JWT only)
    user_id: str | None = getattr(request.state, "user_id", None)
    logger.debug("sign: user_id=%s, certificate_json type=%s, len=%s, preview=%.200s",
                 user_id, type(certificate_json).__name__, len(certificate_json) if certificate_json else 0,
                 certificate_json[:200] if certificate_json else "<empty>")
    if user_id is not None:
        try:
            cert_data = json.loads(certificate_json)
            cert_author_id = cert_data.get("author_id", "")
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("sign: invalid certificate_json: %s — raw value: %.500s", exc, certificate_json)
            raise HTTPException(status_code=422, detail="Invalid certificate JSON")
        if cert_author_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Certificate does not belong to the authenticated user",
            )

    suffix = Path(file.filename or "upload.aac").suffix or ".aac"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        media_path = tmp.name

    # Pre-queue duration validation — reject too-short / too-long files immediately
    try:
        from kernel_backend.infrastructure.media.media_service import MediaService
        profile = MediaService().probe(Path(media_path))
        _MIN_VIDEO_DURATION = 85   # 17 segments × 5 s/segment
        _MIN_AUDIO_DURATION = 34   # 17 segments × 2 s/segment
        _MAX_DURATION = 3600       # 1 hour

        if profile.duration_s > _MAX_DURATION:
            raise HTTPException(
                status_code=422,
                detail=f"File is too long. Your file is approximately {int(profile.duration_s)} seconds, "
                       f"but the maximum allowed duration is {_MAX_DURATION} seconds (1 hour).",
            )
        if profile.has_video and profile.duration_s < _MIN_VIDEO_DURATION:
            raise HTTPException(
                status_code=422,
                detail=f"Video is too short to sign. Your file is approximately {int(profile.duration_s)} seconds, "
                       f"but the minimum required duration is {_MIN_VIDEO_DURATION} seconds.",
            )
        if profile.has_audio and not profile.has_video and profile.duration_s < _MIN_AUDIO_DURATION:
            raise HTTPException(
                status_code=422,
                detail=f"Audio is too short to sign. Your file is approximately {int(profile.duration_s)} seconds, "
                       f"but the minimum required duration is {_MIN_AUDIO_DURATION} seconds.",
            )
        if not profile.has_audio and not profile.has_video:
            raise HTTPException(
                status_code=422,
                detail="File contains no audio or video streams.",
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read media file: {exc}")
    except Exception as exc:
        logger.warning("sign: media probe failed (non-fatal): %s", exc)

    org_id = getattr(request.state, "org_id", None)

    # Fetch org-specific pepper for cryptographic isolation
    org_pepper_hex = await _get_org_pepper_hex(org_id, request.app.state.db_session_factory)

    redis_pool = request.app.state.redis_pool
    if redis_pool is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable — configure REDIS_HOST and REDIS_PASSWORD in .env")
    job = await redis_pool.enqueue_job(
        "process_sign_job",
        media_path=media_path,
        certificate_json=certificate_json,
        private_key_pem=private_key_pem,
        org_id=str(org_id) if org_id is not None else None,
        org_pepper_hex=org_pepper_hex,
        original_filename=file.filename or "",
    )

    # Initialize job status in Redis for progress tracking
    await redis_pool.set(
        f"job:{job.job_id}:status",
        json.dumps({"job_id": job.job_id, "status": "pending", "progress": 0}),
        ex=3600,
    )

    return SignJobResponse(job_id=job.job_id, status="queued")


@router.get("/sign/{job_id}", status_code=200, response_model=SignJobStatusResponse)
async def sign_status(job_id: str, request: Request) -> SignJobStatusResponse:
    """Poll the status of an enqueued signing job."""
    redis_pool = request.app.state.redis_pool
    if redis_pool is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable — configure REDIS_HOST and REDIS_PASSWORD in .env")

    # Check Redis for progress-tracked status (set by POST /sign and updated by worker)
    status_json = await redis_pool.get(f"job:{job_id}:status")
    if status_json is not None:
        data = json.loads(status_json)
        result_data = data.get("result")
        result = None
        if result_data:
            result = SignJobResult(
                content_id=result_data.get("content_id", ""),
                signed_media_key=result_data.get("signed_media_key", ""),
                active_signals=result_data.get("active_signals", []),
                rs_n=result_data.get("rs_n", 0),
            )
        return SignJobStatusResponse(
            job_id=job_id,
            status=data.get("status", "unknown"),
            progress=data.get("progress", 0),
            result=result,
            error=data.get("error"),
        )

    # Fall back to ARQ native job status (no progress tracking)
    job = Job(job_id=job_id, redis=redis_pool)
    status = await job.status()

    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    if status in (JobStatus.queued, JobStatus.deferred):
        return SignJobStatusResponse(job_id=job_id, status="queued", progress=0)

    if status == JobStatus.in_progress:
        return SignJobStatusResponse(job_id=job_id, status="in_progress", progress=0)

    # complete or failed
    info = await job.info()
    if info is None or info.result is None:
        return SignJobStatusResponse(job_id=job_id, status="failed", progress=0)

    raw = info.result
    if isinstance(raw, Exception):
        return SignJobStatusResponse(job_id=job_id, status="failed", progress=0, error=str(raw))

    result_dict = raw if isinstance(raw, dict) else {}
    return SignJobStatusResponse(
        job_id=job_id,
        status="complete",
        progress=100,
        result=SignJobResult(
            content_id=result_dict.get("content_id", ""),
            signed_media_key=result_dict.get("signed_media_key", ""),
            active_signals=result_dict.get("active_signals", []),
            rs_n=result_dict.get("rs_n", 0),
        ),
    )
