from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

logger = logging.getLogger("kernel.signing")
from arq.jobs import Job, JobStatus

from kernel_backend.api.rate_limit import limiter
from kernel_backend.api.signing.schemas import (
    SignJobResponse,
    SignJobResult,
    SignJobStatusResponse,
)

router = APIRouter(tags=["signing"])

_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


async def _get_org_pepper_hex(org_id, session_factory) -> str | None:
    """Return hex-encoded pepper_v1 for the org, or None if not set.

    Pepper lookup is best-effort: if the database is momentarily unreachable
    we log a warning and fall back to the system pepper rather than failing
    the whole /sign request.
    """
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
        logger.warning(
            "pepper_lookup_failed",
            extra={"org_id": str(org_id)},
            exc_info=True,
        )
    return None


@router.post("/sign", status_code=202, response_model=SignJobResponse)
@limiter.limit("10/minute")
async def sign(
    request: Request,
    file: UploadFile = File(..., description="Audio or AV media file"),
    certificate_json: str = Form(..., description="Certificate JSON from POST /identity/generate"),
    private_key_pem: str = Form(..., description="Ed25519 private key PEM"),
) -> SignJobResponse:
    """Enqueue a signing job and return immediately with job_id.

    The upload body is streamed to a temp file on the shared ``signing_tmp``
    volume; probe, normalisation and duration validation all run inside the
    ARQ worker (``process_sign_job``) so this endpoint returns as soon as the
    bytes are on disk. Validation failures surface via ``GET /sign/{job_id}``
    with ``status="failed"`` and a user-facing ``error`` string.
    """
    logger.debug(
        "sign.form_received",
        extra={
            "upload_filename": file.filename,
            "cert_json_len": len(certificate_json) if certificate_json else 0,
            "pkey_len": len(private_key_pem) if private_key_pem else 0,
        },
    )

    user_id: str | None = getattr(request.state, "user_id", None)
    user_email: str | None = getattr(request.state, "email", None)
    auth_type: str = getattr(request.state, "auth_type", "")

    # Scope check for API key auth
    if auth_type == "api_key":
        if "sign" not in getattr(request.state, "scopes", []):
            raise HTTPException(status_code=403, detail="API key does not have 'sign' scope")

    # Parse certificate JSON — required for both JWT and API key auth
    # NB: never log ``certificate_json`` — it contains the author_id and public key.
    try:
        cert_data = json.loads(certificate_json)
        cert_author_id = cert_data.get("author_id", "")
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "sign.cert_json_invalid",
            extra={"reason": str(exc), "cert_json_len": len(certificate_json) if certificate_json else 0},
        )
        raise HTTPException(status_code=422, detail="Invalid certificate JSON")

    if user_id is not None:
        # JWT path: certificate must belong to the authenticated user
        if cert_author_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Certificate does not belong to the authenticated user",
            )
    elif auth_type == "api_key":
        # API key path: author_id must be an identity registered under the same org
        if not cert_author_id:
            raise HTTPException(status_code=422, detail="Certificate missing author_id")
        org_id_for_check = getattr(request.state, "org_id", None)
        try:
            from kernel_backend.infrastructure.database.models import Identity
            from sqlalchemy import select as _sa_select
            session_factory = request.app.state.db_session_factory
            async with session_factory() as _session:
                _result = await _session.execute(
                    _sa_select(Identity.org_id).where(Identity.author_id == cert_author_id)
                )
                identity_org_id = _result.scalar_one_or_none()
        except Exception:
            logger.exception("sign.identity_org_lookup_failed", extra={"author_id": cert_author_id})
            raise HTTPException(status_code=500, detail="Could not verify certificate ownership")
        if identity_org_id is None:
            raise HTTPException(
                status_code=403,
                detail="Certificate author_id not found",
            )
        if identity_org_id != org_id_for_check:
            raise HTTPException(
                status_code=403,
                detail="Certificate author_id does not belong to the authenticated organization",
            )

    suffix = Path(file.filename or "upload.aac").suffix or ".aac"

    # Extension allowlist — reject unsupported formats early
    from kernel_backend.core.services.format_validation import validate_media_extension
    try:
        validate_media_extension(file.filename or f"upload{suffix}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Stream the request body to disk in 1 MB chunks. Avoids a 70 MB+ RAM
    # spike per upload and enforces the size cap without materialising the
    # full body. On size violation, the partially-written temp file is
    # unlinked immediately.
    _CHUNK = 1024 * 1024
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        media_path = tmp.name
        total = 0
        while True:
            chunk = await file.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_BYTES:
                tmp.close()
                Path(media_path).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 2 GB)")
            tmp.write(chunk)

    org_id = getattr(request.state, "org_id", None)

    # Fetch org-specific pepper for cryptographic isolation
    org_pepper_hex = await _get_org_pepper_hex(org_id, request.app.state.db_session_factory)

    redis_pool = request.app.state.redis_pool
    if redis_pool is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable — configure REDIS_HOST and REDIS_PASSWORD in .env")

    request_id = getattr(request.state, "request_id", None)
    trace_id = getattr(request.state, "trace_id", None)

    job = await redis_pool.enqueue_job(
        "process_sign_job",
        media_path=media_path,
        certificate_json=certificate_json,
        private_key_pem=private_key_pem,
        org_id=str(org_id) if org_id is not None else None,
        org_pepper_hex=org_pepper_hex,
        original_filename=file.filename or "",
        user_email=user_email,
        request_id=request_id,
        trace_id=trace_id,
    )

    # Initialize job status in Redis for progress tracking
    await redis_pool.set(
        f"job:{job.job_id}:status",
        json.dumps({"job_id": job.job_id, "status": "pending", "progress": 0}),
        ex=3600,
    )

    logger.info(
        "sign.enqueued",
        extra={
            "job_id": job.job_id,
            "upload_filename": file.filename,
            "bytes": total,
            "org_id": str(org_id) if org_id is not None else None,
        },
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
