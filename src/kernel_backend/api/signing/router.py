from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from kernel_backend.api.signing.schemas import (
    SignJobResponse,
    SignJobResult,
    SignJobStatusResponse,
)

router = APIRouter(tags=["signing"])

_MAX_BYTES = 150 * 1024 * 1024  # 150 MB


@router.post("/sign", status_code=202, response_model=SignJobResponse)
async def sign(
    request: Request,
    file: UploadFile = File(..., description="Audio or AV media file"),
    certificate_json: str = Form(..., description="Certificate JSON from POST /identity/generate"),
    private_key_pem: str = Form(..., description="Ed25519 private key PEM"),
) -> SignJobResponse:
    """Enqueue a signing job and return immediately with job_id."""
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 150 MB)")

    # Persist upload to a temporary path accessible by the worker
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "upload.aac").suffix or ".aac"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        media_path = tmp.name

    redis_pool = request.app.state.redis_pool
    job = await redis_pool.enqueue_job(
        "process_sign_job",
        media_path=media_path,
        certificate_json=certificate_json,
        private_key_pem=private_key_pem,
    )

    return SignJobResponse(job_id=job.job_id, status="queued")


@router.get("/sign/{job_id}", status_code=200, response_model=SignJobStatusResponse)
async def sign_status(job_id: str, request: Request) -> SignJobStatusResponse:
    """Poll the status of an enqueued signing job."""
    from arq.jobs import Job, JobStatus

    redis_pool = request.app.state.redis_pool
    job = Job(job_id=job_id, redis=redis_pool)

    status = await job.status()

    if status == JobStatus.not_found:
        return SignJobStatusResponse(job_id=job_id, status="not_found")

    if status in (JobStatus.queued, JobStatus.deferred):
        return SignJobStatusResponse(job_id=job_id, status="queued")

    if status == JobStatus.in_progress:
        return SignJobStatusResponse(job_id=job_id, status="in_progress")

    # complete or failed
    info = await job.info()
    if info is None or info.result is None:
        return SignJobStatusResponse(job_id=job_id, status="failed")

    raw = info.result
    if isinstance(raw, Exception):
        return SignJobStatusResponse(job_id=job_id, status="failed")

    result_dict = raw if isinstance(raw, dict) else {}
    return SignJobStatusResponse(
        job_id=job_id,
        status="complete",
        result=SignJobResult(
            content_id=result_dict.get("content_id", ""),
            signed_media_key=result_dict.get("signed_media_key", ""),
            active_signals=result_dict.get("active_signals", []),
            rs_n=result_dict.get("rs_n", 0),
        ),
    )
