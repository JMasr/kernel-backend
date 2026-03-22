from __future__ import annotations

from pydantic import BaseModel


class SignJobResponse(BaseModel):
    """Returned immediately on POST /sign — job has been enqueued."""
    job_id: str
    status: str = "queued"


class SignJobResult(BaseModel):
    """Embedded in SignJobStatusResponse when the job is complete."""
    content_id: str
    signed_media_key: str
    active_signals: list[str]
    rs_n: int


class SignJobStatusResponse(BaseModel):
    """Returned on GET /sign/{job_id}."""
    job_id: str
    status: str          # "queued" | "in_progress" | "complete" | "not_found" | "failed"
    progress: int = 0    # 0-100
    result: SignJobResult | None = None
    error: str | None = None  # Human-readable error message when status == "failed"
