"""
Unit tests for GET /sign/{job_id} job status endpoint.

Tests that the endpoint reads from Redis key job:{job_id}:status and
returns the correct JobStatusResponse, including progress tracking.
"""
from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kernel_backend.api.signing.router import router as signing_router
from kernel_backend.api.signing.schemas import SignJobStatusResponse


def _build_app(redis_pool: object) -> FastAPI:
    """Build a minimal app with a mock redis_pool in app.state."""
    app = FastAPI()
    app.include_router(signing_router)
    app.state.redis_pool = redis_pool
    app.state.storage = AsyncMock()
    return app


def _mock_redis(get_return: object = None) -> MagicMock:
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=get_return)
    redis.set = AsyncMock()
    redis.enqueue_job = AsyncMock()
    return redis


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_get_job_status_processing():
    """Returns processing status with progress from Redis."""
    job_id = uuid4()
    status_data = {
        "job_id": str(job_id),
        "status": "processing",
        "progress": 50,
    }
    redis = _mock_redis(get_return=json.dumps(status_data))
    client = TestClient(_build_app(redis), raise_server_exceptions=False)

    response = client.get(f"/sign/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == str(job_id)
    assert data["status"] == "processing"
    assert data["progress"] == 50
    assert data["result"] is None


def test_get_job_status_not_found_raises_404():
    """Returns 404 when job key is absent from Redis and ARQ."""
    from unittest.mock import patch
    from arq.jobs import JobStatus

    job_id = uuid4()
    redis = _mock_redis(get_return=None)

    # Also mock ARQ Job.status() to return not_found
    mock_job = AsyncMock()
    mock_job.status = AsyncMock(return_value=JobStatus.not_found)

    client = TestClient(_build_app(redis), raise_server_exceptions=False)

    with patch("kernel_backend.api.signing.router.Job", return_value=mock_job):
        response = client.get(f"/sign/{job_id}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_job_status_completed_with_result():
    """Returns completed status with result dict when job is done."""
    job_id = uuid4()
    content_id = str(uuid4())
    status_data = {
        "job_id": str(job_id),
        "status": "completed",
        "progress": 100,
        "result": {
            "content_id": content_id,
            "signed_media_key": f"org/signed/{content_id}.mp4",
            "active_signals": ["audio", "video"],
            "rs_n": 24,
        },
    }
    redis = _mock_redis(get_return=json.dumps(status_data))
    client = TestClient(_build_app(redis), raise_server_exceptions=False)

    response = client.get(f"/sign/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["progress"] == 100
    assert data["result"]["content_id"] == content_id
    assert data["result"]["rs_n"] == 24


def test_get_job_status_failed_with_error():
    """Returns failed status when worker set error."""
    job_id = uuid4()
    status_data = {
        "job_id": str(job_id),
        "status": "failed",
        "progress": 0,
        "error": "Signing failed: audio too short",
    }
    redis = _mock_redis(get_return=json.dumps(status_data))
    client = TestClient(_build_app(redis), raise_server_exceptions=False)

    response = client.get(f"/sign/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["progress"] == 0
    # error is not in the response schema (SignJobStatusResponse), just verifying status/progress
