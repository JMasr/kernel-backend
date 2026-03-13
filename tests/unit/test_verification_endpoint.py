"""
tests/unit/test_verification_endpoint.py

Unit tests for POST /verify router — Phase 4 Step 5.
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kernel_backend.api.verification.router import router as verification_router
from kernel_backend.api.verification.schemas import VerificationResponse
from kernel_backend.core.domain.verification import RedReason, VerificationResult, Verdict


def _build_app() -> FastAPI:
    """Build a minimal FastAPI app with mocked app.state for testing."""
    app = FastAPI()
    app.include_router(verification_router)
    # Provide fake state objects that the router reads
    app.state.storage = AsyncMock()
    app.state.registry = AsyncMock()
    return app


def _result(**kwargs) -> VerificationResult:
    defaults = dict(
        verdict=Verdict.VERIFIED,
        content_id="cid-123",
        author_id="auth-1",
        author_public_key="---PEM---",
        red_reason=None,
        wid_match=True,
        signature_valid=True,
        n_segments_total=20,
        n_segments_decoded=18,
        n_erasures=2,
        fingerprint_confidence=0.95,
    )
    defaults.update(kwargs)
    return VerificationResult(**defaults)


def _fake_file(content: bytes = b"fake-video-data", filename: str = "test.mp4"):
    return ("file", (filename, io.BytesIO(content), "video/mp4"))


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_post_verify_returns_200_on_verified():
    """[BLOCKING] Returns HTTP 200 with VERIFIED verdict."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    result = _result(verdict=Verdict.VERIFIED)
    with patch(
        "kernel_backend.api.verification.router.VerificationService.verify",
        new_callable=AsyncMock,
        return_value=result,
    ), patch(
        "kernel_backend.api.verification.router.MediaService",
        return_value=MagicMock(),
    ):
        response = client.post("/verify", files=[_fake_file()])

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "VERIFIED"
    assert data["wid_match"] is True
    assert data["signature_valid"] is True


def test_post_verify_returns_200_on_red_candidate_not_found():
    """[BLOCKING] RED verdict returns HTTP 200 (not 404 or 422)."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    result = _result(
        verdict=Verdict.RED,
        red_reason=RedReason.CANDIDATE_NOT_FOUND,
        content_id=None,
        wid_match=False,
        signature_valid=False,
    )
    with patch(
        "kernel_backend.api.verification.router.VerificationService.verify",
        new_callable=AsyncMock,
        return_value=result,
    ), patch(
        "kernel_backend.api.verification.router.MediaService",
        return_value=MagicMock(),
    ):
        response = client.post("/verify", files=[_fake_file()])

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "RED"
    assert data["red_reason"] == "candidate_not_found"


def test_post_verify_returns_200_on_red_wid_mismatch():
    """[BLOCKING] WID_MISMATCH returns HTTP 200 (a valid forensic result)."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    result = _result(
        verdict=Verdict.RED,
        red_reason=RedReason.WID_MISMATCH,
        wid_match=False,
        signature_valid=False,
    )
    with patch(
        "kernel_backend.api.verification.router.VerificationService.verify",
        new_callable=AsyncMock,
        return_value=result,
    ), patch(
        "kernel_backend.api.verification.router.MediaService",
        return_value=MagicMock(),
    ):
        response = client.post("/verify", files=[_fake_file()])

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "RED"
    assert data["red_reason"] == "wid_mismatch"


def test_post_verify_returns_400_on_missing_file():
    """[BLOCKING] 422 (FastAPI validation) when no file is submitted."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/verify")
    # FastAPI returns 422 for missing required fields
    assert response.status_code == 422


def test_response_schema_has_no_top_level_score_field():
    """
    [BLOCKING] The response JSON must NOT contain top-level 'confidence' or 'score' fields.
    Those fields would imply score-based authentication — explicitly prohibited by spec.
    fingerprint_confidence IS allowed as a clearly-labelled diagnostic sub-field.
    """
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    result = _result(verdict=Verdict.VERIFIED)
    with patch(
        "kernel_backend.api.verification.router.VerificationService.verify",
        new_callable=AsyncMock,
        return_value=result,
    ), patch(
        "kernel_backend.api.verification.router.MediaService",
        return_value=MagicMock(),
    ):
        response = client.post("/verify", files=[_fake_file()])

    assert response.status_code == 200
    data = response.json()

    assert "confidence" not in data, (
        "Top-level 'confidence' field found — prohibited (use fingerprint_confidence sub-field)"
    )
    assert "score" not in data, (
        "Top-level 'score' field found — prohibited by Phase 4 spec"
    )
    # fingerprint_confidence IS allowed as a labelled diagnostic sub-field
    assert "fingerprint_confidence" in data
