"""Unit tests for POST /identity/generate (Phase B.1 — auth required, author_id = user_id)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from kernel_backend.api.dependencies import get_session
from kernel_backend.api.identity.router import router

_TEST_USER_ID = "user_test_abc123"
_TEST_ORG_ID = UUID("00000000-0000-0000-0000-000000000001")


class _InjectAuth(BaseHTTPMiddleware):
    """Test-only middleware: injects user_id and org_id into request.state."""

    def __init__(self, app, user_id: str | None = _TEST_USER_ID, org_id: UUID | None = _TEST_ORG_ID) -> None:
        super().__init__(app)
        self._user_id = user_id
        self._org_id = org_id

    async def dispatch(self, request: Request, call_next):
        request.state.user_id = self._user_id
        request.state.org_id = self._org_id
        request.state.is_admin = False
        request.state.auth_type = "neon_auth" if self._user_id else "public"
        return await call_next(request)


def _make_app(user_id=_TEST_USER_ID, org_id=_TEST_ORG_ID) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/identity")
    app.add_middleware(_InjectAuth, user_id=user_id, org_id=org_id)
    mock_session = AsyncMock(spec=AsyncSession)
    app.dependency_overrides[get_session] = lambda: mock_session
    return app


def _no_existing_cert():
    """IdentityRepository mock: no existing cert for user."""
    mock = MagicMock()
    mock.get_by_author_id = AsyncMock(return_value=None)
    mock.create_with_org = AsyncMock(return_value=None)
    return mock


def _existing_cert():
    """IdentityRepository mock: cert already exists."""
    mock = MagicMock()
    mock.get_by_author_id = AsyncMock(return_value=MagicMock())  # truthy = exists
    return mock


def _delete_found():
    """IdentityRepository mock: delete succeeds (identity existed)."""
    mock = MagicMock()
    mock.delete_by_author_id = AsyncMock(return_value=True)
    return mock


def _delete_not_found():
    """IdentityRepository mock: delete fails (no identity)."""
    mock = MagicMock()
    mock.delete_by_author_id = AsyncMock(return_value=False)
    return mock


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


def test_generate_identity_returns_201() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_no_existing_cert(),
        ):
            response = client.post(
                "/identity/generate",
                json={"name": "Alice", "institution": "ACME Corp"},
            )
    assert response.status_code == 201


def test_author_id_equals_user_id() -> None:
    """After Phase B.1, author_id must be the authenticated user_id (not a hash)."""
    app = _make_app()
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_no_existing_cert(),
        ):
            response = client.post(
                "/identity/generate",
                json={"name": "Alice", "institution": "ACME"},
            )
    assert response.status_code == 201
    assert response.json()["author_id"] == _TEST_USER_ID


def test_response_contains_private_key_pem() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_no_existing_cert(),
        ):
            response = client.post(
                "/identity/generate",
                json={"name": "Bob", "institution": "Uni"},
            )
    assert response.status_code == 201
    data = response.json()
    assert "private_key_pem" in data
    assert data["private_key_pem"].startswith("-----BEGIN PRIVATE KEY-----")


def test_private_key_is_valid_ed25519_pem() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_no_existing_cert(),
        ):
            response = client.post(
                "/identity/generate",
                json={"name": "Carol", "institution": "Lab"},
            )
    data = response.json()
    pub_key = load_pem_public_key(data["public_key_pem"].encode())
    assert isinstance(pub_key, Ed25519PublicKey)


def test_empty_name_returns_422() -> None:
    app = _make_app()
    with TestClient(app) as client:
        response = client.post(
            "/identity/generate",
            json={"name": "", "institution": "Uni"},
        )
    assert response.status_code == 422


def test_empty_institution_returns_422() -> None:
    app = _make_app()
    with TestClient(app) as client:
        response = client.post(
            "/identity/generate",
            json={"name": "Alice", "institution": ""},
        )
    assert response.status_code == 422


def test_no_user_id_returns_401() -> None:
    """Unauthenticated request (no user_id) → 401."""
    app = _make_app(user_id=None)
    with TestClient(app) as client:
        response = client.post(
            "/identity/generate",
            json={"name": "Alice", "institution": "ACME"},
        )
    assert response.status_code == 401


def test_no_org_returns_403() -> None:
    """Authenticated user with no org → 403."""
    app = _make_app(org_id=None)
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_no_existing_cert(),
        ):
            response = client.post(
                "/identity/generate",
                json={"name": "Alice", "institution": "ACME"},
            )
    assert response.status_code == 403


def test_duplicate_key_returns_409() -> None:
    """Second call for same user → 409 Conflict."""
    app = _make_app()
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_existing_cert(),
        ):
            response = client.post(
                "/identity/generate",
                json={"name": "Alice", "institution": "ACME"},
            )
    assert response.status_code == 409


# ---------------------------------------------------------------------------
# DELETE /identity/me tests
# ---------------------------------------------------------------------------


def test_delete_identity_returns_204() -> None:
    """Authenticated user with existing identity → 204."""
    app = _make_app()
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_delete_found(),
        ):
            response = client.delete("/identity/me")
    assert response.status_code == 204


def test_delete_identity_not_found_returns_404() -> None:
    """Authenticated user with no identity → 404."""
    app = _make_app()
    with TestClient(app) as client:
        with patch(
            "kernel_backend.api.identity.router.IdentityRepository",
            return_value=_delete_not_found(),
        ):
            response = client.delete("/identity/me")
    assert response.status_code == 404


def test_delete_identity_unauthenticated_returns_401() -> None:
    """No user_id → 401."""
    app = _make_app(user_id=None)
    with TestClient(app) as client:
        response = client.delete("/identity/me")
    assert response.status_code == 401
