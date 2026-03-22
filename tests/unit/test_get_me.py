"""Unit tests for GET /me (Phase D)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from kernel_backend.api.users.router import router

_ORG_ID = UUID("00000000-0000-0000-0000-000000000002")
_USER_ID = "user_test_xyz"
_ADMIN_EMAIL = "admin@example.com"


class _InjectState(BaseHTTPMiddleware):
    def __init__(self, app, *, user_id=_USER_ID, org_id=_ORG_ID, is_admin=False, email=None) -> None:
        super().__init__(app)
        self._user_id = user_id
        self._org_id = org_id
        self._is_admin = is_admin
        self._email = email

    async def dispatch(self, request: Request, call_next):
        request.state.user_id = self._user_id
        request.state.org_id = self._org_id
        request.state.is_admin = self._is_admin
        request.state.email = self._email
        return await call_next(request)


def _make_app(**kwargs) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(_InjectState, **kwargs)

    # Mock db_session_factory on app.state
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    app.state.db_session_factory = MagicMock(return_value=mock_session)
    return app


def _mock_org_repo(org_name="Kernel Alpha"):
    mock = MagicMock()
    org = MagicMock()
    org.name = org_name
    mock.get_organization_by_id = AsyncMock(return_value=org)
    return mock


def _mock_identity_repo(has_cert=True):
    mock = MagicMock()
    mock.get_by_author_id = AsyncMock(return_value=MagicMock() if has_cert else None)
    return mock


class TestGetMe:
    def test_unauthenticated_returns_401(self) -> None:
        app = _make_app(user_id=None)
        with TestClient(app) as client:
            response = client.get("/me")
        assert response.status_code == 401

    def test_admin_returns_master_admin_role(self) -> None:
        app = _make_app(is_admin=True, email=_ADMIN_EMAIL)
        with TestClient(app) as client:
            with (
                patch("kernel_backend.api.users.router.OrganizationRepository", return_value=_mock_org_repo()),
                patch("kernel_backend.api.users.router.IdentityRepository", return_value=_mock_identity_repo()),
            ):
                response = client.get("/me")
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "master_admin"
        assert data["email"] == _ADMIN_EMAIL
        assert data["user_id"] == _USER_ID

    def test_member_returns_member_role(self) -> None:
        app = _make_app(is_admin=False, email="user@example.com")
        with TestClient(app) as client:
            with (
                patch("kernel_backend.api.users.router.OrganizationRepository", return_value=_mock_org_repo()),
                patch("kernel_backend.api.users.router.IdentityRepository", return_value=_mock_identity_repo()),
            ):
                response = client.get("/me")
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "member"

    def test_includes_org_name(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            with (
                patch("kernel_backend.api.users.router.OrganizationRepository", return_value=_mock_org_repo("Acme Corp")),
                patch("kernel_backend.api.users.router.IdentityRepository", return_value=_mock_identity_repo()),
            ):
                response = client.get("/me")
        assert response.status_code == 200
        assert response.json()["org_name"] == "Acme Corp"

    def test_has_key_pair_true_when_cert_exists(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            with (
                patch("kernel_backend.api.users.router.OrganizationRepository", return_value=_mock_org_repo()),
                patch("kernel_backend.api.users.router.IdentityRepository", return_value=_mock_identity_repo(has_cert=True)),
            ):
                response = client.get("/me")
        assert response.json()["has_key_pair"] is True

    def test_has_key_pair_false_when_no_cert(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            with (
                patch("kernel_backend.api.users.router.OrganizationRepository", return_value=_mock_org_repo()),
                patch("kernel_backend.api.users.router.IdentityRepository", return_value=_mock_identity_repo(has_cert=False)),
            ):
                response = client.get("/me")
        assert response.json()["has_key_pair"] is False

    def test_no_org_returns_null_org_fields(self) -> None:
        app = _make_app(org_id=None)
        with TestClient(app) as client:
            with (
                patch("kernel_backend.api.users.router.IdentityRepository", return_value=_mock_identity_repo()),
            ):
                response = client.get("/me")
        assert response.status_code == 200
        data = response.json()
        assert data["org_id"] is None
        assert data["org_name"] is None
