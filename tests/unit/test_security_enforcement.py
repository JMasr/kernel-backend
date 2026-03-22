"""Unit tests for Phase B — ownership enforcement and user-scoped content."""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from kernel_backend.api.content.router import router as content_router
from kernel_backend.api.dependencies import get_session

_ORG_ID = UUID("00000000-0000-0000-0000-000000000001")
_USER_A = "user_alice_123"
_USER_B = "user_bob_456"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _InjectState(BaseHTTPMiddleware):
    def __init__(self, app, user_id=_USER_A, org_id=_ORG_ID, is_admin=False) -> None:
        super().__init__(app)
        self._user_id = user_id
        self._org_id = org_id
        self._is_admin = is_admin

    async def dispatch(self, request: Request, call_next):
        request.state.user_id = self._user_id
        request.state.org_id = self._org_id
        request.state.is_admin = self._is_admin
        return await call_next(request)


def _make_content_app(user_id=_USER_A, org_id=_ORG_ID, is_admin=False) -> FastAPI:
    app = FastAPI()
    app.include_router(content_router)
    app.add_middleware(_InjectState, user_id=user_id, org_id=org_id, is_admin=is_admin)
    mock_session = AsyncMock(spec=AsyncSession)
    app.dependency_overrides[get_session] = lambda: mock_session
    return app


def _mock_repo(user_a_count=3, user_b_count=2, total=5):
    mock = MagicMock()

    def list_side_effect(org_id, author_id=None, limit=20, offset=0):
        count = user_a_count if author_id == _USER_A else (user_b_count if author_id == _USER_B else total)
        return [(MagicMock(content_id=str(uuid4()), author_id=author_id or _USER_A, org_id=org_id, signed_media_key="org/signed_test.wav"), None, None)] * count

    def count_side_effect(org_id, author_id=None):
        if author_id == _USER_A:
            return user_a_count
        if author_id == _USER_B:
            return user_b_count
        return total

    mock.list_by_org_id = AsyncMock(side_effect=list_side_effect)
    mock.count_by_org_id = AsyncMock(side_effect=count_side_effect)
    mock.delete_by_content_id = AsyncMock(return_value=True)
    return mock


# ---------------------------------------------------------------------------
# Phase B.3 — GET /content user-scoped for members
# ---------------------------------------------------------------------------


class TestContentUserScoping:
    def test_member_sees_only_own_content(self) -> None:
        """Members get content scoped to their own user_id regardless of query param."""
        app = _make_content_app(user_id=_USER_A, is_admin=False)
        with TestClient(app) as client:
            with patch(
                "kernel_backend.api.content.router.VideoRepository",
                return_value=_mock_repo(),
            ) as mock_cls:
                # Member tries to view user B's content — should be ignored
                response = client.get(f"/content?author_id={_USER_B}")

        assert response.status_code == 200
        # The repo must have been called with author_id=USER_A (not USER_B)
        call_kwargs = mock_cls.return_value.list_by_org_id.call_args
        assert call_kwargs.kwargs.get("author_id") == _USER_A

    def test_admin_can_filter_by_any_author(self) -> None:
        """Admins can pass author_id query param freely."""
        app = _make_content_app(user_id=_USER_A, is_admin=True)
        with TestClient(app) as client:
            with patch(
                "kernel_backend.api.content.router.VideoRepository",
                return_value=_mock_repo(),
            ) as mock_cls:
                response = client.get(f"/content?author_id={_USER_B}")

        assert response.status_code == 200
        call_kwargs = mock_cls.return_value.list_by_org_id.call_args
        # Admin's requested author_id should be passed through
        assert call_kwargs.kwargs.get("author_id") == _USER_B

    def test_admin_no_filter_sees_all(self) -> None:
        """Admins without author_id filter see all org content."""
        app = _make_content_app(user_id=_USER_A, is_admin=True)
        with TestClient(app) as client:
            with patch(
                "kernel_backend.api.content.router.VideoRepository",
                return_value=_mock_repo(),
            ) as mock_cls:
                response = client.get("/content")

        assert response.status_code == 200
        call_kwargs = mock_cls.return_value.list_by_org_id.call_args
        assert call_kwargs.kwargs.get("author_id") is None

    def test_api_key_user_no_user_id_sees_all_org_content(self) -> None:
        """API key users have user_id=None — no forced author_id scope."""
        app = _make_content_app(user_id=None, is_admin=False)
        with TestClient(app) as client:
            with patch(
                "kernel_backend.api.content.router.VideoRepository",
                return_value=_mock_repo(),
            ) as mock_cls:
                response = client.get("/content")

        assert response.status_code == 200
        call_kwargs = mock_cls.return_value.list_by_org_id.call_args
        # user_id is None → no forced author filter
        assert call_kwargs.kwargs.get("author_id") is None


# ---------------------------------------------------------------------------
# Phase B.2 — POST /sign ownership check is tested via signing router
# The check rejects cert.author_id != user_id for JWT users.
# We test the logic via direct import.
# ---------------------------------------------------------------------------


class TestSignOwnershipCheck:
    def test_cert_belongs_to_caller_passes(self) -> None:
        """cert.author_id == user_id → no 403."""
        from kernel_backend.api.signing.router import router as signing_router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(signing_router)

        class _StateMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, req: Request, call_next):
                req.state.user_id = _USER_A
                req.state.org_id = _ORG_ID
                req.state.is_admin = False
                req.app.state.redis_pool = None
                req.app.state.db_session_factory = None
                return await call_next(req)

        app.add_middleware(_StateMiddleware)

        cert = {"author_id": _USER_A, "name": "Alice", "institution": "Lab", "public_key_pem": "pem", "created_at": "2025-01-01"}

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/sign",
                files={"file": ("test.aac", BytesIO(b"x" * 100), "audio/aac")},
                data={
                    "certificate_json": json.dumps(cert),
                    "private_key_pem": "private_pem",
                },
            )
        # Redis is None → 503 (queue unavailable), NOT 403 (auth failure)
        assert response.status_code == 503

    def test_wrong_author_id_returns_403(self) -> None:
        """cert.author_id != user_id → 403 Forbidden."""
        from kernel_backend.api.signing.router import router as signing_router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(signing_router)

        class _StateMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, req: Request, call_next):
                req.state.user_id = _USER_A
                req.state.org_id = _ORG_ID
                req.state.is_admin = False
                return await call_next(req)

        app.add_middleware(_StateMiddleware)

        cert = {"author_id": _USER_B, "name": "Bob", "institution": "Lab", "public_key_pem": "pem", "created_at": "2025-01-01"}

        with TestClient(app) as client:
            response = client.post(
                "/sign",
                files={"file": ("test.aac", BytesIO(b"x" * 100), "audio/aac")},
                data={
                    "certificate_json": json.dumps(cert),
                    "private_key_pem": "private_pem",
                },
            )
        assert response.status_code == 403

    def test_api_key_no_user_id_skips_check(self) -> None:
        """API key auth (user_id=None) → ownership check is skipped."""
        from kernel_backend.api.signing.router import router as signing_router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(signing_router)

        class _StateMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, req: Request, call_next):
                req.state.user_id = None  # API key
                req.state.org_id = _ORG_ID
                req.state.is_admin = False
                req.app.state.redis_pool = None
                req.app.state.db_session_factory = None
                return await call_next(req)

        app.add_middleware(_StateMiddleware)

        cert = {"author_id": "any-author-id", "name": "Bot", "institution": "CI", "public_key_pem": "pem", "created_at": "2025-01-01"}

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/sign",
                files={"file": ("test.aac", BytesIO(b"x" * 100), "audio/aac")},
                data={
                    "certificate_json": json.dumps(cert),
                    "private_key_pem": "private_pem",
                },
            )
        # Redis None → 503, not 403 — ownership check was skipped
        assert response.status_code == 503
