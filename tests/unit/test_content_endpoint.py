"""Unit tests for content management endpoints (GET /content, DELETE /content/{id})."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from kernel_backend.api.content.router import router
from kernel_backend.api.dependencies import get_session

_ORG_ID = uuid4()


class _InjectOrgId(BaseHTTPMiddleware):
    """Test-only middleware that injects a fixed org_id into request.state."""

    def __init__(self, app, org_id: UUID | None) -> None:
        super().__init__(app)
        self._org_id = org_id

    async def dispatch(self, request: Request, call_next):
        request.state.org_id = self._org_id
        return await call_next(request)


def _make_app(org_id: UUID | None = _ORG_ID) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(_InjectOrgId, org_id=org_id)

    mock_session = AsyncMock(spec=AsyncSession)
    app.dependency_overrides[get_session] = lambda: mock_session
    return app


def _mock_repo(delete_return: bool = True) -> MagicMock:
    mock = MagicMock()
    mock.delete_by_content_id = AsyncMock(return_value=delete_return)
    mock.list_by_org_id = AsyncMock(return_value=[])
    mock.count_by_org_id = AsyncMock(return_value=0)
    return mock


# ---------------------------------------------------------------------------
# DELETE /content/{content_id}
# ---------------------------------------------------------------------------


class TestDeleteContent:
    def test_delete_success_returns_204(self) -> None:
        app = _make_app()
        with TestClient(app, raise_server_exceptions=True) as client:
            with (
                __import__("unittest.mock", fromlist=["patch"]).patch(
                    "kernel_backend.api.content.router.VideoRepository",
                    return_value=_mock_repo(delete_return=True),
                )
            ):
                response = client.delete(f"/content/{uuid4()}")

        assert response.status_code == 204

    def test_delete_not_found_returns_404(self) -> None:
        app = _make_app()
        with TestClient(app, raise_server_exceptions=True) as client:
            with (
                __import__("unittest.mock", fromlist=["patch"]).patch(
                    "kernel_backend.api.content.router.VideoRepository",
                    return_value=_mock_repo(delete_return=False),
                )
            ):
                response = client.delete(f"/content/{uuid4()}")

        assert response.status_code == 404
        detail = response.json()["detail"].lower()
        assert "not found" in detail or "access denied" in detail

    def test_delete_no_auth_returns_401(self) -> None:
        """No org_id in request.state (org_id=None injected) → 401."""
        app = _make_app(org_id=None)
        with TestClient(app, raise_server_exceptions=True) as client:
            with (
                __import__("unittest.mock", fromlist=["patch"]).patch(
                    "kernel_backend.api.content.router.VideoRepository",
                    return_value=_mock_repo(delete_return=True),
                )
            ):
                response = client.delete(f"/content/{uuid4()}")

        assert response.status_code == 401

    def test_delete_wrong_org_returns_404(self) -> None:
        """Org mismatch: repository returns False because WHERE org_id=X finds nothing."""
        app = _make_app()
        with TestClient(app, raise_server_exceptions=True) as client:
            with (
                __import__("unittest.mock", fromlist=["patch"]).patch(
                    "kernel_backend.api.content.router.VideoRepository",
                    return_value=_mock_repo(delete_return=False),
                )
            ):
                response = client.delete(f"/content/{uuid4()}")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /content — smoke test to verify DELETE didn't break routing
# ---------------------------------------------------------------------------


class TestListContent:
    def test_list_returns_200_with_empty_result(self) -> None:
        app = _make_app()
        with TestClient(app) as client:
            with (
                __import__("unittest.mock", fromlist=["patch"]).patch(
                    "kernel_backend.api.content.router.VideoRepository",
                    return_value=_mock_repo(),
                )
            ):
                response = client.get("/content")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["page"] == 1

    def test_list_no_auth_returns_401(self) -> None:
        app = _make_app(org_id=None)
        with TestClient(app) as client:
            response = client.get("/content")

        assert response.status_code == 401
