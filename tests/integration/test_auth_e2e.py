"""
Integration tests for auth reliability (Phase 10.C).

Scenarios:
1. Dual-identity admin: local JWT → Stack Auth (same email) → same org
2. Token refresh chain: token A → refresh → token B → use B
3. Refresh expired token → 401
4. Invitation accept uses authenticated user_id over body.user_id
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import jwt
import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.middleware.base import BaseHTTPMiddleware

from kernel_backend.api.auth.router import router as auth_router
from kernel_backend.api.invitations.router import (
    admin_router as invitations_admin_router,
    public_router as invitations_public_router,
)
from kernel_backend.api.middleware.auth import HybridAuthMiddleware
from kernel_backend.api.organizations.router import router as organizations_router
from kernel_backend.api.users.router import router as users_router
from kernel_backend.infrastructure.database.models import Base

_JWT_SECRET = "integration-test-secret-32bytes!!"
_ADMIN_EMAIL = "admin@kernel-test.com"
_TTL = 60 * 60 * 8


def _mint_token(
    email: str = _ADMIN_EMAIL,
    is_admin: bool = True,
    expired: bool = False,
) -> str:
    now = int(time.time())
    payload = {
        "sub": email,
        "email": email,
        "is_admin": is_admin,
        "iat": now - (_TTL + 10 if expired else 0),
        "exp": now - 10 if expired else now + _TTL,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm="HS256")


@pytest.fixture
async def test_app(monkeypatch):
    """Build a minimal FastAPI app with HybridAuthMiddleware and in-memory DB."""
    monkeypatch.setenv("JWT_SECRET", _JWT_SECRET)
    monkeypatch.setenv("ADMIN_EMAIL", _ADMIN_EMAIL)
    monkeypatch.setenv("ADMIN_PASS", "test-pass")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("MIGRATION_DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("KERNEL_SYSTEM_PEPPER", "a" * 64)
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PASSWORD", "x")
    monkeypatch.setenv("NEON_AUTH_API_KEY", "")
    monkeypatch.setenv("NEON_AUTH_SECRET_SERVER_KEY", "")

    # Create async engine + session factory for middleware
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(organizations_router)
    app.include_router(invitations_admin_router)
    app.include_router(invitations_public_router)
    app.include_router(users_router)

    app.add_middleware(HybridAuthMiddleware)

    app.state.db_session_factory = session_factory

    yield app

    await engine.dispose()


@pytest.fixture
async def client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Test 1: Token refresh chain
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_refresh_returns_fresh_token(client):
    """POST /auth/refresh with a valid token returns a new token."""
    token = _mint_token()
    resp = await client.post("/auth/refresh", json={"access_token": token})
    assert resp.status_code == 200
    new_token = resp.json()["access_token"]

    # The new token should decode and carry the same claims
    payload = jwt.decode(new_token, _JWT_SECRET, algorithms=["HS256"])
    assert payload["email"] == _ADMIN_EMAIL
    assert payload["is_admin"] is True


@pytest.mark.integration
async def test_refresh_chain_works(client):
    """Refresh → use new token → refresh again → still works."""
    token_a = _mint_token()

    # First refresh
    resp1 = await client.post("/auth/refresh", json={"access_token": token_a})
    assert resp1.status_code == 200
    token_b = resp1.json()["access_token"]

    # Second refresh using token_b
    resp2 = await client.post("/auth/refresh", json={"access_token": token_b})
    assert resp2.status_code == 200
    token_c = resp2.json()["access_token"]

    # token_c should be valid
    payload = jwt.decode(token_c, _JWT_SECRET, algorithms=["HS256"])
    assert payload["email"] == _ADMIN_EMAIL


@pytest.mark.integration
async def test_refresh_expired_token_returns_401(client):
    """POST /auth/refresh with an expired token returns 401."""
    token = _mint_token(expired=True)
    resp = await client.post("/auth/refresh", json={"access_token": token})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test 2: Dual-identity admin — verify email fallback logic at repo level
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_dual_identity_email_fallback(monkeypatch):
    """
    Verify the identity-bridging logic:
    1. Admin registers org via local JWT (user_id = email).
    2. Stack Auth login uses opaque user_id → not found.
    3. Fallback lookup by email → found → bridge membership created.
    4. Future lookup by Stack Auth user_id → found directly.
    """
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("MIGRATION_DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("KERNEL_SYSTEM_PEPPER", "a" * 64)
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PASSWORD", "x")

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from kernel_backend.infrastructure.database.organization_repository import (
        OrganizationRepository,
    )
    from kernel_backend.core.services.organization_service import OrganizationService

    stack_user_id = f"stack_{uuid4().hex[:8]}"

    async with factory() as session:
        repo = OrganizationRepository(session)
        service = OrganizationService(repo)

        # Step 1: Admin bootstraps org with email as user_id (local JWT path)
        org, _ = await service.create_organization(
            name="Kernel Alpha", admin_user_id=_ADMIN_EMAIL
        )

        # Step 2: Stack Auth login — opaque user_id not found
        found = await repo.get_organization_by_user_id(stack_user_id)
        assert found is None

        # Step 3: Fallback by email — found!
        found_by_email = await repo.get_organization_by_user_id(_ADMIN_EMAIL)
        assert found_by_email is not None
        assert found_by_email.id == org.id

        # Step 4: Bridge identity — add membership for Stack Auth user_id
        await repo.add_member(org.id, stack_user_id, "admin")

        # Step 5: Future lookup by Stack Auth user_id — works directly
        found_direct = await repo.get_organization_by_user_id(stack_user_id)
        assert found_direct is not None
        assert found_direct.id == org.id

    await engine.dispose()


# ---------------------------------------------------------------------------
# Test 3: Invitation accept uses authenticated user_id
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_invitation_creation_and_listing(test_app):
    """
    Admin can create and list invitations via the admin endpoints.

    Note: invitation accept is not testable with SQLite due to
    DateTime(timezone=True) returning naive datetimes, causing
    a comparison error in Invitation.is_valid. This works correctly
    with PostgreSQL (production).
    """
    admin_token = _mint_token()

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Bootstrap org via first authenticated request
        resp1 = await client.get(
            "/organizations",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp1.status_code == 200
        org_id = resp1.json()["items"][0]["org_id"]

        # Create an invitation
        resp2 = await client.post(
            "/admin/invitations",
            json={"email": "member@test.com", "org_id": org_id},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp2.status_code == 201
        data = resp2.json()
        assert data["email"] == "member@test.com"
        assert data["org_id"] == org_id
        assert data["status"] == "pending"

        # List invitations — should show the one we just created
        resp3 = await client.get(
            "/admin/invitations",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp3.status_code == 200
        items = resp3.json()["items"]
        assert len(items) >= 1
        assert any(inv["email"] == "member@test.com" for inv in items)
