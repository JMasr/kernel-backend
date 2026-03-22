"""Unit tests for POST /auth/refresh (Phase 10.A)."""
from __future__ import annotations

import time

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kernel_backend.api.auth.router import router

_JWT_SECRET = "test-secret-for-unit-tests-only"
_ADMIN_EMAIL = "admin@kernel.test"
_TTL = 60 * 60 * 8  # 8 hours


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _mint_token(
    email: str = _ADMIN_EMAIL,
    is_admin: bool = True,
    secret: str = _JWT_SECRET,
    ttl: int = _TTL,
    expired: bool = False,
) -> str:
    now = int(time.time())
    payload = {
        "sub": email,
        "email": email,
        "is_admin": is_admin,
        "iat": now - (ttl + 10 if expired else 0),
        "exp": now - 10 if expired else now + ttl,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


class TestAuthRefresh:
    def test_refresh_valid_token_returns_200(self, monkeypatch) -> None:
        monkeypatch.setenv("JWT_SECRET", _JWT_SECRET)
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("MIGRATION_DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("KERNEL_SYSTEM_PEPPER", "a" * 64)
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PASSWORD", "x")

        token = _mint_token()
        app = _make_app()
        with TestClient(app) as client:
            response = client.post("/auth/refresh", json={"access_token": token})

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Verify the new token is valid and carries the same claims
        new_payload = jwt.decode(data["access_token"], _JWT_SECRET, algorithms=["HS256"])
        assert new_payload["email"] == _ADMIN_EMAIL
        assert new_payload["is_admin"] is True

    def test_refresh_returns_fresh_expiry(self, monkeypatch) -> None:
        monkeypatch.setenv("JWT_SECRET", _JWT_SECRET)
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("MIGRATION_DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("KERNEL_SYSTEM_PEPPER", "a" * 64)
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PASSWORD", "x")

        token = _mint_token()
        app = _make_app()
        with TestClient(app) as client:
            response = client.post("/auth/refresh", json={"access_token": token})

        new_payload = jwt.decode(response.json()["access_token"], _JWT_SECRET, algorithms=["HS256"])
        # New expiry should be ~8h from now
        assert new_payload["exp"] - new_payload["iat"] == _TTL

    def test_refresh_expired_token_returns_401(self, monkeypatch) -> None:
        monkeypatch.setenv("JWT_SECRET", _JWT_SECRET)
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("MIGRATION_DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("KERNEL_SYSTEM_PEPPER", "a" * 64)
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PASSWORD", "x")

        token = _mint_token(expired=True)
        app = _make_app()
        with TestClient(app) as client:
            response = client.post("/auth/refresh", json={"access_token": token})

        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()

    def test_refresh_tampered_token_returns_401(self, monkeypatch) -> None:
        monkeypatch.setenv("JWT_SECRET", _JWT_SECRET)
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("MIGRATION_DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("KERNEL_SYSTEM_PEPPER", "a" * 64)
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PASSWORD", "x")

        # Sign with a different secret
        token = _mint_token(secret="wrong-secret")
        app = _make_app()
        with TestClient(app) as client:
            response = client.post("/auth/refresh", json={"access_token": token})

        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    def test_refresh_no_jwt_secret_returns_503(self, monkeypatch) -> None:
        monkeypatch.setenv("JWT_SECRET", "")
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("MIGRATION_DATABASE_URL", "sqlite+aiosqlite://")
        monkeypatch.setenv("KERNEL_SYSTEM_PEPPER", "a" * 64)
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PASSWORD", "x")

        app = _make_app()
        with TestClient(app) as client:
            response = client.post("/auth/refresh", json={"access_token": "anything"})

        assert response.status_code == 503
