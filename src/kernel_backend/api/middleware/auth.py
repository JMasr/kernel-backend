"""
Hybrid authentication middleware.

Supports two authentication methods:
1. API Keys (krnl_...) — programmatic access
2. Stack Auth / Neon Auth JWT session tokens — human users via the frontend
"""
from __future__ import annotations

import hashlib

import httpx
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Paths that do not require any authentication
_PUBLIC_PATHS = {
    ("GET", "/"),
    ("POST", "/organizations"),
    ("GET", "/docs"),
    ("GET", "/openapi.json"),
    ("GET", "/redoc"),
}

# Path *prefixes* that are always public (presigned downloads, invitation accept)
_PUBLIC_PREFIXES = ("/download/", "/invitations/accept/", "/verify/public")


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API-key-only middleware (legacy, preserved for tests that inject this class directly).
    Production code uses HybridAuthMiddleware below.
    """

    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path

        if (method, path) in _PUBLIC_PATHS or any(
            path.startswith(p) for p in _PUBLIC_PREFIXES
        ):
            request.state.org_id = None
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        plaintext_key = auth_header[len("Bearer "):]
        if not plaintext_key.startswith("krnl_"):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key format"},
            )

        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
        try:
            session_factory = request.app.state.db_session_factory
            async with session_factory() as session:
                from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository

                repo = OrganizationRepository(session)
                api_key = await repo.verify_api_key(key_hash)
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"detail": "Authentication service unavailable"},
            )

        if api_key is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or inactive API key"},
            )

        request.state.org_id = api_key.org_id
        request.state.auth_type = "api_key"
        request.state.user_id = None
        request.state.is_admin = False
        return await call_next(request)


class HybridAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware supporting both API Keys and Stack Auth (Neon Auth) JWT tokens.

    Token type detection:
    - Starts with "krnl_" → API Key path (existing flow)
    - Otherwise → Stack Auth session token path (frontend users)

    Sets on request.state:
    - org_id: UUID | None
    - auth_type: "api_key" | "neon_auth"
    - user_id: str | None   (Neon Auth only)
    - is_admin: bool        (True if user_id == ADMIN_USER_ID)
    """

    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path

        if (method, path) in _PUBLIC_PATHS or any(
            path.startswith(p) for p in _PUBLIC_PREFIXES
        ):
            request.state.org_id = None
            request.state.auth_type = "public"
            request.state.user_id = None
            request.state.is_admin = False
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        token = auth_header[len("Bearer "):]

        try:
            if token.startswith("krnl_"):
                await self._handle_api_key(token, request)
            else:
                await self._handle_neon_auth(token, request)
        except _AuthError as exc:
            return JSONResponse(status_code=401, content={"detail": str(exc)})
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"detail": "Authentication service unavailable"},
            )

        return await call_next(request)

    async def _handle_api_key(self, token: str, request: Request) -> None:
        key_hash = hashlib.sha256(token.encode()).hexdigest()
        session_factory = request.app.state.db_session_factory
        async with session_factory() as session:
            from kernel_backend.infrastructure.database.organization_repository import (
                OrganizationRepository,
            )

            repo = OrganizationRepository(session)
            api_key = await repo.verify_api_key(key_hash)

        if api_key is None:
            raise _AuthError("Invalid or inactive API key")

        request.state.org_id = api_key.org_id
        request.state.auth_type = "api_key"
        request.state.user_id = None
        request.state.is_admin = False

    async def _handle_neon_auth(self, token: str, request: Request) -> None:
        """
        Verify a Stack Auth / Neon Auth session token by calling the auth API.

        Expected response: {"user_id": "<uid>", ...}
        """
        from kernel_backend.config import Settings

        settings = Settings()  # reads from env / .env file
        if not settings.NEON_AUTH_API_KEY:
            raise _AuthError("Neon Auth not configured on this server")

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.NEON_AUTH_URL}/api/v1/users/me",
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-stack-project-id": settings.NEON_AUTH_API_KEY,
                },
            )

        if resp.status_code != 200:
            raise _AuthError("Invalid or expired session token")

        session_data = resp.json()
        user_id: str | None = session_data.get("id") or session_data.get("user_id")
        if not user_id:
            raise _AuthError("Session response missing user_id")

        # Look up the user's organisation
        session_factory = request.app.state.db_session_factory
        async with session_factory() as session:
            from kernel_backend.infrastructure.database.organization_repository import (
                OrganizationRepository,
            )

            repo = OrganizationRepository(session)
            org = await repo.get_organization_by_user_id(user_id)

        request.state.org_id = org.id if org else None
        request.state.auth_type = "neon_auth"
        request.state.user_id = user_id
        request.state.is_admin = (
            bool(settings.ADMIN_USER_ID) and user_id == settings.ADMIN_USER_ID
        )


class _AuthError(Exception):
    """Internal sentinel — converted to 401 JSON by the middleware."""
