"""
Hybrid authentication middleware.

Supports three authentication methods:
1. API Keys (krnl_...) — programmatic access
2. Local admin JWT (HS256, signed with JWT_SECRET) — admin login via POST /auth/login
3. Stack Auth / Neon Auth JWT session tokens — human users via the frontend
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import httpx
import jwt
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

_log = logging.getLogger("kernel.auth")

# Paths that do not require any authentication
_PUBLIC_PATHS = {
    ("GET", "/"),
    ("GET", "/docs"),
    ("GET", "/openapi.json"),
    ("GET", "/redoc"),
    ("POST", "/auth/login"),
    ("POST", "/auth/refresh"),
}

# Path *prefixes* that are always public (presigned downloads, invitation accept)
_PUBLIC_PREFIXES = ("/download/", "/invitations/accept/", "/verify/public")

# ---------------------------------------------------------------------------
# In-memory TTL cache for Stack Auth verified tokens
# ---------------------------------------------------------------------------
_CACHE_TTL = 90  # seconds
_CACHE_MAX_SIZE = 2000


@dataclass(frozen=True, slots=True)
class _CachedAuth:
    user_id: str
    email: str
    is_admin: bool
    org_id: str | None
    expires_at: float  # time.monotonic()


_stack_auth_cache: dict[str, _CachedAuth] = {}


def _cache_get(token_hash: str) -> _CachedAuth | None:
    entry = _stack_auth_cache.get(token_hash)
    if entry is None:
        return None
    if time.monotonic() > entry.expires_at:
        _stack_auth_cache.pop(token_hash, None)
        return None
    return entry


def _cache_put(token_hash: str, entry: _CachedAuth) -> None:
    # Simple size-bound eviction: drop oldest entry when over limit
    if len(_stack_auth_cache) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_stack_auth_cache))
        _stack_auth_cache.pop(oldest_key, None)
    _stack_auth_cache[token_hash] = entry


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
    Middleware supporting API Keys, local admin JWTs, and Stack Auth tokens.

    Token type detection (in order):
    1. Starts with "krnl_" → API Key path
    2. JWT header alg == "HS256" and JWT_SECRET set → local admin path
    3. Otherwise → Stack Auth session token path (frontend users)

    Sets on request.state:
    - org_id: UUID | None
    - auth_type: "api_key" | "local_jwt" | "neon_auth"
    - user_id: str | None
    - email: str | None
    - is_admin: bool
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
        token_preview = f"{token[:12]}..." if len(token) > 12 else token

        try:
            if token.startswith("krnl_"):
                _log.debug("auth: API key path  token=%s", token_preview)
                await self._handle_api_key(token, request)
            else:
                # Smart routing: inspect JWT header to decide path
                from kernel_backend.config import get_settings
                settings = get_settings()

                token_alg = self._get_token_alg(token)

                if token_alg == "HS256" and settings.JWT_SECRET:
                    _log.debug("auth: local JWT path (HS256)  token=%s", token_preview)
                    await self._handle_local_jwt(token, request, settings)
                else:
                    _log.debug("auth: Stack Auth path (alg=%s)  token=%s",
                               token_alg or "unknown", token_preview)
                    await self._handle_neon_auth(token, request)
        except _AuthError as exc:
            _log.warning("auth: rejected  path=%s  reason=%s", request.url.path, exc)
            return JSONResponse(status_code=401, content={"detail": str(exc)})
        except Exception as exc:
            _log.exception("auth: unexpected error  path=%s  error=%s", request.url.path, exc)
            return JSONResponse(
                status_code=500,
                content={"detail": "Authentication service unavailable"},
            )

        return await call_next(request)

    @staticmethod
    def _get_token_alg(token: str) -> str | None:
        """Extract the `alg` claim from a JWT header without verification (zero-cost base64 parse)."""
        try:
            header = jwt.get_unverified_header(token)
            return header.get("alg")
        except jwt.exceptions.DecodeError:
            return None

    async def _handle_local_jwt(self, token: str, request: Request, settings) -> None:
        """
        Decode token as a local HS256 admin JWT (issued by POST /auth/login).

        Since we already confirmed alg==HS256 via header inspection, decode failure
        means the token is invalid (not a different type), so we raise _AuthError.
        """
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise _AuthError("Local admin token has expired — please log in again")
        except jwt.InvalidTokenError as exc:
            raise _AuthError(f"Invalid local admin token: {exc}")

        email: str = payload.get("email") or payload.get("sub") or ""
        if not email:
            raise _AuthError("Local admin token missing email claim")

        is_admin = bool(payload.get("is_admin", False))
        _log.debug("local JWT decoded  email=%s  is_admin=%s", email, is_admin)

        # Lazy bootstrap: create "Kernel Alpha" org on first admin login
        session_factory = request.app.state.db_session_factory
        async with session_factory() as session:
            from kernel_backend.infrastructure.database.organization_repository import (
                OrganizationRepository,
            )

            repo = OrganizationRepository(session)
            org = await repo.get_organization_by_user_id(email)
            _log.debug("local JWT org lookup  user=%s  org=%s", email, org.id if org else None)

            if org is None and is_admin:
                _log.info("local JWT bootstrap: creating 'Kernel Alpha' org for admin %s", email)
                from kernel_backend.core.services.organization_service import OrganizationService

                try:
                    service = OrganizationService(repo)
                    org, _ = await service.create_organization(
                        name="Kernel Alpha", admin_user_id=email
                    )
                    _log.info("local JWT bootstrap complete  org_id=%s", org.id)
                except Exception:
                    # Race condition: another request created it first — retry lookup
                    _log.info("local JWT bootstrap conflict, retrying lookup for %s", email)
                    org = await repo.get_organization_by_user_id(email)

        request.state.org_id = org.id if org else None
        request.state.auth_type = "local_jwt"
        request.state.user_id = email
        request.state.email = email
        request.state.is_admin = is_admin

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
        request.state.email = None
        request.state.is_admin = False

    async def _handle_neon_auth(self, token: str, request: Request) -> None:
        """
        Verify a Stack Auth / Neon Auth session token.

        Uses an in-memory TTL cache to avoid repeated HTTP calls for the same token,
        and a shared httpx.AsyncClient (from app.state) for connection pooling.
        """
        # --- Cache check ---
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        cached = _cache_get(token_hash)
        if cached is not None:
            _log.debug("Stack Auth: cache hit  user_id=%s", cached.user_id)
            request.state.org_id = cached.org_id
            request.state.auth_type = "neon_auth"
            request.state.user_id = cached.user_id
            request.state.email = cached.email
            request.state.is_admin = cached.is_admin
            return

        # --- Settings ---
        from kernel_backend.config import get_settings
        settings = get_settings()

        if not settings.NEON_AUTH_API_KEY:
            _log.error(
                "Stack Auth path requested but NEON_AUTH_API_KEY is not set in .env. "
                "Set it to your Stack Auth project ID."
            )
            raise _AuthError("Neon Auth not configured on this server")

        if not settings.NEON_AUTH_SECRET_SERVER_KEY:
            _log.error("NEON_AUTH_SECRET_SERVER_KEY is not set in .env")
            raise _AuthError("Neon Auth not configured on this server")

        # --- HTTP verification (shared client or ephemeral fallback) ---
        verify_url = f"{settings.NEON_AUTH_URL}/api/v1/users/me"

        client: httpx.AsyncClient | None = getattr(request.app.state, "httpx_client", None)
        owns_client = False
        if client is None:
            # Fallback for tests or when lifespan hasn't created the shared client
            client = httpx.AsyncClient(timeout=5.0)
            owns_client = True

        try:
            t0 = time.perf_counter()
            resp = await client.get(
                verify_url,
                headers={
                    "x-stack-access-token": token,
                    "x-stack-project-id": settings.NEON_AUTH_API_KEY,
                    "x-stack-secret-server-key": settings.NEON_AUTH_SECRET_SERVER_KEY,
                    "x-stack-access-type": "server",
                },
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _log.debug("Stack Auth: HTTP %d in %.0fms", resp.status_code, elapsed_ms)
        finally:
            if owns_client:
                await client.aclose()

        if resp.status_code != 200:
            _log.warning("Stack Auth: token rejected  status=%d  body=%s",
                         resp.status_code, resp.text[:500])
            raise _AuthError("Invalid or expired session token")

        session_data = resp.json()
        user_id: str | None = session_data.get("id") or session_data.get("user_id")
        if not user_id:
            _log.error("Stack Auth: response missing user id field  keys=%s", list(session_data.keys()))
            raise _AuthError("Session response missing user_id")

        primary_email: str = session_data.get("primary_email") or ""
        is_admin = bool(settings.ADMIN_EMAIL) and primary_email == settings.ADMIN_EMAIL
        _log.debug(
            "Stack Auth: user resolved  user_id=%s  email=%s  is_admin=%s",
            user_id, primary_email, is_admin,
        )
        request.state.email = primary_email

        # Look up the user's organisation
        session_factory = request.app.state.db_session_factory
        async with session_factory() as session:
            from kernel_backend.infrastructure.database.organization_repository import (
                OrganizationRepository,
            )

            repo = OrganizationRepository(session)
            org = await repo.get_organization_by_user_id(user_id)
            _log.debug("Stack Auth: org lookup  user_id=%s  org=%s", user_id, org.id if org else None)

            # Fallback: Stack Auth user_id is opaque — try email if org not found
            if org is None and primary_email:
                _log.debug("Stack Auth: fallback lookup by email %s", primary_email)
                org = await repo.get_organization_by_user_id(primary_email)
                if org is not None:
                    # Bridge the identity: add membership for Stack Auth user_id
                    _log.info(
                        "Stack Auth: bridging identity  stack_uid=%s  email=%s  org=%s",
                        user_id, primary_email, org.id,
                    )
                    await repo.add_member(org.id, user_id, "admin" if is_admin else "member")

            # Lazy bootstrap: create "Kernel Alpha" org on first admin login
            if org is None and is_admin:
                _log.info("Stack Auth bootstrap: creating 'Kernel Alpha' org for admin %s", user_id)
                from kernel_backend.core.services.organization_service import OrganizationService

                try:
                    service = OrganizationService(repo)
                    org, _ = await service.create_organization(
                        name="Kernel Alpha", admin_user_id=user_id
                    )
                    _log.info("Stack Auth bootstrap complete  org_id=%s", org.id)
                except Exception:
                    # Race condition: another request created it first — retry lookup
                    _log.info("Stack Auth bootstrap conflict, retrying lookup for %s", user_id)
                    org = await repo.get_organization_by_user_id(user_id)

        org_id = org.id if org else None
        request.state.org_id = org_id
        request.state.auth_type = "neon_auth"
        request.state.user_id = user_id
        request.state.is_admin = is_admin

        # --- Cache populate (skip bootstrap case — org will be created next request) ---
        if not (org_id is None and is_admin):
            _cache_put(
                token_hash,
                _CachedAuth(
                    user_id=user_id,
                    email=primary_email,
                    is_admin=is_admin,
                    org_id=org_id,
                    expires_at=time.monotonic() + _CACHE_TTL,
                ),
            )


class _AuthError(Exception):
    """Internal sentinel — converted to 401 JSON by the middleware."""
