import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from kernel_backend.infrastructure.logging import configure_logging, get_logger

logger = get_logger(__name__)

from kernel_backend.api.auth.router import router as auth_router
from kernel_backend.api.health.router import router as health_router
from kernel_backend.api.content.router import router as content_router
from kernel_backend.api.users.router import router as users_router
from kernel_backend.api.downloads.router import router as downloads_router
from kernel_backend.api.identity.router import router as identity_router
from kernel_backend.api.admin.router import router as admin_users_router
from kernel_backend.api.invitations.router import admin_router as invitations_admin_router
from kernel_backend.api.invitations.router import public_router as invitations_public_router
from kernel_backend.api.leads.router import admin_router as leads_admin_router
from kernel_backend.api.leads.router import public_router as leads_public_router
from kernel_backend.api.middleware.auth import HybridAuthMiddleware
from kernel_backend.api.middleware.request_id import RequestIdMiddleware
from kernel_backend.api.organizations.router import router as organizations_router
from kernel_backend.api.public.router import router as public_verify_router
from kernel_backend.api.signing.router import router as signing_router
from kernel_backend.api.verification.router import router as verification_router
from kernel_backend.config import Settings, get_settings
from kernel_backend.infrastructure.database.repositories import SessionFactoryRegistry
from kernel_backend.infrastructure.database.session import make_engine, make_session_factory
from kernel_backend.infrastructure.queue.redis_pool import make_redis_settings
from kernel_backend.infrastructure.storage import make_storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Logging — structlog bootstrap, redaction, optional Sentry.
    configure_logging(settings)

    # Storage
    app.state.storage = make_storage(settings)

    # DB engine
    engine = make_engine(settings.DATABASE_URL)
    app.state.db_engine = engine
    app.state.db_session_factory = make_session_factory(engine)

    # Shared httpx client for Stack Auth verification (connection pooling)
    http_client = httpx.AsyncClient(
        timeout=5.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    app.state.httpx_client = http_client

    # ARQ Redis pool (optional — signing endpoints disabled if unavailable)
    from arq import create_pool

    redis_pool = None
    try:
        redis_pool = await create_pool(make_redis_settings(settings))
        app.state.redis_pool = redis_pool
    except Exception as exc:
        logger.warning(
            "redis.unavailable",
            hint="signing endpoints will return 503 — set REDIS_HOST / REDIS_PASSWORD in .env",
            error=str(exc),
        )
        app.state.redis_pool = None

    # Registry for verification endpoints (session-per-call wrapper)
    app.state.registry = SessionFactoryRegistry(app.state.db_session_factory)

    yield

    # Shutdown
    await http_client.aclose()
    await engine.dispose()
    if redis_pool is not None:
        await redis_pool.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kernel Security Backend",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.include_router(health_router, tags=["health"])
    app.include_router(auth_router)
    app.include_router(identity_router, prefix="/identity")
    app.include_router(signing_router)
    app.include_router(verification_router)
    app.include_router(public_verify_router)
    app.include_router(organizations_router)
    app.include_router(admin_users_router)
    app.include_router(invitations_admin_router)
    app.include_router(invitations_public_router)
    app.include_router(leads_public_router)
    app.include_router(leads_admin_router)
    app.include_router(content_router)
    app.include_router(downloads_router)
    app.include_router(users_router)

    # ------------------------------------------------------------------
    # Request / response access log — innermost so it sees post-auth state
    # ------------------------------------------------------------------
    from starlette.middleware.base import BaseHTTPMiddleware

    _access_log = get_logger("kernel.access")

    class AccessLogMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                raw = auth_header[7:]
                token_scheme = "krnl_" if raw.startswith("krnl_") else "bearer"
            else:
                token_scheme = "none"

            _access_log.debug(
                "request.started",
                method=request.method,
                path=request.url.path,
                origin=request.headers.get("origin"),
                token_scheme=token_scheme,
            )
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _access_log.info(
                "request.finished",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                elapsed_ms=round(elapsed_ms, 1),
                auth_type=getattr(request.state, "auth_type", None),
                user_id=getattr(request.state, "user_id", None),
                org_id=str(getattr(request.state, "org_id", None) or "") or None,
                is_admin=getattr(request.state, "is_admin", None),
            )
            return response

    # Order: request is processed outside-in. CORS must run first so preflight
    # OPTIONS bypasses auth; RequestId binds correlation before auth logs;
    # AccessLog is innermost so it sees post-auth request.state.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(HybridAuthMiddleware)
    app.add_middleware(RequestIdMiddleware)

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()
