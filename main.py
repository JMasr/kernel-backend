import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Logging — configure once at import time so every module picks it up
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers but keep ours at DEBUG
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("arq").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from kernel_backend.api.auth.router import router as auth_router
from kernel_backend.api.content.router import router as content_router
from kernel_backend.api.users.router import router as users_router
from kernel_backend.api.downloads.router import router as downloads_router
from kernel_backend.api.identity.router import router as identity_router
from kernel_backend.api.invitations.router import admin_router as invitations_admin_router
from kernel_backend.api.invitations.router import public_router as invitations_public_router
from kernel_backend.api.middleware.auth import HybridAuthMiddleware
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
            "Redis unavailable — signing endpoints will return 503. "
            "Set REDIS_HOST / REDIS_PASSWORD in .env to enable. Error: %s",
            exc,
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

    @app.get("/")
    async def health() -> dict:
        return {"status": "ok", "version": "2.0.0"}

    app.include_router(auth_router)
    app.include_router(identity_router, prefix="/identity")
    app.include_router(signing_router)
    app.include_router(verification_router)
    app.include_router(public_verify_router)
    app.include_router(organizations_router)
    app.include_router(invitations_admin_router)
    app.include_router(invitations_public_router)
    app.include_router(content_router)
    app.include_router(downloads_router)
    app.include_router(users_router)

    # ------------------------------------------------------------------
    # Request / response access log — innermost so it sees post-auth state
    # ------------------------------------------------------------------
    from starlette.middleware.base import BaseHTTPMiddleware

    _access_log = logging.getLogger("kernel.access")

    class AccessLogMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            origin = request.headers.get("origin", "-")
            auth_header = request.headers.get("authorization", "")
            token_hint = ""
            if auth_header.startswith("Bearer "):
                raw = auth_header[7:]
                token_hint = f"krnl_..." if raw.startswith("krnl_") else f"{raw[:12]}..."
            _access_log.debug(
                "→ %s %s  origin=%s  token=%s",
                request.method,
                request.url.path,
                origin,
                token_hint or "(none)",
            )
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _access_log.info(
                "← %s %s  status=%d  %.1fms  auth_type=%s  user=%s  org=%s  admin=%s",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                getattr(request.state, "auth_type", "?"),
                getattr(request.state, "user_id", "?"),
                getattr(request.state, "org_id", "?"),
                getattr(request.state, "is_admin", "?"),
            )
            return response

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(HybridAuthMiddleware)

    # CORS — must be outermost middleware so preflight OPTIONS requests are
    # handled before HybridAuthMiddleware rejects them with 401.
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
