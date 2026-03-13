from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from kernel_backend.api.identity.router import router as identity_router
from kernel_backend.api.signing.router import router as signing_router
from kernel_backend.api.verification.router import router as verification_router
from kernel_backend.config import Settings
from kernel_backend.infrastructure.database.session import make_engine, make_session_factory
from kernel_backend.infrastructure.queue.redis_pool import make_redis_settings
from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()

    # Storage
    app.state.storage = LocalStorageAdapter(base_path=settings.STORAGE_LOCAL_BASE_PATH)

    # DB engine
    engine = make_engine(settings.DATABASE_URL)
    app.state.db_engine = engine
    app.state.db_session_factory = make_session_factory(engine)

    # ARQ Redis pool
    from arq import create_pool

    redis_pool = await create_pool(make_redis_settings(settings))
    app.state.redis_pool = redis_pool

    yield

    # Shutdown
    await engine.dispose()
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

    app.include_router(identity_router, prefix="/identity")
    app.include_router(signing_router)
    app.include_router(verification_router)

    return app


app = create_app()
