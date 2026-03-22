"""ARQ WorkerSettings — registers jobs, wires Redis, and populates ctx on startup."""
from __future__ import annotations

import concurrent.futures

from kernel_backend.config import Settings
from kernel_backend.infrastructure.database.repositories import SessionFactoryRegistry
from kernel_backend.infrastructure.database.session import make_engine, make_session_factory
from kernel_backend.infrastructure.queue.jobs import process_sign_job
from kernel_backend.infrastructure.queue.redis_pool import make_redis_settings
from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter


async def on_startup(ctx: dict) -> None:
    """Populate ctx with the dependencies that job functions expect."""
    settings = Settings()

    engine = make_engine(settings.DATABASE_URL)
    session_factory = make_session_factory(engine)

    ctx["storage"] = LocalStorageAdapter(
        base_path=settings.STORAGE_LOCAL_BASE_PATH,
        secret_key=settings.STORAGE_HMAC_SECRET,
    )
    ctx["registry"] = SessionFactoryRegistry(session_factory)
    ctx["pepper"] = settings.system_pepper_bytes
    ctx["process_pool"] = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    ctx["_engine"] = engine


async def on_shutdown(ctx: dict) -> None:
    if pool := ctx.get("process_pool"):
        pool.shutdown(wait=False)
    if engine := ctx.get("_engine"):
        await engine.dispose()


_settings = Settings()


class WorkerSettings:
    functions = [process_sign_job]
    redis_settings = make_redis_settings(_settings)
    job_timeout: int = 180
    max_jobs: int = 4
    on_startup = on_startup
    on_shutdown = on_shutdown
