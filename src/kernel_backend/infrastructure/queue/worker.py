"""ARQ WorkerSettings — registers jobs, wires Redis, and populates ctx on startup."""
from __future__ import annotations

import concurrent.futures

from kernel_backend.config import Settings
from kernel_backend.infrastructure.database.repositories import SessionFactoryRegistry
from kernel_backend.infrastructure.database.session import make_engine, make_session_factory
from kernel_backend.infrastructure.queue.health_job import health_check_job
from kernel_backend.infrastructure.queue.jobs import process_sign_job
from kernel_backend.infrastructure.queue.redis_pool import make_redis_settings
from kernel_backend.infrastructure.storage import make_storage


async def on_startup(ctx: dict) -> None:
    """Populate ctx with the dependencies that job functions expect."""
    settings = Settings()

    engine = make_engine(settings.DATABASE_URL)
    session_factory = make_session_factory(engine)

    ctx["storage"] = make_storage(settings)
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
    functions = [process_sign_job, health_check_job]
    redis_settings = make_redis_settings(_settings)

    # Maximum time a single job may run before ARQ cancels it.
    # Must be >= the worst-case signing job duration (300 s for large video).
    job_timeout = 360  # 6 minutes

    # How long ARQ waits for in-flight jobs to finish after receiving SIGTERM.
    # Must be <= stop_grace_period in docker-compose.yml (set that to 380 s).
    graceful_shutdown_timeout = 360  # 6 minutes

    # Concurrent jobs per worker process. CPU-intensive jobs; leave headroom for the OS.
    max_jobs = 4

    # How long to keep job results in Redis.
    keep_result = 3600  # 1 hour

    on_startup = on_startup
    on_shutdown = on_shutdown
