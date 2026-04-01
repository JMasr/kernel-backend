"""Health endpoints.

- GET /health/live  — liveness probe: only checks that the process is alive.
                     No external calls; must respond in <10 ms.
                     Use this as the Docker HEALTHCHECK command.

- GET /health       — readiness probe: checks DB, Redis, disk, and worker.
                     May take up to 10 s (worker check enqueues a trivial job).
                     Use this in deploy scripts to validate a successful deploy.
                     Do NOT use as a liveness probe.
"""
from __future__ import annotations

import asyncio
import shutil
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter()


@router.get("/health/live")
async def liveness() -> dict:
    """Process-level liveness. Never calls external services."""
    return {"status": "alive", "timestamp": time.time()}


@router.get("/health")
async def readiness(request: Request) -> JSONResponse:
    """Checks all external dependencies. Returns 200 if healthy, 503 otherwise."""
    db_session_factory = request.app.state.db_session_factory
    redis_pool = getattr(request.app.state, "redis_pool", None)

    # Disk check omitted intentionally for MVP:
    # backend containers are stateless (no mounted volumes),
    # so shutil.disk_usage reads the container overlay FS, not the real host disk.
    # Reinstate when a /tmp volume is mounted in docker-compose.yml.
    checks = await asyncio.gather(
        _check_database(db_session_factory),
        _check_valkey(redis_pool),
        _check_worker(redis_pool),
        return_exceptions=True,
    )

    labels = ["database", "valkey", "worker"]
    results: dict[str, dict] = {}
    for label, result in zip(labels, checks):
        if isinstance(result, Exception):
            results[label] = {"status": "unhealthy", "error": str(result)}
        else:
            results[label] = result  # type: ignore[assignment]

    is_healthy = all(r.get("status") == "healthy" for r in results.values())

    return JSONResponse(
        status_code=200 if is_healthy else 503,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "checks": results,
            "timestamp": time.time(),
        },
    )


async def _check_database(session_factory: object) -> dict:
    """SELECT 1 against Neon. 10 s timeout accommodates free-tier cold starts (5–8 s)."""
    try:
        async with asyncio.timeout(10.0):
            from sqlalchemy.ext.asyncio import async_sessionmaker

            async with session_factory() as session:  # type: ignore[operator]
                await session.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except asyncio.TimeoutError:
        return {"status": "unhealthy", "error": "timeout after 10s (possible cold start)"}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


async def _check_valkey(redis_pool: object) -> dict:
    """PING against Redis/Valkey. 3 s timeout — it's a local Docker service."""
    if redis_pool is None:
        return {"status": "unhealthy", "error": "redis pool not initialised"}
    try:
        async with asyncio.timeout(3.0):
            await redis_pool.ping()  # type: ignore[union-attr]
        return {"status": "healthy"}
    except asyncio.TimeoutError:
        return {"status": "unhealthy", "error": "timeout after 3s"}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


async def _check_disk() -> dict:
    """Disk usage on the temp volume.

    In production, TEMP_DIR must be a host-mounted volume so this reads the real
    disk and not the container overlay FS. Configure in docker-compose.yml:

        volumes:
          - /var/lib/kernel/temp:/tmp
        environment:
          TEMP_DIR: /tmp
    """
    from kernel_backend.config import get_settings

    settings = get_settings()
    data_path: str = getattr(settings, "TEMP_DIR", "/tmp") or "/tmp"

    try:
        stat = shutil.disk_usage(data_path)
        percent_used = (stat.used / stat.total) * 100
        free_gb = stat.free / (1024**3)

        if percent_used > 90:
            return {
                "status": "unhealthy",
                "path": data_path,
                "percent_used": round(percent_used, 1),
                "free_gb": round(free_gb, 1),
                "message": "disk almost full — cleanup required",
            }
        if percent_used > 80:
            return {
                "status": "degraded",
                "path": data_path,
                "percent_used": round(percent_used, 1),
                "free_gb": round(free_gb, 1),
                "message": "disk usage high — monitor closely",
            }
        return {
            "status": "healthy",
            "path": data_path,
            "percent_used": round(percent_used, 1),
            "free_gb": round(free_gb, 1),
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


async def _check_worker(redis_pool: object) -> dict:
    """Enqueue a trivial job and wait for the result (up to 10 s).

    Detects the scenario where DB and Redis are healthy but the ARQ worker has
    crashed — jobs would queue indefinitely without this check.

    This check can add up to 10 s of latency to GET /health. Never use /health
    as a Docker liveness probe; use /health/live instead.
    """
    if redis_pool is None:
        return {"status": "unhealthy", "error": "redis pool not initialised"}
    try:
        async with asyncio.timeout(10.0):
            job = await redis_pool.enqueue_job(  # type: ignore[union-attr]
                "health_check_job",
                _job_id=f"health-{time.time()}",
            )
            result = await job.result(timeout=8.0)

        if result == "ok":
            return {"status": "healthy"}
        return {"status": "unhealthy", "error": f"unexpected result: {result}"}

    except asyncio.TimeoutError:
        return {"status": "unhealthy", "error": "worker not responding after 10s"}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}
