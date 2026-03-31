"""Trivial ARQ job used by the /health readiness endpoint to verify the worker is alive."""


async def health_check_job(ctx: dict) -> str:
    """Returns 'ok' immediately. Used by _check_worker() in the health router."""
    return "ok"
