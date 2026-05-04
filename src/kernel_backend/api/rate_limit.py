"""Per-API-key rate limiter (Redis-backed in production, in-memory fallback)."""
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _rate_limit_key(request: Request) -> str:
    """Use api_key_id as bucket for API key auth; fall back to client IP."""
    api_key_id = getattr(request.state, "api_key_id", None)
    if api_key_id is not None:
        return f"api_key:{api_key_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
