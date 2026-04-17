"""Request correlation middleware.

Stamps every request with a ``request_id`` that:

* is read from the ``X-Request-Id`` header if the client supplied one and it
  passes a cheap shape check (UUID hex or ``[A-Za-z0-9_-]{8,128}``); otherwise
  a fresh ``uuid4().hex`` is generated;
* is bound into the ambient structlog context so every subsequent log line in
  the request (auth middleware, routers, job enqueue) carries it;
* is echoed back as the ``X-Request-Id`` response header so the frontend can
  surface it to the user on failure.

``X-Trace-Id`` is handled the same way but is optional — it represents a
user-level flow (e.g. one upload) that may span multiple HTTP hops.
"""
from __future__ import annotations

import re
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from kernel_backend.infrastructure.logging.setup import (
    bind_request_context,
    clear_request_context,
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _accept_or_mint(candidate: str | None) -> str:
    if candidate and _ID_PATTERN.match(candidate):
        return candidate
    return uuid.uuid4().hex


def _accept_optional(candidate: str | None) -> str | None:
    if candidate and _ID_PATTERN.match(candidate):
        return candidate
    return None


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    return request.client.host if request.client else None


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = _accept_or_mint(request.headers.get("x-request-id"))
        trace_id = _accept_optional(request.headers.get("x-trace-id"))
        peer_ip = _client_ip(request)

        request.state.request_id = request_id
        request.state.trace_id = trace_id

        bind_request_context(
            request_id=request_id,
            trace_id=trace_id,
            peer_ip=peer_ip,
        )
        try:
            response = await call_next(request)
        finally:
            clear_request_context()

        response.headers["X-Request-Id"] = request_id
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response
