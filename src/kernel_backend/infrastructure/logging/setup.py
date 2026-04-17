"""Single entry point for application-wide logging configuration.

Called from:

* ``main.py`` lifespan (FastAPI process)
* ``worker.py`` ``on_startup`` (ARQ worker process)
* ``_sign_sync`` child (ProcessPoolExecutor — ``contextvars`` do not cross
  the fork/spawn boundary, and neither does the stdlib ``logging`` config
  when the pool uses ``spawn``)

Safe to call more than once — it clears any existing root handlers before
reinstalling a fresh one.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import structlog
from structlog.types import Processor

from kernel_backend.config import Settings
from kernel_backend.infrastructure.logging.redact import (
    RedactSecretsProcessor,
    scrub_mapping,
    scrub_string,
)

# Third-party loggers that would otherwise drown out our own output at DEBUG.
_THIRD_PARTY_LEVELS: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "sqlalchemy.engine": logging.WARNING,
    "sqlalchemy.pool": logging.WARNING,
    "arq": logging.INFO,
    # Uvicorn's access logger duplicates our AccessLogMiddleware output.
    "uvicorn.access": logging.WARNING,
}

# Fields that Sentry tends to capture from request/body; same denylist applies.
_SENTRY_STRIP_FIELDS = ("request", "extra", "contexts", "breadcrumbs", "exception")


def configure_logging(settings: Settings) -> None:
    """Install the structlog + stdlib bridge. Idempotent."""
    log_level = _resolve_level(settings.LOG_LEVEL)
    renderer = _build_renderer(settings)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # Pulls ``extra={...}`` kwargs from stdlib ``Logger.info(..., extra=...)``
        # calls into the event dict so legacy stdlib callers can emit structured
        # fields without switching to ``structlog.get_logger``.
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        RedactSecretsProcessor(),
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(log_level)

    for name, level in _THIRD_PARTY_LEVELS.items():
        logging.getLogger(name).setLevel(level)

    _maybe_init_sentry(settings)


def _resolve_level(raw: str) -> int:
    level = logging.getLevelName(raw.upper() if raw else "INFO")
    return level if isinstance(level, int) else logging.INFO


def _build_renderer(settings: Settings) -> Processor:
    fmt = (getattr(settings, "LOG_FORMAT", None) or "json").lower()
    if fmt == "console":
        return structlog.dev.ConsoleRenderer(colors=True)
    return structlog.processors.JSONRenderer()


def _maybe_init_sentry(settings: Settings) -> None:
    dsn = (settings.SENTRY_DSN or "").strip()
    # Guard against empty values and against ``.env`` inline comments that
    # leak into the value (e.g. ``SENTRY_DSN=    # disabled``).
    if not dsn or dsn.startswith("#") or "://" not in dsn:
        return

    # Import lazily so installs without Sentry don't need the dependency loaded.
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

    release = os.getenv("BACKEND_TAG") or os.getenv("GIT_SHA") or "dev"

    sentry_sdk.init(
        dsn=dsn,
        environment=settings.ENV,
        release=release,
        traces_sample_rate=0.0,
        send_default_pii=False,
        before_send=_scrub_sentry_event,
        integrations=[
            FastApiIntegration(),
            AsyncioIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            SqlalchemyIntegration(),
        ],
    )


def _scrub_sentry_event(event: Any, _hint: Any) -> Any:
    for field in _SENTRY_STRIP_FIELDS:
        if field in event and isinstance(event[field], dict):
            event[field] = scrub_mapping(event[field])
    if isinstance(event.get("message"), str):
        event["message"] = scrub_string(event["message"])
    return event


def bind_request_context(**values: Any) -> None:
    """Merge keys into the ambient structlog context for the current task."""
    structlog.contextvars.bind_contextvars(**values)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.stdlib.get_logger(name)
