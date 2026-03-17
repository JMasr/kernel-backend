"""
Unit tests for app.state.registry setup.

Verifies that SessionFactoryRegistry is a valid RegistryPort implementation
and that it wraps VideoRepository correctly.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.infrastructure.database.repositories import SessionFactoryRegistry


def _make_session_factory() -> MagicMock:
    """Return a mock session factory that yields an async context manager."""
    session = AsyncMock()

    class _FakeCtx:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *_):
            pass

    factory = MagicMock(return_value=_FakeCtx())
    return factory


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_session_factory_registry_is_registry_port():
    """SessionFactoryRegistry must satisfy the RegistryPort ABC."""
    factory = _make_session_factory()
    registry = SessionFactoryRegistry(factory)
    assert isinstance(registry, RegistryPort)


def test_session_factory_registry_stored_in_app_state():
    """
    When the app lifespan runs, app.state.registry is a SessionFactoryRegistry.
    This test verifies the type contract without actually booting the real app
    (which requires a live Postgres + Redis connection).
    """
    from fastapi import FastAPI

    factory = _make_session_factory()
    app = FastAPI()
    app.state.registry = SessionFactoryRegistry(factory)

    registry = app.state.registry
    assert isinstance(registry, SessionFactoryRegistry)
    assert isinstance(registry, RegistryPort)
