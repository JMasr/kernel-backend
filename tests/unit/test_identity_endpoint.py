from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from httpx import ASGITransport, AsyncClient

from main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def mock_session():
    """Async context manager mock for AsyncSession."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def app_with_mock_db(app, mock_session):
    # Factory is a regular callable returning an async context manager
    app.state.db_session_factory = MagicMock(return_value=mock_session)
    return app, mock_session


async def test_generate_identity_returns_201(app_with_mock_db) -> None:
    app, mock_session = app_with_mock_db
    with patch(
        "kernel_backend.infrastructure.database.repositories.IdentityRepository.create",
        new_callable=AsyncMock,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/identity/generate",
                json={"name": "Alice", "institution": "ACME Corp"},
            )
    assert response.status_code == 201


async def test_response_contains_private_key_pem(app_with_mock_db) -> None:
    app, _ = app_with_mock_db
    with patch(
        "kernel_backend.infrastructure.database.repositories.IdentityRepository.create",
        new_callable=AsyncMock,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/identity/generate",
                json={"name": "Bob", "institution": "Uni"},
            )
    assert response.status_code == 201
    data = response.json()
    assert "private_key_pem" in data
    assert data["private_key_pem"].startswith("-----BEGIN PRIVATE KEY-----")


async def test_private_key_is_valid_ed25519_pem(app_with_mock_db) -> None:
    app, _ = app_with_mock_db
    with patch(
        "kernel_backend.infrastructure.database.repositories.IdentityRepository.create",
        new_callable=AsyncMock,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/identity/generate",
                json={"name": "Carol", "institution": "Lab"},
            )
    data = response.json()
    pub_key = load_pem_public_key(data["public_key_pem"].encode())
    assert isinstance(pub_key, Ed25519PublicKey)


async def test_empty_name_returns_422(app_with_mock_db) -> None:
    app, _ = app_with_mock_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/identity/generate",
            json={"name": "", "institution": "Uni"},
        )
    assert response.status_code == 422


async def test_empty_institution_returns_422(app_with_mock_db) -> None:
    app, _ = app_with_mock_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/identity/generate",
            json={"name": "Alice", "institution": ""},
        )
    assert response.status_code == 422
