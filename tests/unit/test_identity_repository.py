from datetime import datetime, timezone

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.infrastructure.database.repositories import IdentityRepository


async def test_create_and_retrieve(db_session):
    repo = IdentityRepository(db_session)
    cert = Certificate(
        author_id="abcd1234abcd1234",
        name="Test Author",
        institution="Test Org",
        public_key_pem="-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    await repo.create(cert)
    await db_session.commit()

    result = await repo.get_by_author_id("abcd1234abcd1234")
    assert result is not None
    assert result.name == "Test Author"
    assert result.institution == "Test Org"


async def test_create_is_idempotent(db_session):
    repo = IdentityRepository(db_session)
    cert = Certificate(
        author_id="abcd1234abcd1234",
        name="Test Author",
        institution="Test Org",
        public_key_pem="-----BEGIN PUBLIC KEY-----\nfake\n-----END PUBLIC KEY-----",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    await repo.create(cert)
    await db_session.commit()
    # Second call must not raise — ON CONFLICT DO NOTHING
    await repo.create(cert)
    await db_session.commit()


async def test_get_nonexistent_returns_none(db_session):
    repo = IdentityRepository(db_session)
    result = await repo.get_by_author_id("nonexistent000000")
    assert result is None
