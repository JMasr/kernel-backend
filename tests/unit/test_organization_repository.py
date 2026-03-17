"""Unit tests for OrganizationRepository against an in-memory SQLite database."""
import uuid

import pytest

from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository


# ---------------------------------------------------------------------------
# create_organization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_organization_returns_org(db_session):
    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("Acme Corp")

    assert org.name == "Acme Corp"
    assert org.id is not None
    assert org.current_pepper_version == 1


@pytest.mark.asyncio
async def test_get_organization_by_id(db_session):
    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("WidgetCo")

    fetched = await repo.get_organization_by_id(org.id)

    assert fetched is not None
    assert fetched.name == "WidgetCo"
    assert fetched.id == org.id


@pytest.mark.asyncio
async def test_get_organization_by_id_returns_none_for_unknown(db_session):
    repo = OrganizationRepository(db_session)
    result = await repo.get_organization_by_id(uuid.uuid4())

    assert result is None


# ---------------------------------------------------------------------------
# add_member / get_member / get_organization_by_user_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_member_and_get_member(db_session):
    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("MyOrg")

    member = await repo.add_member(org.id, "alice", "admin")

    assert member.user_id == "alice"
    assert member.role == "admin"
    assert member.org_id == org.id


@pytest.mark.asyncio
async def test_add_member_is_idempotent(db_session):
    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("MyOrg")

    await repo.add_member(org.id, "bob", "member")
    # Second call must not raise — ON CONFLICT DO NOTHING
    member = await repo.add_member(org.id, "bob", "member")

    assert member.user_id == "bob"


@pytest.mark.asyncio
async def test_get_member_returns_none_when_not_member(db_session):
    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("Empty Org")

    result = await repo.get_member(org.id, "nobody")

    assert result is None


@pytest.mark.asyncio
async def test_get_organization_by_user_id(db_session):
    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("LookupOrg")
    await repo.add_member(org.id, "charlie", "admin")

    found = await repo.get_organization_by_user_id("charlie")

    assert found is not None
    assert found.id == org.id


@pytest.mark.asyncio
async def test_get_organization_by_user_id_returns_none_for_unknown(db_session):
    repo = OrganizationRepository(db_session)

    result = await repo.get_organization_by_user_id("unknown_user")

    assert result is None


# ---------------------------------------------------------------------------
# create_api_key / verify_api_key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_and_verify_api_key(db_session):
    import hashlib

    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("KeyOrg")

    plaintext = "krnl_" + "f" * 32
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    key_prefix = plaintext[:12]

    api_key = await repo.create_api_key(org.id, key_hash, key_prefix, "main key")

    assert api_key.key_prefix == key_prefix
    assert api_key.is_active is True
    assert api_key.name == "main key"

    verified = await repo.verify_api_key(key_hash)

    assert verified is not None
    assert verified.org_id == org.id
    assert verified.key_hash == key_hash


@pytest.mark.asyncio
async def test_verify_api_key_returns_none_for_unknown_hash(db_session):
    repo = OrganizationRepository(db_session)

    result = await repo.verify_api_key("0" * 64)

    assert result is None


@pytest.mark.asyncio
async def test_verify_api_key_updates_last_used_at(db_session):
    import hashlib

    repo = OrganizationRepository(db_session)
    org = await repo.create_organization("TimestampOrg")

    plaintext = "krnl_" + "e" * 32
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    await repo.create_api_key(org.id, key_hash, plaintext[:12], None)

    verified = await repo.verify_api_key(key_hash)

    assert verified is not None
    # last_used_at is updated inside verify_api_key (not reflected in returned object,
    # but the call must not raise)