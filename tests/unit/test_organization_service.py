"""Unit tests for OrganizationService using a mock OrganizationPort."""
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from kernel_backend.core.domain.organization import APIKey, Organization, OrganizationMember
from kernel_backend.core.services.organization_service import OrganizationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_org(name: str = "Test Org") -> Organization:
    return Organization(
        id=uuid.uuid4(),
        name=name,
        pepper_v1=None,
        current_pepper_version=1,
        created_at=datetime.now(timezone.utc),
    )


def _make_member(org_id: UUID, user_id: str = "user_abc", role: str = "admin") -> OrganizationMember:
    return OrganizationMember(
        id=uuid.uuid4(),
        org_id=org_id,
        user_id=user_id,
        role=role,
        created_at=datetime.now(timezone.utc),
    )


def _make_api_key(org_id: UUID, key_hash: str = "deadbeef" * 8, prefix: str = "krnl_abc123") -> APIKey:
    return APIKey(
        id=uuid.uuid4(),
        org_id=org_id,
        key_hash=key_hash,
        key_prefix=prefix,
        name="test",
        created_at=datetime.now(timezone.utc),
        last_used_at=None,
        is_active=True,
    )


def _mock_repo(**overrides) -> MagicMock:
    repo = MagicMock()
    repo.create_organization = AsyncMock()
    repo.get_organization_by_id = AsyncMock()
    repo.get_organization_by_user_id = AsyncMock()
    repo.create_api_key = AsyncMock()
    repo.verify_api_key = AsyncMock()
    repo.add_member = AsyncMock()
    repo.get_member = AsyncMock()
    for k, v in overrides.items():
        setattr(repo, k, v)
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_organization_returns_org_and_admin_member():
    org = _make_org()
    member = _make_member(org.id)
    repo = _mock_repo(
        create_organization=AsyncMock(return_value=org),
        add_member=AsyncMock(return_value=member),
    )
    service = OrganizationService(repo)

    result_org, result_member = await service.create_organization("Test Org", "user_abc")

    repo.create_organization.assert_awaited_once_with("Test Org")
    repo.add_member.assert_awaited_once_with(org.id, "user_abc", "admin")
    assert result_org.name == "Test Org"
    assert result_member.role == "admin"


@pytest.mark.asyncio
async def test_create_api_key_generates_krnl_prefix():
    org_id = uuid.uuid4()
    captured = {}

    async def fake_create_api_key(org_id, key_hash, key_prefix, name):
        captured["key_hash"] = key_hash
        captured["key_prefix"] = key_prefix
        return _make_api_key(org_id, key_hash=key_hash, prefix=key_prefix)

    repo = _mock_repo(create_api_key=fake_create_api_key)
    service = OrganizationService(repo)

    api_key, plaintext = await service.create_api_key(org_id, name="my key")

    assert plaintext.startswith("krnl_")
    assert len(plaintext) == 5 + 32  # "krnl_" + 32 hex chars
    assert hashlib.sha256(plaintext.encode()).hexdigest() == captured["key_hash"]
    assert captured["key_prefix"] == plaintext[:12]


@pytest.mark.asyncio
async def test_verify_api_key_hashes_before_lookup():
    org_id = uuid.uuid4()
    plaintext = "krnl_" + "a" * 32
    expected_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    expected_key = _make_api_key(org_id, key_hash=expected_hash)

    repo = _mock_repo(verify_api_key=AsyncMock(return_value=expected_key))
    service = OrganizationService(repo)

    result = await service.verify_api_key(plaintext)

    repo.verify_api_key.assert_awaited_once_with(expected_hash)
    assert result is not None
    assert result.org_id == org_id


@pytest.mark.asyncio
async def test_verify_api_key_returns_none_for_invalid():
    repo = _mock_repo(verify_api_key=AsyncMock(return_value=None))
    service = OrganizationService(repo)

    result = await service.verify_api_key("krnl_invalid_key_000000000000000")

    assert result is None


@pytest.mark.asyncio
async def test_is_admin_returns_true_for_admin_member():
    org_id = uuid.uuid4()
    member = _make_member(org_id, role="admin")
    repo = _mock_repo(get_member=AsyncMock(return_value=member))
    service = OrganizationService(repo)

    assert await service.is_admin(org_id, "user_abc") is True


@pytest.mark.asyncio
async def test_is_admin_returns_false_for_regular_member():
    org_id = uuid.uuid4()
    member = _make_member(org_id, role="member")
    repo = _mock_repo(get_member=AsyncMock(return_value=member))
    service = OrganizationService(repo)

    assert await service.is_admin(org_id, "user_abc") is False


@pytest.mark.asyncio
async def test_is_admin_returns_false_when_not_member():
    repo = _mock_repo(get_member=AsyncMock(return_value=None))
    service = OrganizationService(repo)

    assert await service.is_admin(uuid.uuid4(), "unknown_user") is False


@pytest.mark.asyncio
async def test_get_user_organization_delegates_to_repo():
    org = _make_org()
    repo = _mock_repo(get_organization_by_user_id=AsyncMock(return_value=org))
    service = OrganizationService(repo)

    result = await service.get_user_organization("user_abc")

    repo.get_organization_by_user_id.assert_awaited_once_with("user_abc")
    assert result is org


@pytest.mark.asyncio
async def test_get_user_organization_returns_none_when_not_found():
    repo = _mock_repo(get_organization_by_user_id=AsyncMock(return_value=None))
    service = OrganizationService(repo)

    result = await service.get_user_organization("unknown")

    assert result is None