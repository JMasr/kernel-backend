"""Unit tests for InvitationService using mock ports."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from kernel_backend.core.domain.invitation import Invitation
from kernel_backend.core.domain.organization import Organization
from kernel_backend.core.services.invitation_service import InvitationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_invitation(
    *,
    status: str = "pending",
    expires_delta: timedelta = timedelta(days=7),
    org_id: UUID | None = None,
) -> Invitation:
    return Invitation(
        id=uuid.uuid4(),
        token=uuid.uuid4(),
        email="test@example.com",
        org_id=org_id or uuid.uuid4(),
        status=status,
        expires_at=datetime.now(timezone.utc) + expires_delta,
        created_at=datetime.now(timezone.utc),
    )


def _make_org(org_id: UUID | None = None, name: str = "Test Org") -> Organization:
    return Organization(
        id=org_id or uuid.uuid4(),
        name=name,
        pepper_v1=None,
        current_pepper_version=1,
        created_at=datetime.now(timezone.utc),
    )


def _mock_inv_repo(**overrides) -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.get_by_token = AsyncMock()
    repo.get_pending_by_email_and_org = AsyncMock(return_value=None)
    repo.update = AsyncMock()
    repo.list = AsyncMock(return_value=[])
    repo.count = AsyncMock(return_value=0)
    for k, v in overrides.items():
        setattr(repo, k, v)
    return repo


def _mock_org_repo(**overrides) -> MagicMock:
    repo = MagicMock()
    repo.get_organization_by_id = AsyncMock(return_value=_make_org())
    repo.add_member = AsyncMock()
    for k, v in overrides.items():
        setattr(repo, k, v)
    return repo


# ---------------------------------------------------------------------------
# create_invitation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_invitation_persists_and_returns_with_org_name():
    invitation = _make_invitation()
    org = _make_org(org_id=invitation.org_id, name="Acme Corp")
    inv_repo = _mock_inv_repo(create=AsyncMock(return_value=invitation))
    org_repo = _mock_org_repo(get_organization_by_id=AsyncMock(return_value=org))

    service = InvitationService(inv_repo, org_repo)
    result = await service.create_invitation(
        email=invitation.email,
        org_id=invitation.org_id,
        expires_at=invitation.expires_at,
    )

    inv_repo.create.assert_awaited_once()
    assert result.org_name == "Acme Corp"


@pytest.mark.asyncio
async def test_create_invitation_org_name_none_when_org_missing():
    invitation = _make_invitation()
    inv_repo = _mock_inv_repo(create=AsyncMock(return_value=invitation))
    org_repo = _mock_org_repo(get_organization_by_id=AsyncMock(return_value=None))

    service = InvitationService(inv_repo, org_repo)
    result = await service.create_invitation(
        email=invitation.email,
        org_id=invitation.org_id,
        expires_at=invitation.expires_at,
    )

    assert result.org_name is None


# ---------------------------------------------------------------------------
# validate_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_token_returns_pending_invitation():
    invitation = _make_invitation(status="pending")
    inv_repo = _mock_inv_repo(get_by_token=AsyncMock(return_value=invitation))

    service = InvitationService(inv_repo, _mock_org_repo())
    result = await service.validate_token(invitation.token)

    assert result is not None
    assert result.status == "pending"
    # Should NOT call update — invitation is valid
    inv_repo.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_token_marks_expired_and_returns():
    invitation = _make_invitation(status="pending", expires_delta=timedelta(days=-1))
    updated = Invitation(
        id=invitation.id, token=invitation.token, email=invitation.email,
        org_id=invitation.org_id, status="expired",
        expires_at=invitation.expires_at, created_at=invitation.created_at,
    )
    inv_repo = _mock_inv_repo(
        get_by_token=AsyncMock(return_value=invitation),
        update=AsyncMock(return_value=updated),
    )

    service = InvitationService(inv_repo, _mock_org_repo())
    result = await service.validate_token(invitation.token)

    inv_repo.update.assert_awaited_once()
    assert result.status == "expired"


@pytest.mark.asyncio
async def test_validate_token_returns_none_when_not_found():
    inv_repo = _mock_inv_repo(get_by_token=AsyncMock(return_value=None))

    service = InvitationService(inv_repo, _mock_org_repo())
    result = await service.validate_token(uuid.uuid4())

    assert result is None


# ---------------------------------------------------------------------------
# accept_invitation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_invitation_adds_member_and_marks_accepted():
    invitation = _make_invitation(status="pending")
    accepted = Invitation(
        id=invitation.id, token=invitation.token, email=invitation.email,
        org_id=invitation.org_id, status="accepted",
        expires_at=invitation.expires_at, created_at=invitation.created_at,
        accepted_at=datetime.now(timezone.utc),
    )
    inv_repo = _mock_inv_repo(
        get_by_token=AsyncMock(return_value=invitation),
        update=AsyncMock(return_value=accepted),
    )
    org_repo = _mock_org_repo()

    service = InvitationService(inv_repo, org_repo)
    result = await service.accept_invitation(token=invitation.token, user_id="user_xyz")

    org_repo.add_member.assert_awaited_once_with(
        org_id=invitation.org_id, user_id="user_xyz", role="member"
    )
    assert result.status == "accepted"


@pytest.mark.asyncio
async def test_accept_invitation_raises_for_expired():
    invitation = _make_invitation(status="pending", expires_delta=timedelta(days=-1))
    expired = Invitation(
        id=invitation.id, token=invitation.token, email=invitation.email,
        org_id=invitation.org_id, status="expired",
        expires_at=invitation.expires_at, created_at=invitation.created_at,
    )
    inv_repo = _mock_inv_repo(
        get_by_token=AsyncMock(return_value=invitation),
        update=AsyncMock(return_value=expired),
    )

    service = InvitationService(inv_repo, _mock_org_repo())
    with pytest.raises(ValueError, match="expired"):
        await service.accept_invitation(token=invitation.token, user_id="user_xyz")


@pytest.mark.asyncio
async def test_accept_invitation_raises_for_already_accepted():
    invitation = _make_invitation(status="accepted")
    inv_repo = _mock_inv_repo(get_by_token=AsyncMock(return_value=invitation))

    service = InvitationService(inv_repo, _mock_org_repo())
    with pytest.raises(ValueError):
        await service.accept_invitation(token=invitation.token, user_id="user_xyz")


# ---------------------------------------------------------------------------
# list_invitations / count_invitations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_invitations_delegates_to_repo():
    invs = [_make_invitation(), _make_invitation()]
    inv_repo = _mock_inv_repo(list=AsyncMock(return_value=invs))

    service = InvitationService(inv_repo, _mock_org_repo())
    result = await service.list_invitations(org_id=None, limit=10, offset=0)

    inv_repo.list.assert_awaited_once_with(org_id=None, limit=10, offset=0)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_count_invitations_delegates_to_repo():
    inv_repo = _mock_inv_repo(count=AsyncMock(return_value=5))

    service = InvitationService(inv_repo, _mock_org_repo())
    result = await service.count_invitations(org_id=None)

    assert result == 5
