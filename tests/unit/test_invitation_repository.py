"""Unit tests for InvitationRepository against an in-memory SQLite database."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kernel_backend.infrastructure.database.invitation_repository import InvitationRepository
from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository
from kernel_backend.core.domain.invitation import Invitation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_org(db_session, name: str = "Test Org"):
    repo = OrganizationRepository(db_session)
    return await repo.create_organization(name)


def _make_invitation(org_id, *, status: str = "pending") -> Invitation:
    return Invitation(
        id=uuid.uuid4(),
        token=uuid.uuid4(),
        email="invite@example.com",
        org_id=org_id,
        status=status,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_invitation_returns_persisted_record(db_session):
    org = await _create_org(db_session)
    repo = InvitationRepository(db_session)
    inv = _make_invitation(org.id)

    saved = await repo.create(inv)

    assert saved.id == inv.id
    assert saved.token == inv.token
    assert saved.email == inv.email
    assert saved.status == "pending"


# ---------------------------------------------------------------------------
# get_by_token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_token_returns_invitation(db_session):
    org = await _create_org(db_session)
    repo = InvitationRepository(db_session)
    inv = _make_invitation(org.id)
    await repo.create(inv)

    fetched = await repo.get_by_token(inv.token)

    assert fetched is not None
    assert fetched.id == inv.id
    assert fetched.email == inv.email


@pytest.mark.asyncio
async def test_get_by_token_includes_org_name(db_session):
    org = await _create_org(db_session, name="Widget Co")
    repo = InvitationRepository(db_session)
    inv = _make_invitation(org.id)
    await repo.create(inv)

    fetched = await repo.get_by_token(inv.token)

    assert fetched.org_name == "Widget Co"


@pytest.mark.asyncio
async def test_get_by_token_returns_none_for_unknown(db_session):
    repo = InvitationRepository(db_session)
    result = await repo.get_by_token(uuid.uuid4())

    assert result is None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_marks_accepted(db_session):
    org = await _create_org(db_session)
    repo = InvitationRepository(db_session)
    inv = _make_invitation(org.id)
    await repo.create(inv)

    inv.status = "accepted"
    inv.accepted_at = datetime.now(timezone.utc)
    updated = await repo.update(inv)

    assert updated.status == "accepted"
    assert updated.accepted_at is not None


# ---------------------------------------------------------------------------
# list / count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_returns_invitations_for_org(db_session):
    org = await _create_org(db_session)
    repo = InvitationRepository(db_session)
    await repo.create(_make_invitation(org.id))
    await repo.create(_make_invitation(org.id))

    results = await repo.list(org_id=org.id)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_list_without_filter_returns_all(db_session):
    org1 = await _create_org(db_session, "Org A")
    org2 = await _create_org(db_session, "Org B")
    repo = InvitationRepository(db_session)
    await repo.create(_make_invitation(org1.id))
    await repo.create(_make_invitation(org2.id))

    results = await repo.list()

    assert len(results) == 2


@pytest.mark.asyncio
async def test_count_returns_total(db_session):
    org = await _create_org(db_session)
    repo = InvitationRepository(db_session)
    await repo.create(_make_invitation(org.id))
    await repo.create(_make_invitation(org.id))

    total = await repo.count(org_id=org.id)

    assert total == 2


@pytest.mark.asyncio
async def test_count_without_filter_counts_all(db_session):
    org1 = await _create_org(db_session, "Org A")
    org2 = await _create_org(db_session, "Org B")
    repo = InvitationRepository(db_session)
    await repo.create(_make_invitation(org1.id))
    await repo.create(_make_invitation(org2.id))

    total = await repo.count()

    assert total == 2
