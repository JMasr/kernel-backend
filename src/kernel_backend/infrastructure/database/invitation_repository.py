"""Invitation repository — SQLAlchemy implementation."""
from __future__ import annotations

import uuid
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.core.domain.invitation import Invitation
from kernel_backend.core.ports.invitation import InvitationPort
from kernel_backend.infrastructure.database.models import InvitationRecord, OrgRecord


def _row_to_domain(row: InvitationRecord, org_name: str | None = None) -> Invitation:
    return Invitation(
        id=row.id,
        token=row.token,
        email=row.email,
        org_id=row.org_id,
        status=row.status,
        expires_at=row.expires_at,
        created_at=row.created_at,
        accepted_at=row.accepted_at,
        org_name=org_name,
    )


class InvitationRepository(InvitationPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, invitation: Invitation) -> Invitation:
        record = InvitationRecord(
            id=invitation.id,
            token=invitation.token,
            email=invitation.email,
            org_id=invitation.org_id,
            status=invitation.status,
            expires_at=invitation.expires_at,
            created_at=invitation.created_at,
            accepted_at=invitation.accepted_at,
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        await self._session.commit()
        return _row_to_domain(record, org_name=invitation.org_name)

    async def get_by_token(self, token: UUID) -> Invitation | None:
        result = await self._session.execute(
            select(InvitationRecord, OrgRecord.name)
            .outerjoin(OrgRecord, InvitationRecord.org_id == OrgRecord.id)
            .where(InvitationRecord.token == token)
        )
        row = result.one_or_none()
        if row is None:
            return None
        inv_row, org_name = row
        return _row_to_domain(inv_row, org_name=org_name)

    async def update(self, invitation: Invitation) -> Invitation:
        await self._session.execute(
            update(InvitationRecord)
            .where(InvitationRecord.id == invitation.id)
            .values(status=invitation.status, accepted_at=invitation.accepted_at)
        )
        await self._session.commit()

        result = await self._session.execute(
            select(InvitationRecord, OrgRecord.name)
            .outerjoin(OrgRecord, InvitationRecord.org_id == OrgRecord.id)
            .where(InvitationRecord.id == invitation.id)
        )
        row = result.one()
        inv_row, org_name = row
        return _row_to_domain(inv_row, org_name=org_name)

    async def list(
        self,
        org_id: UUID | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Invitation]:
        stmt = (
            select(InvitationRecord, OrgRecord.name)
            .outerjoin(OrgRecord, InvitationRecord.org_id == OrgRecord.id)
            .order_by(InvitationRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if org_id is not None:
            stmt = stmt.where(InvitationRecord.org_id == org_id)

        result = await self._session.execute(stmt)
        return [_row_to_domain(inv, org_name=org_name) for inv, org_name in result.all()]

    async def count(self, org_id: UUID | None = None) -> int:
        stmt = select(func.count()).select_from(InvitationRecord)
        if org_id is not None:
            stmt = stmt.where(InvitationRecord.org_id == org_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()
