import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import delete as sa_delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.core.domain.organization import APIKey, Organization, OrganizationMember
from kernel_backend.core.ports.organization import OrganizationPort
from kernel_backend.infrastructure.database.models import ApiKeyRecord, OrgMemberRecord, OrgRecord


def _org_row_to_domain(row: OrgRecord) -> Organization:
    return Organization(
        id=row.id,
        name=row.name,
        pepper_v1=row.pepper_v1,
        current_pepper_version=row.current_pepper_version,
        created_at=row.created_at,
    )


def _apikey_row_to_domain(row: ApiKeyRecord) -> APIKey:
    return APIKey(
        id=row.id,
        org_id=row.org_id,
        key_hash=row.key_hash,
        key_prefix=row.key_prefix,
        name=row.name,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        is_active=row.is_active,
    )


def _member_row_to_domain(row: OrgMemberRecord) -> OrganizationMember:
    return OrganizationMember(
        id=row.id,
        org_id=row.org_id,
        user_id=row.user_id,
        role=row.role,
        created_at=row.created_at,
    )


class OrganizationRepository(OrganizationPort):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_organization(self, name: str) -> Organization:
        org = OrgRecord(id=uuid.uuid4(), name=name, pepper_v1=secrets.token_hex(32))
        self._session.add(org)
        await self._session.flush()
        await self._session.refresh(org)
        await self._session.commit()
        return _org_row_to_domain(org)

    async def get_organization_by_id(self, org_id: UUID) -> Optional[Organization]:
        result = await self._session.execute(
            select(OrgRecord).where(OrgRecord.id == org_id)
        )
        row = result.scalar_one_or_none()
        return None if row is None else _org_row_to_domain(row)

    async def get_organization_by_user_id(self, user_id: str) -> Optional[Organization]:
        result = await self._session.execute(
            select(OrgRecord)
            .join(OrgMemberRecord, OrgMemberRecord.org_id == OrgRecord.id)
            .where(OrgMemberRecord.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        return None if row is None else _org_row_to_domain(row)

    async def create_api_key(
        self,
        org_id: UUID,
        key_hash: str,
        key_prefix: str,
        name: Optional[str],
    ) -> APIKey:
        record = ApiKeyRecord(
            id=uuid.uuid4(),
            org_id=org_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            is_active=True,
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        await self._session.commit()
        return _apikey_row_to_domain(record)

    async def verify_api_key(self, key_hash: str) -> Optional[APIKey]:
        result = await self._session.execute(
            select(ApiKeyRecord).where(
                ApiKeyRecord.key_hash == key_hash,
                ApiKeyRecord.is_active.is_(True),
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        # Update last_used_at
        await self._session.execute(
            update(ApiKeyRecord)
            .where(ApiKeyRecord.id == row.id)
            .values(last_used_at=datetime.now(timezone.utc))
        )
        await self._session.commit()
        return _apikey_row_to_domain(row)

    async def add_member(self, org_id: UUID, user_id: str, role: str) -> OrganizationMember:
        stmt = (
            insert(OrgMemberRecord)
            .values(
                id=uuid.uuid4(),
                org_id=org_id,
                user_id=user_id,
                role=role,
            )
            .on_conflict_do_nothing(constraint="uq_org_members_org_user")
        )
        await self._session.execute(stmt)
        await self._session.commit()

        result = await self._session.execute(
            select(OrgMemberRecord).where(
                OrgMemberRecord.org_id == org_id,
                OrgMemberRecord.user_id == user_id,
            )
        )
        row = result.scalar_one()
        return _member_row_to_domain(row)

    async def get_member(self, org_id: UUID, user_id: str) -> Optional[OrganizationMember]:
        result = await self._session.execute(
            select(OrgMemberRecord).where(
                OrgMemberRecord.org_id == org_id,
                OrgMemberRecord.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()
        return None if row is None else _member_row_to_domain(row)

    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Organization]:
        result = await self._session.execute(
            select(OrgRecord).order_by(OrgRecord.created_at.desc()).limit(limit).offset(offset)
        )
        return [_org_row_to_domain(r) for r in result.scalars().all()]

    async def count_all(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(OrgRecord))
        return result.scalar_one()

    async def update(self, org: Organization) -> Organization:
        await self._session.execute(
            update(OrgRecord).where(OrgRecord.id == org.id).values(name=org.name)
        )
        await self._session.commit()
        result = await self._session.execute(select(OrgRecord).where(OrgRecord.id == org.id))
        row = result.scalar_one()
        return _org_row_to_domain(row)

    async def delete(self, org_id: UUID) -> None:
        await self._session.execute(sa_delete(OrgRecord).where(OrgRecord.id == org_id))
        await self._session.commit()