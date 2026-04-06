import hashlib
import secrets
from typing import Optional
from uuid import UUID

from kernel_backend.core.domain.organization import APIKey, Organization, OrganizationMember
from kernel_backend.core.ports.organization import OrganizationPort


class OrganizationService:
    """Business logic for organization, API key, and membership management."""

    def __init__(self, repo: OrganizationPort) -> None:
        self._repo = repo

    async def create_organization(
        self, name: str, admin_user_id: str
    ) -> tuple[Organization, OrganizationMember]:
        """Create an org and add the requesting user as admin."""
        org = await self._repo.create_organization(name)
        member = await self._repo.add_member(org.id, admin_user_id, "admin")
        return org, member

    async def create_api_key(
        self, org_id: UUID, name: Optional[str] = None
    ) -> tuple[APIKey, str]:
        """Generate a new API key for the org. Returns (APIKey, plaintext_key)."""
        plaintext = f"krnl_{secrets.token_hex(16)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        key_prefix = plaintext[:12]
        api_key = await self._repo.create_api_key(org_id, key_hash, key_prefix, name)
        return api_key, plaintext

    async def verify_api_key(self, plaintext_key: str) -> Optional[APIKey]:
        """Verify a plaintext key. Returns APIKey if valid and active, else None."""
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
        return await self._repo.verify_api_key(key_hash)

    async def add_member(
        self, org_id: UUID, user_id: str, role: str
    ) -> OrganizationMember:
        """Add a user to an organization with the given role."""
        return await self._repo.add_member(org_id, user_id, role)

    async def get_user_organization(self, user_id: str) -> Optional[Organization]:
        """Return the organization the user belongs to, or None."""
        return await self._repo.get_organization_by_user_id(user_id)

    async def is_admin(self, org_id: UUID, user_id: str) -> bool:
        """Return True if user is an admin of the given org."""
        member = await self._repo.get_member(org_id, user_id)
        return member is not None and member.role == "admin"

    async def list_organizations(
        self, limit: int = 20, offset: int = 0
    ) -> tuple[list[Organization], int]:
        """Return (organizations, total_count) — admin use only."""
        orgs = await self._repo.list_all(limit=limit, offset=offset)
        total = await self._repo.count_all()
        return orgs, total

    async def update_organization(self, org_id: UUID, name: str) -> Organization:
        """Update the name of an existing organization."""
        org = await self._repo.get_organization_by_id(org_id)
        if org is None:
            raise ValueError("Organization not found")
        org.name = name
        return await self._repo.update(org)

    async def delete_organization(self, org_id: UUID) -> None:
        """Delete an organization (cascades to members, keys, and content)."""
        org = await self._repo.get_organization_by_id(org_id)
        if org is None:
            raise ValueError("Organization not found")
        await self._repo.delete(org_id)

    async def list_members(
        self, org_id: UUID, limit: int = 20, offset: int = 0
    ) -> tuple[list[OrganizationMember], int]:
        """Return (members, total_count) for an organization."""
        members = await self._repo.list_members(org_id, limit=limit, offset=offset)
        total = await self._repo.count_members(org_id)
        return members, total

    async def remove_member(self, org_id: UUID, user_id: str) -> None:
        """Remove a user from an organization."""
        if not await self._repo.get_organization_by_id(org_id):
            raise ValueError("Organization not found")
        await self._repo.remove_member(org_id, user_id)

    async def update_member_role(
        self, org_id: UUID, user_id: str, role: str
    ) -> OrganizationMember:
        """Change a member's role."""
        if not await self._repo.get_organization_by_id(org_id):
            raise ValueError("Organization not found")
        if not await self._repo.get_member(org_id, user_id):
            raise ValueError("Member not found")
        return await self._repo.update_member_role(org_id, user_id, role)