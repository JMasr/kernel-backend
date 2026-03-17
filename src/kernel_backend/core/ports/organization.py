from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from kernel_backend.core.domain.organization import APIKey, Organization, OrganizationMember


class OrganizationPort(ABC):
    """Repository port for organization management."""

    @abstractmethod
    async def create_organization(self, name: str) -> Organization:
        """Create a new organization and return it."""

    @abstractmethod
    async def get_organization_by_id(self, org_id: UUID) -> Optional[Organization]:
        """Return organization by id, or None if not found."""

    @abstractmethod
    async def get_organization_by_user_id(self, user_id: str) -> Optional[Organization]:
        """Return the organization the user belongs to, or None."""

    @abstractmethod
    async def create_api_key(
        self,
        org_id: UUID,
        key_hash: str,
        key_prefix: str,
        name: Optional[str],
    ) -> APIKey:
        """Persist a new API key and return it."""

    @abstractmethod
    async def verify_api_key(self, key_hash: str) -> Optional[APIKey]:
        """Return the active APIKey matching the hash, or None. Updates last_used_at."""

    @abstractmethod
    async def add_member(self, org_id: UUID, user_id: str, role: str) -> OrganizationMember:
        """Add a user to an organization with the given role."""

    @abstractmethod
    async def get_member(self, org_id: UUID, user_id: str) -> Optional[OrganizationMember]:
        """Return membership record, or None if user is not a member."""

    @abstractmethod
    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Organization]:
        """Return all organizations (admin use), paginated."""

    @abstractmethod
    async def count_all(self) -> int:
        """Return total count of all organizations."""

    @abstractmethod
    async def update(self, org: Organization) -> Organization:
        """Persist name changes to an existing organization and return it."""

    @abstractmethod
    async def delete(self, org_id: UUID) -> None:
        """Hard-delete an organization (cascades to members, keys, videos)."""