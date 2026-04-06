"""Invitation port (repository interface)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from kernel_backend.core.domain.invitation import Invitation


class InvitationPort(ABC):
    """Repository ABC for invitation persistence."""

    @abstractmethod
    async def create(self, invitation: Invitation) -> Invitation: ...

    @abstractmethod
    async def get_by_token(self, token: UUID) -> Invitation | None: ...

    @abstractmethod
    async def update(self, invitation: Invitation) -> Invitation: ...

    @abstractmethod
    async def list(
        self,
        org_id: UUID | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Invitation]: ...

    @abstractmethod
    async def count(self, org_id: UUID | None = None) -> int: ...

    @abstractmethod
    async def get_pending_by_email_and_org(self, email: str, org_id: UUID) -> Invitation | None: ...
