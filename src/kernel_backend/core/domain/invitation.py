"""Invitation domain model."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID


@dataclass
class Invitation:
    """One-time invitation token to join an organization."""

    id: UUID
    token: UUID
    email: str
    org_id: UUID
    status: str          # "pending" | "accepted" | "expired"
    expires_at: datetime
    created_at: datetime
    accepted_at: datetime | None = None
    org_name: str | None = None  # populated via JOIN, not stored in DB

    @property
    def is_valid(self) -> bool:
        """Return True if the invitation can still be accepted."""
        if self.status != "pending":
            return False
        now = datetime.now(
            self.expires_at.tzinfo if self.expires_at.tzinfo else timezone.utc
        )
        return now <= self.expires_at
