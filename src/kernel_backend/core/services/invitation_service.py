"""Invitation service — business logic for invitation management."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from kernel_backend.core.domain.invitation import Invitation
from kernel_backend.core.ports.invitation import InvitationPort
from kernel_backend.core.ports.organization import OrganizationPort


class InvitationService:
    def __init__(
        self,
        invitation_repo: InvitationPort,
        org_repo: OrganizationPort,
    ) -> None:
        self._inv = invitation_repo
        self._org = org_repo

    async def create_invitation(
        self,
        email: str,
        org_id: UUID,
        expires_at: datetime,
    ) -> Invitation:
        """Create and persist a new invitation token.

        If a valid pending invitation for the same email + org already exists,
        it is returned as-is (idempotent) so the caller can re-send the email
        without creating duplicates.
        """
        # Return existing pending invitation if it is still valid
        existing = await self._inv.get_pending_by_email_and_org(email, org_id)
        if existing is not None and existing.is_valid:
            org = await self._org.get_organization_by_id(org_id)
            existing.org_name = org.name if org else None
            return existing

        invitation = Invitation(
            id=uuid4(),
            token=uuid4(),
            email=email,
            org_id=org_id,
            status="pending",
            expires_at=expires_at,
            created_at=datetime.now(timezone.utc),
        )
        saved = await self._inv.create(invitation)

        # Attach org name for email template
        org = await self._org.get_organization_by_id(org_id)
        saved.org_name = org.name if org else None
        return saved

    async def get_by_token(self, token: UUID) -> Invitation | None:
        return await self._inv.get_by_token(token)

    async def validate_token(self, token: UUID) -> Invitation | None:
        """Return the invitation if it exists and is still pending/not expired.

        If the token exists but has expired, marks it expired and returns it
        (caller can check `invitation.status == "expired"`).
        """
        invitation = await self._inv.get_by_token(token)
        if invitation is None:
            return None

        if invitation.status == "pending" and not invitation.is_valid:
            invitation.status = "expired"
            invitation = await self._inv.update(invitation)

        return invitation

    async def accept_invitation(self, token: UUID, user_id: str) -> Invitation:
        """Accept the invitation — add user to org and consume the token.

        Raises ValueError if the token is invalid or expired.
        """
        invitation = await self.validate_token(token)
        if invitation is None or not invitation.is_valid:
            raise ValueError("Invalid or expired invitation")

        await self._org.add_member(
            org_id=invitation.org_id,
            user_id=user_id,
            role="member",
        )

        invitation.status = "accepted"
        invitation.accepted_at = datetime.now(timezone.utc)
        return await self._inv.update(invitation)

    async def list_invitations(
        self,
        org_id: UUID | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Invitation]:
        return await self._inv.list(org_id=org_id, limit=limit, offset=offset)

    async def count_invitations(self, org_id: UUID | None = None) -> int:
        return await self._inv.count(org_id=org_id)
