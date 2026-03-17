"""Invitation endpoints.

- POST /admin/invitations    — create + send email (admin only)
- GET  /admin/invitations    — list invitations (admin only)
- GET  /invitations/accept/{token} — validate token (public)
- POST /invitations/accept/{token} — accept invitation (authenticated user)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.api.dependencies import get_session
from kernel_backend.core.services.invitation_service import InvitationService
from kernel_backend.infrastructure.database.invitation_repository import InvitationRepository
from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository

admin_router = APIRouter(prefix="/admin/invitations", tags=["admin"])
public_router = APIRouter(prefix="/invitations", tags=["invitations"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateInvitationRequest(BaseModel):
    email: EmailStr
    org_id: UUID


class InvitationResponse(BaseModel):
    id: UUID
    token: UUID
    email: str
    org_id: UUID
    org_name: str | None
    status: str
    expires_at: datetime
    created_at: datetime


class AcceptInvitationRequest(BaseModel):
    user_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_service(session: AsyncSession) -> InvitationService:
    return InvitationService(
        invitation_repo=InvitationRepository(session),
        org_repo=OrganizationRepository(session),
    )


def _require_admin(request: Request) -> None:
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")


def _to_response(inv) -> InvitationResponse:
    return InvitationResponse(
        id=inv.id,
        token=inv.token,
        email=inv.email,
        org_id=inv.org_id,
        org_name=inv.org_name,
        status=inv.status,
        expires_at=inv.expires_at,
        created_at=inv.created_at,
    )


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@admin_router.post("", status_code=201, response_model=InvitationResponse)
async def create_invitation(
    body: CreateInvitationRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> InvitationResponse:
    """Create an invitation and send an email (admin only). Token valid for 7 days."""
    _require_admin(request)

    service = _get_service(session)
    invitation = await service.create_invitation(
        email=body.email,
        org_id=body.org_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )

    # Fire-and-forget email (errors are non-fatal)
    try:
        from kernel_backend.config import Settings
        settings = Settings()
        if settings.RESEND_API_KEY:
            from kernel_backend.infrastructure.email.resend_adapter import ResendEmailAdapter
            adapter = ResendEmailAdapter(
                api_key=settings.RESEND_API_KEY,
                from_email=settings.RESEND_FROM_EMAIL,
                frontend_base_url=settings.FRONTEND_BASE_URL,
            )
            await adapter.send_invitation(
                to_email=body.email,
                org_name=invitation.org_name or "Kernel Security",
                invite_token=str(invitation.token),
            )
    except Exception:
        pass  # Email failure does not abort invitation creation

    return _to_response(invitation)


@admin_router.get("", response_model=dict)
async def list_invitations(
    request: Request,
    session: AsyncSession = Depends(get_session),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    org_id: UUID | None = Query(None),
) -> dict:
    """List all invitations (admin only)."""
    _require_admin(request)

    service = _get_service(session)
    offset = (page - 1) * limit
    invitations = await service.list_invitations(org_id=org_id, limit=limit, offset=offset)
    total = await service.count_invitations(org_id=org_id)

    return {
        "items": [_to_response(inv).model_dump() for inv in invitations],
        "total": total,
        "page": page,
        "total_pages": max(1, (total + limit - 1) // limit),
    }


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------


@public_router.get("/accept/{token}", response_model=InvitationResponse)
async def validate_token(
    token: UUID,
    session: AsyncSession = Depends(get_session),
) -> InvitationResponse:
    """Validate an invitation token (public). Used by the frontend to preview org info."""
    service = _get_service(session)
    invitation = await service.validate_token(token)
    if invitation is None:
        raise HTTPException(status_code=404, detail="Invitation not found")
    if not invitation.is_valid:
        raise HTTPException(
            status_code=410,
            detail=f"Invitation is {invitation.status}",
        )
    return _to_response(invitation)


@public_router.post("/accept/{token}", response_model=InvitationResponse)
async def accept_invitation(
    token: UUID,
    body: AcceptInvitationRequest,
    session: AsyncSession = Depends(get_session),
) -> InvitationResponse:
    """Accept an invitation and add the user to the organisation."""
    service = _get_service(session)
    try:
        invitation = await service.accept_invitation(token=token, user_id=body.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=410, detail=str(exc))
    return _to_response(invitation)
