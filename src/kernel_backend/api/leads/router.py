"""Leads router.

public_router:
  POST /leads — registers a new inbound lead (no auth required)

admin_router:
  GET  /admin/leads — paginated lead list (admin only)
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select

from kernel_backend.infrastructure.database.models import LeadRecord

_log = logging.getLogger("kernel.leads")

public_router = APIRouter(tags=["leads"])
admin_router = APIRouter(prefix="/admin", tags=["admin", "leads"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CreateLeadRequest(BaseModel):
    email: EmailStr
    lead_type: str          # "customer" | "investor"
    message: str | None = None
    source_page: str | None = None


class LeadResponse(BaseModel):
    status: str
    deck_url: str | None = None   # reserved for future use


class AdminLeadItem(BaseModel):
    id: str
    email: str
    lead_type: str
    message: str | None
    source_page: str | None
    created_at: datetime


class PaginatedLeadsResponse(BaseModel):
    items: list[AdminLeadItem]
    total: int
    page: int
    total_pages: int


# ---------------------------------------------------------------------------
# Public endpoint
# ---------------------------------------------------------------------------


@public_router.post("/leads", response_model=LeadResponse)
async def create_lead(body: CreateLeadRequest, request: Request) -> LeadResponse:
    """Register a new inbound lead from the landing page CTA."""
    if body.lead_type not in ("customer", "investor"):
        raise HTTPException(
            status_code=422,
            detail="lead_type must be 'customer' or 'investor'",
        )

    session_factory = request.app.state.db_session_factory
    async with session_factory() as session:
        record = LeadRecord(
            id=uuid4(),
            email=body.email,
            lead_type=body.lead_type,
            message=body.message,
            source_page=body.source_page,
        )
        session.add(record)
        await session.commit()

    _log.info(
        "New lead saved: email=%s type=%s page=%s",
        body.email,
        body.lead_type,
        body.source_page,
    )

    # Fire-and-forget email notification to admin
    try:
        from kernel_backend.config import get_settings
        from kernel_backend.infrastructure.email.resend_adapter import ResendEmailAdapter

        settings = get_settings()
        if settings.RESEND_API_KEY and settings.ADMIN_EMAIL:
            adapter = ResendEmailAdapter(
                api_key=settings.RESEND_API_KEY,
                from_email=settings.RESEND_FROM_EMAIL,
                frontend_base_url=settings.FRONTEND_BASE_URL,
            )
            await adapter.send_new_lead(
                to_email=settings.ADMIN_EMAIL,
                lead_email=body.email,
                lead_type=body.lead_type,
                message=body.message,
                source_page=body.source_page,
            )
    except Exception as exc:
        _log.error("Failed to send lead notification email: %s", exc)

    return LeadResponse(status="ok")


# ---------------------------------------------------------------------------
# Admin endpoint
# ---------------------------------------------------------------------------


@admin_router.get("/leads", response_model=PaginatedLeadsResponse)
async def list_leads(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    lead_type: str | None = Query(None),
) -> PaginatedLeadsResponse:
    """List all inbound leads, newest first. Admin only."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    session_factory = request.app.state.db_session_factory
    async with session_factory() as session:
        q = select(LeadRecord)
        if lead_type:
            q = q.where(LeadRecord.lead_type == lead_type)
        q = q.order_by(LeadRecord.created_at.desc())

        count_q = select(func.count()).select_from(q.subquery())
        total: int = (await session.execute(count_q)).scalar_one()

        q = q.offset((page - 1) * limit).limit(limit)
        rows = (await session.execute(q)).scalars().all()

    items = [
        AdminLeadItem(
            id=str(r.id),
            email=r.email,
            lead_type=r.lead_type,
            message=r.message,
            source_page=r.source_page,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return PaginatedLeadsResponse(
        items=items,
        total=total,
        page=page,
        total_pages=max(1, (total + limit - 1) // limit),
    )
