"""Admin users router — Stack Auth user list enriched with org membership.

GET /admin/users — list all registered users with their org membership status.
Requires master admin access.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.infrastructure.database.models import OrgMemberRecord, OrgRecord

_log = logging.getLogger("kernel.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AdminUserResponse(BaseModel):
    user_id: str
    email: str | None
    email_verified: bool
    display_name: str | None
    signed_up_at: datetime | None
    is_restricted: bool
    org_id: str | None
    org_name: str | None
    role: str | None  # "admin" | "member" | None


class PaginatedAdminUsersResponse(BaseModel):
    items: list[AdminUserResponse]
    total: int
    page: int
    total_pages: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_to_dt(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


async def _batch_lookup_memberships(
    session: AsyncSession,
    user_ids: list[str],
) -> dict[str, dict]:
    """Return {user_id: {org_id, org_name, role}} for all given user_ids."""
    if not user_ids:
        return {}

    result = await session.execute(
        select(OrgMemberRecord, OrgRecord.name)
        .outerjoin(OrgRecord, OrgMemberRecord.org_id == OrgRecord.id)
        .where(OrgMemberRecord.user_id.in_(user_ids))
    )
    rows = result.all()
    return {
        row.OrgMemberRecord.user_id: {
            "org_id": str(row.OrgMemberRecord.org_id),
            "org_name": row.name,
            "role": row.OrgMemberRecord.role,
        }
        for row in rows
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/users", response_model=PaginatedAdminUsersResponse)
async def list_admin_users(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
) -> PaginatedAdminUsersResponse:
    """List all registered users (from Stack Auth) enriched with org membership.

    Uses a single batched DB query to avoid N+1 — no per-user lookup.
    Requires master admin access.
    """
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    from kernel_backend.config import get_settings

    settings = get_settings()

    if not settings.NEON_AUTH_API_KEY or not settings.NEON_AUTH_SECRET_SERVER_KEY:
        raise HTTPException(status_code=503, detail="Stack Auth not configured")

    # ── Fetch users from Stack Auth ──────────────────────────────────────────
    import httpx as _httpx

    client: _httpx.AsyncClient | None = getattr(request.app.state, "httpx_client", None)
    owns_client = False
    if client is None:
        client = _httpx.AsyncClient(timeout=10.0)
        owns_client = True

    try:
        # Stack Auth supports limit + after_cursor pagination; for simplicity
        # we fetch one page matching our own limit. cursor-based pagination can
        # be layered on top when user counts grow beyond a few hundred.
        resp = await client.get(
            f"{settings.NEON_AUTH_URL}/api/v1/users",
            params={"limit": limit, "after_cursor": None},
            headers={
                "x-stack-project-id": settings.NEON_AUTH_API_KEY,
                "x-stack-secret-server-key": settings.NEON_AUTH_SECRET_SERVER_KEY,
                "x-stack-access-type": "server",
            },
        )
    finally:
        if owns_client:
            await client.aclose()

    if resp.status_code != 200:
        _log.error("Stack Auth /users returned %d: %s", resp.status_code, resp.text[:300])
        raise HTTPException(status_code=502, detail="Failed to fetch users from Stack Auth")

    data = resp.json()
    raw_users: list[dict] = data.get("items", [])
    pagination = data.get("pagination", {})

    # ── Batch DB lookup for org membership ──────────────────────────────────
    user_ids = [u["id"] for u in raw_users if u.get("id")]

    session_factory = request.app.state.db_session_factory
    async with session_factory() as session:
        membership_map = await _batch_lookup_memberships(session, user_ids)

    # ── Assemble response ────────────────────────────────────────────────────
    items: list[AdminUserResponse] = []
    for u in raw_users:
        uid = u.get("id", "")
        membership = membership_map.get(uid, {})
        items.append(
            AdminUserResponse(
                user_id=uid,
                email=u.get("primary_email"),
                email_verified=bool(u.get("primary_email_verified", False)),
                display_name=u.get("display_name"),
                signed_up_at=_ms_to_dt(u.get("signed_up_at_millis")),
                is_restricted=bool(u.get("is_restricted", False)),
                org_id=membership.get("org_id"),
                org_name=membership.get("org_name"),
                role=membership.get("role"),
            )
        )

    total = len(items)  # Stack Auth total count not always returned; use items length
    return PaginatedAdminUsersResponse(
        items=items,
        total=total,
        page=page,
        total_pages=max(1, (total + limit - 1) // limit),
    )
