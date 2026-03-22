"""GET /me — current user profile endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository
from kernel_backend.infrastructure.database.repositories import IdentityRepository

router = APIRouter(prefix="/me", tags=["users"])
_log = logging.getLogger("kernel.users")


class MeResponse(BaseModel):
    user_id: str
    email: str | None
    role: str  # "master_admin" | "member"
    org_id: str | None
    org_name: str | None
    has_key_pair: bool


@router.get("", response_model=MeResponse)
async def get_me(request: Request) -> MeResponse:
    """Return the authenticated user's profile, role, and organization.

    - ``role``: ``"master_admin"`` if the user's email matches ``ADMIN_EMAIL``,
      otherwise ``"member"``.
    - ``has_key_pair``: ``True`` if the user has generated an Ed25519 identity
      via ``POST /identity/generate``.
    """
    user_id: str | None = getattr(request.state, "user_id", None)
    _log.debug(
        "GET /me  auth_type=%s  user_id=%s  org_id=%s  is_admin=%s",
        getattr(request.state, "auth_type", "?"),
        user_id,
        getattr(request.state, "org_id", "?"),
        getattr(request.state, "is_admin", "?"),
    )
    if not user_id:
        _log.warning("GET /me rejected: user_id not set on request.state (auth_type=%s)",
                     getattr(request.state, "auth_type", "?"))
        raise HTTPException(status_code=401, detail="Authentication required")

    org_id = getattr(request.state, "org_id", None)
    is_admin: bool = getattr(request.state, "is_admin", False)
    email: str | None = getattr(request.state, "email", None)

    session_factory = request.app.state.db_session_factory

    # Resolve org name
    org_name: str | None = None
    if org_id is not None:
        try:
            async with session_factory() as session:
                repo = OrganizationRepository(session)
                org = await repo.get_organization_by_id(org_id)
                if org:
                    org_name = org.name
        except Exception:
            pass

    # Check if user has an Ed25519 key pair (author_id == user_id after Phase B.1)
    has_key_pair = False
    try:
        async with session_factory() as session:
            identity_repo = IdentityRepository(session)
            cert = await identity_repo.get_by_author_id(user_id)
            has_key_pair = cert is not None
    except Exception:
        pass

    response = MeResponse(
        user_id=user_id,
        email=email,
        role="master_admin" if is_admin else "member",
        org_id=str(org_id) if org_id is not None else None,
        org_name=org_name,
        has_key_pair=has_key_pair,
    )
    _log.info(
        "GET /me  → role=%s  org_id=%s  org_name=%s  has_key_pair=%s",
        response.role, response.org_id, response.org_name, response.has_key_pair,
    )
    return response
