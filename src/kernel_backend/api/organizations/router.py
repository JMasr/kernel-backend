from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.api.dependencies import get_session
from kernel_backend.api.organizations.schemas import (
    ApiKeyResponse,
    CreateApiKeyRequest,
    CreateOrganizationRequest,
    OrganizationResponse,
    PaginatedOrganizationsResponse,
    UpdateOrganizationRequest,
    UserOrganizationResponse,
)
from kernel_backend.core.services.organization_service import OrganizationService
from kernel_backend.infrastructure.database.organization_repository import OrganizationRepository

router = APIRouter(prefix="/organizations", tags=["organizations"])


def _get_service(session: AsyncSession) -> OrganizationService:
    return OrganizationService(OrganizationRepository(session))


@router.post("", status_code=201, response_model=OrganizationResponse)
async def create_organization(
    body: CreateOrganizationRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> OrganizationResponse:
    """Create a new organization. Requires master admin."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    user_id: str | None = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    service = _get_service(session)
    org, _ = await service.create_organization(name=body.name, admin_user_id=user_id)
    return OrganizationResponse(
        org_id=org.id,
        name=org.name,
        created_at=org.created_at,
    )


@router.post(
    "/{org_id}/api-keys",
    status_code=201,
    response_model=ApiKeyResponse,
)
async def create_api_key(
    org_id: UUID,
    body: CreateApiKeyRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ApiKeyResponse:
    """Generate a new API key for the org. Requires admin role."""
    user_id: str | None = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    service = _get_service(session)
    if not await service.is_admin(org_id, user_id):
        raise HTTPException(status_code=403, detail="Admin role required")
    api_key, plaintext = await service.create_api_key(org_id, name=body.name)
    return ApiKeyResponse(
        key_id=api_key.id,
        key_prefix=api_key.key_prefix,
        name=api_key.name,
        plaintext_key=plaintext,
        created_at=api_key.created_at,
    )


@router.get(
    "/users/{user_id}/organization",
    response_model=UserOrganizationResponse,
)
async def get_user_organization(
    user_id: str,
    session: AsyncSession = Depends(get_session),
) -> UserOrganizationResponse:
    """Look up the organization a user belongs to."""
    service = _get_service(session)
    org = await service.get_user_organization(user_id)
    if org is None:
        raise HTTPException(status_code=404, detail="User has no organization")
    return UserOrganizationResponse(
        org_id=org.id,
        name=org.name,
        created_at=org.created_at,
    )


@router.get("", response_model=PaginatedOrganizationsResponse)
async def list_organizations(
    request: Request,
    session: AsyncSession = Depends(get_session),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
) -> PaginatedOrganizationsResponse:
    """List organizations.

    - Admins: all organizations, paginated.
    - Regular users: only their own organization.
    """
    service = _get_service(session)
    is_admin = getattr(request.state, "is_admin", False)

    if is_admin:
        orgs, total = await service.list_organizations(
            limit=limit, offset=(page - 1) * limit
        )
    else:
        user_id = getattr(request.state, "user_id", None)
        org = await service.get_user_organization(user_id) if user_id else None
        orgs = [org] if org else []
        total = 1 if org else 0

    return PaginatedOrganizationsResponse(
        items=[OrganizationResponse(org_id=o.id, name=o.name, created_at=o.created_at) for o in orgs],
        total=total,
        page=page,
        total_pages=max(1, (total + limit - 1) // limit),
    )


@router.patch("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: UUID,
    body: UpdateOrganizationRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> OrganizationResponse:
    """Update organization name (admin only)."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    service = _get_service(session)
    try:
        org = await service.update_organization(org_id, name=body.name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return OrganizationResponse(org_id=org.id, name=org.name, created_at=org.created_at)


@router.delete("/{org_id}", status_code=204)
async def delete_organization(
    org_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete an organization and all its data (admin only)."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    service = _get_service(session)
    try:
        await service.delete_organization(org_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))