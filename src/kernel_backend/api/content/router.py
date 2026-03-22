"""Content management endpoints — org-scoped listing and download URLs."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.api.dependencies import get_session
from kernel_backend.infrastructure.database.repositories import VideoRepository

router = APIRouter(prefix="/content", tags=["content"])

_PRESIGNED_EXPIRY_SECONDS = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DownloadUrlResponse(BaseModel):
    download_url: str
    expires_in: int
    content_id: str
    filename: str


class ContentListItem(BaseModel):
    content_id: str
    author_id: str
    author_name: Optional[str] = None
    created_at: Optional[str] = None
    filename: Optional[str] = None


class ContentListResponse(BaseModel):
    items: list[ContentListItem]
    total: int
    page: int
    limit: int
    total_pages: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.delete("/{content_id}", status_code=204)
async def delete_content(
    content_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Delete signed content (org-scoped, author-scoped for non-admins).

    Admins can delete any content in their org.
    Regular members can only delete their own content.
    Returns 404 if content does not exist or belongs to a different org.
    Storage object is NOT deleted — storage cleanup is a separate concern.
    """
    org_id: UUID | None = getattr(request.state, "org_id", None)
    if org_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    is_admin = getattr(request.state, "is_admin", False)
    user_id: str | None = getattr(request.state, "user_id", None)

    repo = VideoRepository(session)

    # Non-admin members must own the content to delete it
    if not is_admin and user_id is not None:
        entry = await repo.get_by_content_id(str(content_id))
        if entry is None or entry.org_id != org_id:
            raise HTTPException(status_code=404, detail="Content not found or access denied")
        if entry.author_id != user_id:
            raise HTTPException(status_code=403, detail="You can only delete your own content")

    deleted = await repo.delete_by_content_id(content_id, org_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Content not found or access denied")

    return Response(status_code=204)


@router.get("/{content_id}/download", response_model=DownloadUrlResponse)
async def get_download_url(
    content_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> DownloadUrlResponse:
    """Generate a presigned download URL for a signed media file.

    Verifies org ownership — content belonging to a different org returns 404.
    The URL is valid for 1 hour.
    """
    org_id: UUID | None = getattr(request.state, "org_id", None)

    repo = VideoRepository(session)
    entry = await repo.get_by_content_id(str(content_id))

    if entry is None or entry.org_id != org_id:
        raise HTTPException(status_code=404, detail="Content not found or access denied")

    if not entry.signed_media_key:
        raise HTTPException(status_code=404, detail="Signed media file not available")

    storage = request.app.state.storage
    download_url = await storage.presigned_download_url(
        key=entry.signed_media_key,
        expires_in=_PRESIGNED_EXPIRY_SECONDS,
    )

    # Ensure absolute URL — LocalStorageAdapter returns relative paths;
    # R2/S3 adapters already return absolute URLs.
    if download_url.startswith("/"):
        download_url = str(request.base_url).rstrip("/") + download_url

    filename = entry.signed_media_key.rsplit("/", 1)[-1]

    return DownloadUrlResponse(
        download_url=download_url,
        expires_in=_PRESIGNED_EXPIRY_SECONDS,
        content_id=str(content_id),
        filename=filename,
    )


@router.get("", response_model=ContentListResponse)
async def list_content(
    request: Request,
    session: AsyncSession = Depends(get_session),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    author_id: Optional[str] = Query(None, description="Filter by author ID"),
) -> ContentListResponse:
    """List signed content for the authenticated organization.

    Results are strictly org-scoped — content from other orgs is never returned.
    Supports pagination and optional author_id filter.
    """
    org_id: UUID | None = getattr(request.state, "org_id", None)
    if org_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    offset = (page - 1) * limit
    repo = VideoRepository(session)

    # Role-aware scoping: members see only their own content; admins can filter freely.
    is_admin = getattr(request.state, "is_admin", False)
    user_id: str | None = getattr(request.state, "user_id", None)
    if not is_admin and user_id is not None:
        author_id = user_id  # force scope to caller; ignore query param

    rows = await repo.list_by_org_id(
        org_id=org_id,
        author_id=author_id,
        limit=limit,
        offset=offset,
    )
    total = await repo.count_by_org_id(org_id=org_id, author_id=author_id)

    items = [
        ContentListItem(
            content_id=entry.content_id,
            author_id=entry.author_id,
            author_name=author_name,
            created_at=created_at_iso,
            filename=entry.signed_media_key.rsplit("/", 1)[-1] if entry.signed_media_key else None,
        )
        for entry, author_name, created_at_iso in rows
    ]

    total_pages = max(1, (total + limit - 1) // limit)

    return ContentListResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )
