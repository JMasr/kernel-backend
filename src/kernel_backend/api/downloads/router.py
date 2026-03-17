"""Download endpoint with HMAC signature verification."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/download", tags=["downloads"])


@router.get("/{key:path}", summary="Download a presigned file")
async def download_file(
    key: str,
    request: Request,
    signature: str = Query(..., description="HMAC signature"),
    expires: int = Query(..., description="Expiration UNIX timestamp"),
) -> FileResponse:
    """Serve a file identified by a presigned download URL.

    No API key required — the HMAC signature IS the authentication.
    Returns 403 if signature is invalid or URL has expired.
    Returns 404 if the file does not exist in storage.
    """
    storage = request.app.state.storage

    if not storage.verify_download_signature(key, signature, expires):
        raise HTTPException(status_code=403, detail="Invalid or expired download signature")

    file_path = storage._resolve(key)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file_path.name,
    )
