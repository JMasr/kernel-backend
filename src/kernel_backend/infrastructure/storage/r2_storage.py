from kernel_backend.core.ports.storage import StoragePort


class R2StorageAdapter(StoragePort):
    """Cloudflare R2 storage adapter. H1 deliverable — not yet implemented."""

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        raise NotImplementedError("R2StorageAdapter is an H1 deliverable")

    async def get(self, key: str) -> bytes:
        raise NotImplementedError("R2StorageAdapter is an H1 deliverable")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("R2StorageAdapter is an H1 deliverable")

    async def presigned_upload_url(self, key: str, expires_in: int) -> str:
        raise NotImplementedError("R2StorageAdapter is an H1 deliverable")

    async def presigned_download_url(self, key: str, expires_in: int) -> str:
        raise NotImplementedError("R2StorageAdapter is an H1 deliverable")
