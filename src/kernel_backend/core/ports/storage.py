from abc import ABC, abstractmethod


class StorageError(Exception):
    """Base exception for storage operations."""


class StorageKeyNotFoundError(StorageError):
    """Raised when a storage key does not exist."""


class StoragePort(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str) -> None:
        """Store data at key."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Retrieve data at key. Raises StorageKeyNotFoundError if missing."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key. Idempotent — no error if key missing."""

    @abstractmethod
    async def presigned_upload_url(self, key: str, expires_in: int) -> str:
        """Return a presigned URL for uploading to key."""

    @abstractmethod
    async def presigned_download_url(self, key: str, expires_in: int) -> str:
        """Return a presigned URL for downloading from key."""
