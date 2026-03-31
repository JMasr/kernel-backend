"""Storage adapter factory."""
from __future__ import annotations

from kernel_backend.core.ports.storage import StoragePort


def make_storage(settings: object) -> StoragePort:
    """Return the correct StoragePort adapter based on STORAGE_BACKEND."""
    if settings.STORAGE_BACKEND == "s3":
        from kernel_backend.infrastructure.storage.s3_storage import S3StorageAdapter

        return S3StorageAdapter(
            endpoint_url=settings.S3_ENDPOINT_URL,
            access_key_id=settings.S3_ACCESS_KEY_ID,
            secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            bucket_name=settings.S3_BUCKET_NAME,
            region=settings.S3_REGION,
        )

    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter

    return LocalStorageAdapter(
        base_path=settings.STORAGE_LOCAL_BASE_PATH,
        secret_key=settings.STORAGE_HMAC_SECRET,
    )
