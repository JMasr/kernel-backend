import hashlib
import hmac
import time
from pathlib import Path

from kernel_backend.core.ports.storage import StorageKeyNotFoundError, StoragePort


class LocalStorageAdapter(StoragePort):
    """StoragePort adapter backed by the local filesystem. For development only."""

    def __init__(self, base_path: Path, secret_key: str | None = None) -> None:
        self._base = base_path
        self._secret_key = secret_key

    def _resolve(self, key: str) -> Path:
        return self._base / key

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def get(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise StorageKeyNotFoundError(key)

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass  # idempotent

    async def presigned_upload_url(self, key: str, expires_in: int) -> str:
        return f"file://{self._resolve(key).resolve()}"

    async def presigned_download_url(self, key: str, expires_in: int) -> str:
        if self._secret_key is None:
            return f"file://{self._resolve(key).resolve()}"
        expires_at = int(time.time()) + expires_in
        message = f"{key}:{expires_at}"
        signature = hmac.new(
            self._secret_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"/download/{key}?signature={signature}&expires={expires_at}"

    def verify_download_signature(self, key: str, signature: str, expires: int) -> bool:
        """Verify HMAC signature for a presigned download request."""
        if self._secret_key is None:
            return False
        if int(time.time()) > expires:
            return False
        message = f"{key}:{expires}"
        expected = hmac.new(
            self._secret_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)
