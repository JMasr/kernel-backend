"""Contract tests for StoragePort adapters.

Every StoragePort implementation must pass these tests unchanged.
Add new adapters by extending the ``storage`` fixture's ``params``.
"""
from __future__ import annotations

import pytest

from kernel_backend.core.ports.storage import StorageKeyNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["local"])
def storage(request, tmp_path):
    """Yield a StoragePort adapter for each backend under test."""
    if request.param == "local":
        from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter

        return LocalStorageAdapter(base_path=tmp_path)

    raise ValueError(f"Unknown storage backend: {request.param}")


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_and_get_roundtrip(storage):
    data = b"hello world"
    await storage.put("test/key.bin", data, "application/octet-stream")
    result = await storage.get("test/key.bin")
    assert result == data


@pytest.mark.asyncio
async def test_get_missing_key_raises(storage):
    with pytest.raises(StorageKeyNotFoundError):
        await storage.get("nonexistent/key.bin")


@pytest.mark.asyncio
async def test_delete_is_idempotent(storage):
    await storage.put("delete-me.bin", b"x", "application/octet-stream")
    await storage.delete("delete-me.bin")
    await storage.delete("delete-me.bin")  # second call must not raise


@pytest.mark.asyncio
async def test_delete_then_get_raises(storage):
    await storage.put("gone.bin", b"data", "application/octet-stream")
    await storage.delete("gone.bin")
    with pytest.raises(StorageKeyNotFoundError):
        await storage.get("gone.bin")


@pytest.mark.asyncio
async def test_put_overwrites_existing(storage):
    await storage.put("overwrite.bin", b"v1", "application/octet-stream")
    await storage.put("overwrite.bin", b"v2", "application/octet-stream")
    assert await storage.get("overwrite.bin") == b"v2"


@pytest.mark.asyncio
async def test_presigned_upload_url_returns_string(storage):
    url = await storage.presigned_upload_url("upload/test.bin", expires_in=300)
    assert isinstance(url, str)
    assert len(url) > 0


@pytest.mark.asyncio
async def test_presigned_download_url_returns_string(storage):
    await storage.put("dl/test.bin", b"data", "application/octet-stream")
    url = await storage.presigned_download_url("dl/test.bin", expires_in=300)
    assert isinstance(url, str)
    assert len(url) > 0
