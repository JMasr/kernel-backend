"""
StoragePort contract tests.

Runs a generic contract test suite against any StoragePort adapter.
To add a new adapter (e.g. R2), extend the `storage` fixture params —
all tests below must pass with zero changes to the test code.

Current adapters under test:
  local: LocalStorageAdapter (backed by tmp_path)
"""
import pytest

from kernel_backend.core.ports.storage import StorageKeyNotFoundError
from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter


@pytest.fixture(params=["local"])
def storage(request, tmp_path):
    if request.param == "local":
        return LocalStorageAdapter(base_path=tmp_path)


# ── Roundtrip ─────────────────────────────────────────────────────────────────

async def test_put_and_get_roundtrip(storage):
    """Stored bytes are returned byte-for-byte unchanged."""
    data = b"\x00\xff\x42binary-blob"
    await storage.put("test/file.bin", data, "application/octet-stream")
    assert await storage.get("test/file.bin") == data


async def test_put_overwrite_returns_latest(storage):
    """A second put on the same key replaces the first."""
    await storage.put("file.bin", b"v1", "application/octet-stream")
    await storage.put("file.bin", b"v2", "application/octet-stream")
    assert await storage.get("file.bin") == b"v2"


async def test_nested_key_creates_parent_dirs(storage):
    """Keys with slash separators must work; parent directories are created."""
    await storage.put("a/b/c/deep.bin", b"deep", "application/octet-stream")
    assert await storage.get("a/b/c/deep.bin") == b"deep"


async def test_empty_bytes_roundtrip(storage):
    """Zero-length payload is a valid value, not an error."""
    await storage.put("empty.bin", b"", "application/octet-stream")
    assert await storage.get("empty.bin") == b""


# ── Missing key errors ────────────────────────────────────────────────────────

async def test_get_missing_key_raises_domain_exception(storage):
    """Getting a non-existent key must raise StorageKeyNotFoundError."""
    with pytest.raises(StorageKeyNotFoundError):
        await storage.get("does/not/exist.bin")


async def test_get_missing_key_not_file_not_found(storage):
    """The raised exception must be the domain type, not a raw FileNotFoundError."""
    try:
        await storage.get("missing.bin")
        pytest.fail("Expected StorageKeyNotFoundError but nothing was raised")
    except StorageKeyNotFoundError:
        pass
    except FileNotFoundError:
        pytest.fail(
            "Adapter leaked FileNotFoundError — must raise StorageKeyNotFoundError "
            "(hexagonal boundary violation)"
        )


# ── Delete ────────────────────────────────────────────────────────────────────

async def test_delete_removes_key(storage):
    """After delete, a subsequent get must raise StorageKeyNotFoundError."""
    await storage.put("temp.bin", b"bye", "application/octet-stream")
    await storage.delete("temp.bin")
    with pytest.raises(StorageKeyNotFoundError):
        await storage.get("temp.bin")


async def test_delete_missing_key_is_idempotent(storage):
    """Deleting a key that was never stored must not raise."""
    await storage.delete("ghost.bin")


async def test_delete_twice_is_idempotent(storage):
    """Deleting the same key twice must not raise on the second call."""
    await storage.put("once.bin", b"x", "application/octet-stream")
    await storage.delete("once.bin")
    await storage.delete("once.bin")


# ── Presigned URLs ────────────────────────────────────────────────────────────

async def test_presigned_upload_url_returns_nonempty_string(storage):
    url = await storage.presigned_upload_url("upload.bin", expires_in=3600)
    assert isinstance(url, str) and len(url) > 0


async def test_presigned_download_url_returns_nonempty_string(storage):
    url = await storage.presigned_download_url("download.bin", expires_in=3600)
    assert isinstance(url, str) and len(url) > 0
