"""
Shared fake infrastructure for pipeline and integration tests.

FakeStorage: in-memory dict-based StoragePort.
FakeRegistry: in-memory RegistryPort with real Hamming matching.
"""
from __future__ import annotations

from uuid import UUID

from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.core.ports.storage import StoragePort


class FakeStorage(StoragePort):
    """Dict-based in-memory storage. Stores bytes by key."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str = "") -> None:
        self._store[key] = data

    async def get(self, key: str) -> bytes:
        return self._store[key]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def presigned_upload_url(self, key: str, expires_in: int = 3600) -> str:
        return f"fake://{key}"

    async def presigned_download_url(self, key: str, expires_in: int = 3600) -> str:
        return f"fake://{key}"


class FakeRegistry(RegistryPort):
    """In-memory registry with real Hamming-distance fingerprint matching."""

    def __init__(self) -> None:
        self._videos: dict[str, VideoEntry] = {}
        self._segments: dict[str, list[SegmentFingerprint]] = {}

    async def save_video(self, entry: VideoEntry) -> None:
        self._videos[entry.content_id] = entry

    async def get_by_content_id(self, content_id: str) -> VideoEntry | None:
        return self._videos.get(content_id)

    async def get_valid_candidates(self) -> list[VideoEntry]:
        return [e for e in self._videos.values() if e.status == "VALID"]

    async def save_segments(
        self, content_id: str, segments: list[SegmentFingerprint], is_original: bool
    ) -> None:
        existing = self._segments.get(content_id, [])
        self._segments[content_id] = existing + list(segments)

    async def match_fingerprints(
        self, hashes: list[str], max_hamming: int = 10, org_id: UUID | None = None
    ) -> list[VideoEntry]:
        from kernel_backend.engine.audio.fingerprint import (
            hamming_distance,
        )

        matches: set[str] = set()
        for query_hash in hashes:
            for content_id, stored_fps in self._segments.items():
                for sfp in stored_fps:
                    if hamming_distance(query_hash, sfp.hash_hex) <= max_hamming:
                        matches.add(content_id)
        return [self._videos[cid] for cid in matches if cid in self._videos]
