from __future__ import annotations

import pytest

from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry
from kernel_backend.infrastructure.database.repositories import VideoRepository
from tests.helpers.signing_defaults import DEFAULT_EMBEDDING_PARAMS

ENTRY = VideoEntry(
    content_id="test-content-001",
    author_id="author-test-id",
    author_public_key="-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----\n",
    active_signals=["wid_audio", "fingerprint_audio"],
    rs_n=32,
    manifest_signature=b"\x00" * 64,
    embedding_params=DEFAULT_EMBEDDING_PARAMS,
    schema_version=2,
    status="VALID",
)


async def test_save_and_retrieve_video(db_session) -> None:
    repo = VideoRepository(db_session)
    await repo.save_video(ENTRY)
    retrieved = await repo.get_by_content_id(ENTRY.content_id)
    assert retrieved is not None
    assert retrieved.content_id == ENTRY.content_id
    assert retrieved.author_id == ENTRY.author_id
    assert retrieved.rs_n == ENTRY.rs_n
    assert retrieved.manifest_signature == ENTRY.manifest_signature
    assert retrieved.active_signals == ENTRY.active_signals
    assert retrieved.status == ENTRY.status


async def test_get_nonexistent_returns_none(db_session) -> None:
    repo = VideoRepository(db_session)
    result = await repo.get_by_content_id("does-not-exist")
    assert result is None


async def test_save_segments_and_match(db_session) -> None:
    repo = VideoRepository(db_session)
    await repo.save_video(ENTRY)

    segments = [
        SegmentFingerprint(time_offset_ms=0,    hash_hex="abcd1234ef567890"),
        SegmentFingerprint(time_offset_ms=2000, hash_hex="1234abcd56789012"),
    ]
    await repo.save_segments(ENTRY.content_id, segments, is_original=True)

    # Exact match → found
    matches = await repo.match_fingerprints(["abcd1234ef567890"], max_hamming=0)
    assert len(matches) == 1
    assert matches[0].content_id == ENTRY.content_id

    # No match (wrong hash, max_hamming=0)
    no_matches = await repo.match_fingerprints(["0000000000000000"], max_hamming=0)
    assert len(no_matches) == 0
