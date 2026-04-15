from __future__ import annotations

from uuid import uuid4

import pytest

from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry
from kernel_backend.infrastructure.database.models import AudioFingerprint
from kernel_backend.infrastructure.database.repositories import (
    VideoRepository,
    _hash_prefix,
)
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


async def test_save_segments_populates_hash_prefix(db_session) -> None:
    """save_segments must write hash_prefix so the DB prefilter has data to index."""
    from sqlalchemy import select

    repo = VideoRepository(db_session)
    await repo.save_video(ENTRY)
    await repo.save_segments(
        ENTRY.content_id,
        [SegmentFingerprint(time_offset_ms=0, hash_hex="deadbeefcafef00d")],
        is_original=True,
    )

    row = (
        await db_session.execute(
            select(AudioFingerprint).where(AudioFingerprint.content_id == ENTRY.content_id)
        )
    ).scalar_one()
    assert row.hash_prefix == _hash_prefix("deadbeefcafef00d")
    assert row.hash_prefix == 0xDEAD


async def test_match_fingerprints_within_prefix_radius(db_session) -> None:
    """A stored hash that differs from the query by 1 bit in the prefix + 1 bit
    in the tail (total Hamming 2, prefix Hamming 1) must still be matched — this
    is the whole point of querying the prefix with Hamming radius >= 1."""
    repo = VideoRepository(db_session)
    await repo.save_video(ENTRY)

    # Flip bit 48 (first bit of prefix) and bit 0 (last bit of tail)
    stored = f"{0xDEADBEEFCAFEF00D ^ (1 << 48) ^ 1:016x}"
    query = "deadbeefcafef00d"

    await repo.save_segments(
        ENTRY.content_id,
        [SegmentFingerprint(time_offset_ms=0, hash_hex=stored)],
        is_original=True,
    )

    matches = await repo.match_fingerprints([query], max_hamming=5)
    assert len(matches) == 1
    assert matches[0].content_id == ENTRY.content_id


async def test_match_fingerprints_prefilters_out_distant_rows(db_session) -> None:
    """Rows whose prefix is far from every query hash prefix must not even be
    loaded — prove this by writing a row whose hash_hex is actually close to
    the query (Hamming 2 in the tail) but whose hash_prefix column is set to a
    value many bits away, and confirming it is not returned.
    """
    repo = VideoRepository(db_session)
    await repo.save_video(ENTRY)

    query = "deadbeefcafef00d"
    # Persist a legitimate row, then stomp its hash_prefix to a distant value
    # to simulate a row the prefilter should prune.
    await repo.save_segments(
        ENTRY.content_id,
        [SegmentFingerprint(time_offset_ms=0, hash_hex=query)],
        is_original=True,
    )
    # Flip every bit in the prefix → Hamming 16 from query prefix, far outside
    # our radius-1 filter.  Use a Core-level UPDATE because the ORM
    # before_update listener autofills hash_prefix from hash_hex and would
    # revert any stomp done via attribute assignment + session.commit().
    await db_session.execute(
        AudioFingerprint.__table__.update()
        .where(AudioFingerprint.content_id == ENTRY.content_id)
        .values(hash_prefix=_hash_prefix(query) ^ 0xFFFF)
    )
    await db_session.commit()

    matches = await repo.match_fingerprints([query], max_hamming=10)
    # Row exists and its hash_hex is an exact match, but the prefilter pruned
    # it — this is the tradeoff the prefilter makes to avoid loading every row.
    assert matches == []


async def test_match_fingerprints_empty_query(db_session) -> None:
    """Empty query list must short-circuit — no DB scan, no matches."""
    repo = VideoRepository(db_session)
    await repo.save_video(ENTRY)
    await repo.save_segments(
        ENTRY.content_id,
        [SegmentFingerprint(time_offset_ms=0, hash_hex="abcd1234ef567890")],
        is_original=True,
    )
    assert await repo.match_fingerprints([], max_hamming=10) == []


async def test_hash_prefix_is_not_null_at_db_level(db_session) -> None:
    """Core-level insert with NULL hash_prefix must fail at the DB.

    The ORM event listener covers session.add() paths, but a bulk_insert_mappings,
    raw SQL, or Table.insert() bypasses it.  The schema NOT NULL constraint is
    the last line of defense — this test locks that invariant in.
    """
    import sqlalchemy.exc

    repo = VideoRepository(db_session)
    await repo.save_video(ENTRY)

    # Core-level insert bypasses the ORM before_insert listener that normally
    # autofills hash_prefix.  Hitting the DB with a NULL must raise.
    stmt = AudioFingerprint.__table__.insert().values(
        content_id=ENTRY.content_id,
        time_offset_ms=0,
        hash_hex="0" * 16,
        hash_prefix=None,
        is_original=True,
    )
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        await db_session.execute(stmt)


async def test_match_fingerprints_ranks_by_match_count(db_session) -> None:
    """Content with more matching segments should rank first."""
    repo = VideoRepository(db_session)

    entry_a = VideoEntry(**{**ENTRY.__dict__, "content_id": "content-many-matches"})
    entry_b = VideoEntry(**{**ENTRY.__dict__, "content_id": "content-few-matches"})
    await repo.save_video(entry_a)
    await repo.save_video(entry_b)

    q1, q2, q3 = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc"

    # entry_a has 3 matching segments, entry_b has 1
    await repo.save_segments(
        entry_a.content_id,
        [
            SegmentFingerprint(time_offset_ms=0, hash_hex=q1),
            SegmentFingerprint(time_offset_ms=1000, hash_hex=q2),
            SegmentFingerprint(time_offset_ms=2000, hash_hex=q3),
        ],
        is_original=True,
    )
    await repo.save_segments(
        entry_b.content_id,
        [SegmentFingerprint(time_offset_ms=0, hash_hex=q1)],
        is_original=True,
    )

    matches = await repo.match_fingerprints([q1, q2, q3], max_hamming=0)
    assert [m.content_id for m in matches] == [entry_a.content_id, entry_b.content_id]
