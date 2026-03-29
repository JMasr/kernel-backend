"""
Integration tests for org-scoped data isolation (Phase 6.B-1).

Verifies that:
1. Content signed by Org A cannot be found by Org B's fingerprint lookup
2. Identities created by Org A are not visible to Org B
3. Org A can list and retrieve its own identities
"""
from __future__ import annotations

import pytest

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.infrastructure.database.repositories import (
    IdentityRepository,
    VideoRepository,
)
from kernel_backend.infrastructure.database.models import Video
from kernel_backend.core.domain.watermark import VideoEntry
from tests.helpers.signing_defaults import DEFAULT_EMBEDDING_PARAMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cert(author_id: str) -> Certificate:
    from datetime import datetime, timezone

    return Certificate(
        author_id=author_id,
        name=f"Author {author_id}",
        institution="Test Corp",
        public_key_pem="-----BEGIN PUBLIC KEY-----\ntest\n-----END PUBLIC KEY-----",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_entry(content_id: str, author_id: str, org_id) -> VideoEntry:
    return VideoEntry(
        content_id=content_id,
        author_id=author_id,
        author_public_key="",
        active_signals=["fingerprint_audio"],
        rs_n=24,
        manifest_signature=b"\x00" * 64,
        embedding_params=DEFAULT_EMBEDDING_PARAMS,
        manifest_json="{}",
        org_id=org_id,
        signed_media_key=f"signed/{content_id}/output.mp4",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_org_identities_isolated(db_session):
    """Identities created by Org A are not visible to Org B."""
    from uuid import uuid4

    org_a = uuid4()
    org_b = uuid4()

    repo = IdentityRepository(db_session)

    cert_a = _make_cert("author_a_1")
    cert_b = _make_cert("author_b_1")

    await repo.create_with_org(cert_a, org_a)
    await repo.create_with_org(cert_b, org_b)

    identities_a = await repo.get_by_org_id(org_a)
    identities_b = await repo.get_by_org_id(org_b)

    assert len(identities_a) == 1
    assert identities_a[0].author_id == "author_a_1"

    assert len(identities_b) == 1
    assert identities_b[0].author_id == "author_b_1"

    # Org B cannot see Org A's identity
    b_author_ids = {c.author_id for c in identities_b}
    assert "author_a_1" not in b_author_ids


@pytest.mark.integration
async def test_fingerprint_match_isolated_by_org(db_session):
    """Fingerprint lookup for Org B returns no results for Org A's content."""
    from uuid import uuid4
    from kernel_backend.infrastructure.database.models import AudioFingerprint

    org_a = uuid4()
    org_b = uuid4()

    repo = VideoRepository(db_session)

    content_id = f"isolation-test-{uuid4().hex[:8]}"
    entry_a = _make_entry(content_id, "author_a_2", org_a)

    await repo.save_video(entry_a)

    # Store a fingerprint for the content
    fp_hex = "aabbccdd11223344"
    db_session.add(AudioFingerprint(
        content_id=content_id,
        time_offset_ms=0,
        hash_hex=fp_hex,
        is_original=True,
    ))
    await db_session.commit()

    # Org A can find it
    matches_a = await repo.match_fingerprints([fp_hex], max_hamming=0, org_id=org_a)
    assert len(matches_a) == 1
    assert matches_a[0].content_id == content_id

    # Org B cannot find it
    matches_b = await repo.match_fingerprints([fp_hex], max_hamming=0, org_id=org_b)
    assert len(matches_b) == 0


@pytest.mark.integration
async def test_content_listing_isolated_by_org(db_session):
    """list_by_org_id returns only the calling org's content."""
    from uuid import uuid4

    org_a = uuid4()
    org_b = uuid4()

    repo = VideoRepository(db_session)

    cid_a = f"list-test-a-{uuid4().hex[:8]}"
    cid_b = f"list-test-b-{uuid4().hex[:8]}"

    await repo.save_video(_make_entry(cid_a, "author_a_3", org_a))
    await repo.save_video(_make_entry(cid_b, "author_b_3", org_b))

    rows_a = await repo.list_by_org_id(org_a)
    rows_b = await repo.list_by_org_id(org_b)

    a_cids = {entry.content_id for entry, _, _ in rows_a}
    b_cids = {entry.content_id for entry, _, _ in rows_b}

    assert cid_a in a_cids
    assert cid_a not in b_cids

    assert cid_b in b_cids
    assert cid_b not in a_cids

    count_a = await repo.count_by_org_id(org_a)
    count_b = await repo.count_by_org_id(org_b)
    assert count_a >= 1
    assert count_b >= 1
