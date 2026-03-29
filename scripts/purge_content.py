#!/usr/bin/env python3
"""
Content purge tool — delete signed media entries from registry and storage.

Removes all watermark-related data for signed content:
  DB  : videos, audio_fingerprints, video_segments, embedding_recipes,
        pilot_tone_index, transparency_log_entries
  Disk: storage_key file (original) + signed_storage_key file (signed copy)

Tables NOT touched: organizations, api_keys, organization_members,
                    identities, invitations, transparency_log_roots.

Usage:
    # Preview what would be deleted (safe, no writes)
    uv run python scripts/purge_content.py

    # Preview for a single org
    uv run python scripts/purge_content.py --org-id <uuid>

    # Execute
    uv run python scripts/purge_content.py --yes

    # Execute for a single org
    uv run python scripts/purge_content.py --org-id <uuid> --yes
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import delete as sql_delete, func, select, text

from kernel_backend.config import get_settings
from kernel_backend.infrastructure.database.models import (
    AudioFingerprint,
    EmbeddingRecipe,
    PilotToneIndex,
    TransparencyLogEntry,
    Video,
    VideoSegment,
)
from kernel_backend.infrastructure.database.session import make_engine, make_session_factory

# Child tables that reference videos.content_id (delete before parent)
_CHILD_TABLES = [AudioFingerprint, VideoSegment, EmbeddingRecipe, PilotToneIndex]


def _hr() -> None:
    print("─" * 64)


async def _count_rows(session, model) -> int:
    result = await session.execute(select(func.count()).select_from(model))
    return result.scalar_one()


async def run(org_id: UUID | None, execute: bool) -> None:
    settings = get_settings()
    engine = make_engine(settings.MIGRATION_DATABASE_URL)
    factory = make_session_factory(engine)

    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"\nKernel Security — Content Purge Tool  [{mode}]")
    if not execute:
        print("  Pass --yes to actually delete.\n")
    _hr()

    async with factory() as session:
        # ── 1. Fetch only the columns we need (avoids schema-drift errors) ──
        stmt = select(
            Video.content_id,
            Video.status,
            Video.author_id,
            Video.storage_key,
            Video.signed_storage_key,
            Video.org_id,
        )
        if org_id is not None:
            stmt = stmt.where(Video.org_id == org_id)
        rows = (await session.execute(stmt)).all()

        if not rows:
            scope = f"org {org_id}" if org_id else "all orgs"
            print(f"No signed content found for {scope}.")
            await engine.dispose()
            return

        print(f"Found {len(rows)} content entry/entries to purge:\n")
        print(f"  {'content_id':<38}  {'status':<8}  {'author_id':<32}  storage keys")
        print(f"  {'-'*38}  {'-'*8}  {'-'*32}  {'-'*30}")
        for v in rows:
            sk = v.storage_key or "(none)"
            ssk = v.signed_storage_key or "(none)"
            print(f"  {v.content_id:<38}  {v.status:<8}  {str(v.author_id)[:32]:<32}  {sk} / {ssk}")

        _hr()

        if not execute:
            # Show what child row counts would be affected
            print("\nChild rows that would be deleted:")
            for child_model in _CHILD_TABLES + [TransparencyLogEntry]:
                count = await _count_rows(session, child_model)
                print(f"  {child_model.__tablename__:<30}  {count} total rows")
            if org_id:
                print(f"\n  (counts above are totals — only rows for org {org_id} will be deleted)")
            print("\nRun with --yes to execute.\n")
            await engine.dispose()
            return

        # ── 2. Execute deletions ─────────────────────────────────────────────
        print("\nDeleting...")

        storage_base = settings.STORAGE_LOCAL_BASE_PATH
        storage_backend = settings.STORAGE_BACKEND
        storage_deleted = 0
        storage_missing = 0
        storage_skipped = 0

        for v in rows:
            cid = v.content_id
            print(f"\n  [{cid[:16]}...]")

            # 2a. Delete storage files
            if storage_backend == "local":
                for key in (v.storage_key, v.signed_storage_key):
                    if not key:
                        continue
                    fpath = storage_base / key
                    if fpath.exists():
                        fpath.unlink()
                        print(f"    deleted file: {fpath}")
                        storage_deleted += 1
                    else:
                        print(f"    file not found (skipped): {fpath}")
                        storage_missing += 1
            else:
                # R2 / S3 — not yet implemented; log keys for manual cleanup
                for key in (v.storage_key, v.signed_storage_key):
                    if key:
                        print(f"    [R2] storage key to delete manually: {key}")
                        storage_skipped += 1

            # 2b. Delete transparency log entries for this content
            result = await session.execute(
                sql_delete(TransparencyLogEntry).where(
                    TransparencyLogEntry.content_id == cid
                ).returning(text("1"))
            )
            tlog_deleted = len(result.all())
            if tlog_deleted:
                print(f"    deleted {tlog_deleted} transparency_log_entries")

            # 2c. Delete child tables
            for child_model in _CHILD_TABLES:
                result = await session.execute(
                    sql_delete(child_model).where(
                        child_model.content_id == cid
                    ).returning(text("1"))
                )
                n = len(result.all())
                if n:
                    print(f"    deleted {n:>4} rows from {child_model.__tablename__}")

            # 2d. Delete parent Video row
            await session.execute(
                sql_delete(Video).where(Video.content_id == cid)
            )
            print(f"    deleted video row")

        await session.commit()

        _hr()
        print(f"\nDone.")
        print(f"  Content entries purged : {len(rows)}")
        if storage_backend == "local":
            print(f"  Storage files deleted  : {storage_deleted}")
            if storage_missing:
                print(f"  Storage files missing  : {storage_missing}  (already gone)")
        else:
            print(f"  R2 keys to delete manually: {storage_skipped}")
        print()

    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge signed content from registry and storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--org-id",
        metavar="UUID",
        default=None,
        help="Limit purge to a single organization (omit to purge all).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Execute the purge. Without this flag, only a preview is shown.",
    )
    args = parser.parse_args()

    org_id: UUID | None = None
    if args.org_id:
        try:
            org_id = UUID(args.org_id)
        except ValueError:
            print(f"Error: --org-id '{args.org_id}' is not a valid UUID.", file=sys.stderr)
            sys.exit(1)

    asyncio.run(run(org_id=org_id, execute=args.yes))


if __name__ == "__main__":
    main()
