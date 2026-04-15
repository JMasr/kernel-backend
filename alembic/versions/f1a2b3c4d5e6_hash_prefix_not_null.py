"""Tighten audio_fingerprints.hash_prefix to NOT NULL

Revision ID: f1a2b3c4d5e6
Revises: e5f6a7b8c9d0
Create Date: 2026-04-15

The previous migration (e5f6a7b8c9d0) added hash_prefix as nullable so the
deploy would not block on an in-flight backfill, and an ORM before_insert
listener (models._audio_fingerprint_autofill_hash_prefix) keeps new rows
filled.  This migration promotes the invariant from application-enforced to
schema-enforced so that any write path that bypasses the ORM (bulk insert,
raw SQL, dump/restore) fails at the DB instead of silently poisoning the
Hamming prefilter.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Defense-in-depth backfill: e5f6a7b8c9d0 already filled existing rows and
    # the ORM listener fills new ones.  Re-run the same logic in case any
    # edge-case write (manual psql, dump/restore, Core-level insert) left a
    # NULL since then.  Idempotent: a no-op when there are no NULL rows.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "UPDATE audio_fingerprints "
            "SET hash_prefix = ('x' || substr(hash_hex, 1, 4))::bit(16)::int "
            "WHERE hash_prefix IS NULL"
        )
    else:
        rows = bind.execute(
            sa.text(
                "SELECT id, hash_hex FROM audio_fingerprints "
                "WHERE hash_prefix IS NULL"
            )
        ).fetchall()
        for row_id, hash_hex in rows:
            bind.execute(
                sa.text(
                    "UPDATE audio_fingerprints SET hash_prefix = :p WHERE id = :i"
                ),
                {"p": int(hash_hex, 16) >> 48, "i": row_id},
            )

    # batch_alter_table handles SQLite's lack of ALTER COLUMN by recreating
    # the table transparently.  On Postgres it compiles to a plain
    # ALTER TABLE ... ALTER COLUMN ... SET NOT NULL.  The existing index
    # ix_audio_fingerprints_hash_prefix is preserved in both cases.
    with op.batch_alter_table("audio_fingerprints") as batch:
        batch.alter_column(
            "hash_prefix",
            existing_type=sa.Integer(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("audio_fingerprints") as batch:
        batch.alter_column(
            "hash_prefix",
            existing_type=sa.Integer(),
            nullable=True,
        )
