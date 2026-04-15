"""Add hash_prefix index to audio_fingerprints

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-14

Stores the top 16 bits of hash_hex as an indexed integer so public-verify
fingerprint lookups can prefilter candidates at the DB layer instead of
scanning every row into Python for a Hamming comparison.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "audio_fingerprints",
        sa.Column("hash_prefix", sa.Integer(), nullable=True),
    )
    # Backfill prefix (top 16 bits of the 64-bit hex hash) for existing rows.
    # SUBSTR+cast works on both Postgres and SQLite; ('x' || hex4)::bit(16)::int
    # is Postgres-only, so use a dialect-neutral expression.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "UPDATE audio_fingerprints "
            "SET hash_prefix = ('x' || substr(hash_hex, 1, 4))::bit(16)::int "
            "WHERE hash_prefix IS NULL"
        )
    else:
        # SQLite path: pull rows and compute in Python — only used by tests,
        # real deployments run Postgres.
        rows = bind.execute(
            sa.text("SELECT id, hash_hex FROM audio_fingerprints WHERE hash_prefix IS NULL")
        ).fetchall()
        for row_id, hash_hex in rows:
            prefix = int(hash_hex, 16) >> 48
            bind.execute(
                sa.text("UPDATE audio_fingerprints SET hash_prefix = :p WHERE id = :i"),
                {"p": prefix, "i": row_id},
            )
    op.create_index(
        "ix_audio_fingerprints_hash_prefix",
        "audio_fingerprints",
        ["hash_prefix"],
    )


def downgrade() -> None:
    op.drop_index("ix_audio_fingerprints_hash_prefix", table_name="audio_fingerprints")
    op.drop_column("audio_fingerprints", "hash_prefix")
