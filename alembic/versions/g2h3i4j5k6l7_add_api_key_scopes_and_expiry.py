"""Add scopes and expires_at to api_keys table

Revision ID: g2h3i4j5k6l7
Revises: f1a2b3c4d5e6
Create Date: 2026-05-04

Adds:
- scopes (JSON NOT NULL, default ["sign", "verify"]) — per-key permission control
- expires_at (TIMESTAMPTZ NULL) — optional key expiry; NULL means never expires

Both columns are additive and backward-compatible. Old application code ignores them;
new code reads them safely (with guards for NULL scopes on pre-migration rows).
"""
from alembic import op
import sqlalchemy as sa

revision = "g2h3i4j5k6l7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column(
            "scopes",
            sa.JSON(),
            nullable=False,
            server_default='["sign", "verify"]',
        ),
    )
    op.add_column(
        "api_keys",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "expires_at")
    op.drop_column("api_keys", "scopes")
