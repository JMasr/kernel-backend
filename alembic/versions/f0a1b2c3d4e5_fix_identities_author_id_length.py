"""fix_identities_author_id_length

Revision ID: f0a1b2c3d4e5
Revises: e7f8a9b0c1d2
Create Date: 2026-03-19
"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "f0a1b2c3d4e5"
down_revision: Union[str, None] = "e7f8a9b0c1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # author_id was created as VARCHAR(16) in baseline schema — too short for
    # email addresses and Stack Auth user IDs (UUIDs, ~36 chars).
    # Alter to unbounded VARCHAR to match the SQLAlchemy model (String).
    op.alter_column(
        "identities",
        "author_id",
        type_=sa.String(),
        existing_type=sa.String(length=16),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "identities",
        "author_id",
        type_=sa.String(length=16),
        existing_type=sa.String(),
        existing_nullable=False,
    )
