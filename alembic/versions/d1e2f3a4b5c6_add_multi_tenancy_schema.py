"""add_multi_tenancy_schema

Revision ID: d1e2f3a4b5c6
Revises: a636765bb0f0
Create Date: 2026-03-16 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd1e2f3a4b5c6'
down_revision: Union[str, None] = 'a636765bb0f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'organizations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('pepper_v1', sa.String(length=64), nullable=True),
        sa.Column('current_pepper_version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'api_keys',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('org_id', sa.UUID(), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('key_prefix', sa.String(length=12), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash'),
    )
    op.create_index('idx_api_keys_org_id', 'api_keys', ['org_id'])
    op.create_index('idx_api_keys_key_hash', 'api_keys', ['key_hash'])

    op.create_table(
        'organization_members',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('org_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), server_default='member', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('org_id', 'user_id', name='uq_org_members_org_user'),
    )
    op.create_index('idx_org_members_org_id', 'organization_members', ['org_id'])
    op.create_index('idx_org_members_user_id', 'organization_members', ['user_id'])

    # Add org_id to identities
    op.add_column('identities', sa.Column('org_id', sa.UUID(), nullable=True))
    op.create_foreign_key('fk_identities_org_id', 'identities', 'organizations', ['org_id'], ['id'])
    op.create_index('idx_identities_org_id', 'identities', ['org_id'])

    # Add org_id to videos
    op.add_column('videos', sa.Column('org_id', sa.UUID(), nullable=True))
    op.create_foreign_key('fk_videos_org_id', 'videos', 'organizations', ['org_id'], ['id'])
    op.create_index('idx_videos_org_id', 'videos', ['org_id'])


def downgrade() -> None:
    op.drop_index('idx_videos_org_id', table_name='videos')
    op.drop_constraint('fk_videos_org_id', 'videos', type_='foreignkey')
    op.drop_column('videos', 'org_id')

    op.drop_index('idx_identities_org_id', table_name='identities')
    op.drop_constraint('fk_identities_org_id', 'identities', type_='foreignkey')
    op.drop_column('identities', 'org_id')

    op.drop_index('idx_org_members_user_id', table_name='organization_members')
    op.drop_index('idx_org_members_org_id', table_name='organization_members')
    op.drop_table('organization_members')

    op.drop_index('idx_api_keys_key_hash', table_name='api_keys')
    op.drop_index('idx_api_keys_org_id', table_name='api_keys')
    op.drop_table('api_keys')

    op.drop_table('organizations')