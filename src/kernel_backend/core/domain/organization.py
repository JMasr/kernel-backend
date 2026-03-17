from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass
class Organization:
    """Organization domain entity."""

    id: UUID
    name: str
    pepper_v1: Optional[str]
    current_pepper_version: int
    created_at: datetime


@dataclass
class APIKey:
    """API key domain entity."""

    id: UUID
    org_id: UUID
    key_hash: str
    key_prefix: str
    name: Optional[str]
    created_at: datetime
    last_used_at: Optional[datetime]
    is_active: bool


@dataclass
class OrganizationMember:
    """Organization member domain entity."""

    id: UUID
    org_id: UUID
    user_id: str
    role: str  # 'admin' | 'member'
    created_at: datetime