from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class OrganizationResponse(BaseModel):
    org_id: UUID
    name: str
    created_at: datetime


class CreateApiKeyRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)


class ApiKeyResponse(BaseModel):
    key_id: UUID
    key_prefix: str
    name: Optional[str]
    plaintext_key: str
    created_at: datetime


class UserOrganizationResponse(BaseModel):
    org_id: UUID
    name: str
    created_at: datetime


class UpdateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class PaginatedOrganizationsResponse(BaseModel):
    items: list[OrganizationResponse]
    total: int
    page: int
    total_pages: int


class OrganizationMemberResponse(BaseModel):
    id: UUID
    org_id: UUID
    user_id: str
    role: str
    created_at: datetime


class PaginatedMembersResponse(BaseModel):
    items: list[OrganizationMemberResponse]
    total: int
    page: int
    total_pages: int


class AddMemberRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    role: str = Field("member", pattern="^(admin|member)$")


class UpdateMemberRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(admin|member)$")