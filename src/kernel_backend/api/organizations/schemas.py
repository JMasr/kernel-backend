from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class OrganizationResponse(BaseModel):
    org_id: UUID
    name: str
    created_at: datetime


class CreateApiKeyRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    scopes: list[str] = Field(default=["sign", "verify"])
    expires_at: Optional[datetime] = Field(None, description="ISO 8601 datetime. None = never expires.")

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, v: list[str]) -> list[str]:
        allowed = {"sign", "verify"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Invalid scopes: {invalid}. Allowed: {allowed}")
        if not v:
            raise ValueError("scopes must contain at least one value")
        return v


class ApiKeyResponse(BaseModel):
    key_id: UUID
    key_prefix: str
    name: Optional[str]
    plaintext_key: Optional[str] = None
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    scopes: list[str] = Field(default=["sign", "verify"])
    expires_at: Optional[datetime] = None


class ApiKeyListResponse(BaseModel):
    items: list[ApiKeyResponse]
    total: int
    page: int
    total_pages: int


class UpdateApiKeyRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None


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