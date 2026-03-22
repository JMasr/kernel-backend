from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class GenerateIdentityRequest(BaseModel):
    name: str = Field(min_length=1)
    institution: str = Field(min_length=1)


class PublicCertificateResponse(BaseModel):
    """Public certificate fields only — no private key."""
    author_id: str
    name: str
    institution: str
    public_key_pem: str
    created_at: str


class CertificateResponse(BaseModel):
    author_id: str
    name: str
    institution: str
    public_key_pem: str
    private_key_pem: str  # returned once, never stored, never logged
    created_at: str
    org_id: UUID | None = None
