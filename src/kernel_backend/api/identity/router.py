from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.api.dependencies import get_session
from kernel_backend.api.identity.schemas import (
    CertificateResponse,
    GenerateIdentityRequest,
    PublicCertificateResponse,
)
from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.services.crypto_service import generate_keypair
from kernel_backend.infrastructure.database.repositories import IdentityRepository

router = APIRouter(tags=["identity"])


@router.delete("/me", status_code=204)
async def delete_my_identity(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete the authenticated user's identity to allow regeneration.

    Previously signed content remains valid — the public key and WID
    are embedded in the content. Only the identity record is removed.
    """
    user_id: str | None = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    repo = IdentityRepository(session)
    deleted = await repo.delete_by_author_id(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No identity found for this user")


@router.get("/me", response_model=PublicCertificateResponse)
async def get_my_identity(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> PublicCertificateResponse:
    """Return the public certificate fields for the authenticated user.

    No private key is ever returned. Returns 404 if no identity exists.
    """
    user_id: str | None = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    repo = IdentityRepository(session)
    cert = await repo.get_by_author_id(user_id)
    if cert is None:
        raise HTTPException(status_code=404, detail="No identity found for this user")

    return PublicCertificateResponse(
        author_id=cert.author_id,
        name=cert.name,
        institution=cert.institution,
        public_key_pem=cert.public_key_pem,
        created_at=cert.created_at,
    )


@router.post("/generate", status_code=201, response_model=CertificateResponse)
async def generate_identity(
    body: GenerateIdentityRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> CertificateResponse:
    """Generate an Ed25519 keypair for the authenticated user.

    author_id is bound to the caller's Stack Auth user_id — prevents one user
    from generating identities on behalf of another.
    Requires JWT authentication (user must belong to an organization).
    """
    user_id: str | None = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    org_id = getattr(request.state, "org_id", None)
    if org_id is None:
        raise HTTPException(status_code=403, detail="User is not a member of any organization")

    repo = IdentityRepository(session)

    # One key pair per user — return 409 if already exists
    existing = await repo.get_by_author_id(user_id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Key pair already exists for this user")

    private_pem, public_pem = generate_keypair()
    now = datetime.now(timezone.utc).isoformat()

    certificate = Certificate(
        name=body.name,
        institution=body.institution,
        author_id=user_id,  # bound to authenticated user, not derived from public key
        public_key_pem=public_pem,
        created_at=now,
    )

    await repo.create_with_org(certificate, org_id)

    return CertificateResponse(
        author_id=certificate.author_id,
        name=certificate.name,
        institution=certificate.institution,
        public_key_pem=certificate.public_key_pem,
        private_key_pem=private_pem,
        created_at=certificate.created_at,
        org_id=org_id,
    )
