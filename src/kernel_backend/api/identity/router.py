import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.api.dependencies import get_session
from kernel_backend.api.identity.schemas import CertificateResponse, GenerateIdentityRequest
from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.services.crypto_service import generate_keypair
from kernel_backend.infrastructure.database.repositories import IdentityRepository

router = APIRouter(tags=["identity"])


@router.post("/generate", status_code=201, response_model=CertificateResponse)
async def generate_identity(
    body: GenerateIdentityRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> CertificateResponse:
    private_pem, public_pem = generate_keypair()

    # Derive a 16-char author_id: first 8 bytes of SHA-256 of public key PEM, hex-encoded
    author_id = hashlib.sha256(public_pem.encode()).hexdigest()[:16]

    now = datetime.now(timezone.utc).isoformat()

    certificate = Certificate(
        name=body.name,
        institution=body.institution,
        author_id=author_id,
        public_key_pem=public_pem,
        created_at=now,
    )

    org_id = getattr(request.state, "org_id", None)
    repo = IdentityRepository(session)
    if org_id is not None:
        await repo.create_with_org(certificate, org_id)
    else:
        await repo.create(certificate)

    return CertificateResponse(
        author_id=certificate.author_id,
        name=certificate.name,
        institution=certificate.institution,
        public_key_pem=certificate.public_key_pem,
        private_key_pem=private_pem,
        created_at=certificate.created_at,
        org_id=org_id,
    )
