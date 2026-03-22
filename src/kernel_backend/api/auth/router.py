"""
Local admin authentication.

POST /auth/login — validates ADMIN_EMAIL + ADMIN_PASS from .env,
returns a signed HS256 JWT usable as a Bearer token.

This is the fallback when Stack Auth / Google OAuth redirect links
are not available (e.g., non-certified providers).
"""
from __future__ import annotations

import secrets
import time

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kernel_backend.config import Settings

router = APIRouter(prefix="/auth", tags=["auth"])

_TOKEN_TTL_SECONDS = 60 * 60 * 8  # 8 hours


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    access_token: str


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(body: RefreshRequest) -> LoginResponse:
    """
    Exchange a valid (non-expired) local admin JWT for a fresh one.

    No DB interaction — purely decodes + re-signs with a new exp.
    """
    settings = Settings()

    if not settings.JWT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Local auth not configured. Set JWT_SECRET in .env.",
        )

    try:
        payload = jwt.decode(body.access_token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    now = int(time.time())
    new_payload = {
        "sub": payload.get("sub", ""),
        "email": payload.get("email", ""),
        "is_admin": payload.get("is_admin", False),
        "iat": now,
        "exp": now + _TOKEN_TTL_SECONDS,
    }
    token = jwt.encode(new_payload, settings.JWT_SECRET, algorithm="HS256")
    return LoginResponse(access_token=token)


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    """
    Authenticate as the master admin using credentials from .env.

    Returns a signed JWT that the client sends as:
        Authorization: Bearer <token>

    Requires JWT_SECRET, ADMIN_EMAIL, and ADMIN_PASS to be set in .env.
    """
    settings = Settings()

    if not settings.JWT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Local auth not configured. Set JWT_SECRET in .env.",
        )
    if not settings.ADMIN_EMAIL or not settings.ADMIN_PASS:
        raise HTTPException(
            status_code=503,
            detail="Local admin credentials not configured. Set ADMIN_EMAIL and ADMIN_PASS in .env.",
        )

    # Constant-time comparison to prevent timing attacks
    email_ok = secrets.compare_digest(body.email, settings.ADMIN_EMAIL)
    pass_ok = secrets.compare_digest(body.password, settings.ADMIN_PASS)

    if not (email_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    now = int(time.time())
    payload = {
        "sub": settings.ADMIN_EMAIL,
        "email": settings.ADMIN_EMAIL,
        "is_admin": True,
        "iat": now,
        "exp": now + _TOKEN_TTL_SECONDS,
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
    return LoginResponse(access_token=token)
