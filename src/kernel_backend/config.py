from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

_HMAC_SECRET_PLACEHOLDER = "change_in_production"


class Settings(BaseSettings):
    DATABASE_URL: str
    MIGRATION_DATABASE_URL: str
    KERNEL_SYSTEM_PEPPER: Annotated[str, Field(min_length=32)]
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str
    REDIS_SSL: bool = True
    STORAGE_BACKEND: Literal["local", "r2"] = "local"
    STORAGE_LOCAL_BASE_PATH: Path = Path("./data/media")
    STORAGE_HMAC_SECRET: str = _HMAC_SECRET_PLACEHOLDER
    ENV: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: str = ""
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    @model_validator(mode="after")
    def _enforce_production_secrets(self) -> "Settings":
        if self.ENV == "production" and self.STORAGE_HMAC_SECRET == _HMAC_SECRET_PLACEHOLDER:
            raise ValueError(
                "STORAGE_HMAC_SECRET must be set to a secure value in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return self

    # Neon Auth / Stack Auth (JWT session verification — optional if local admin is used)
    NEON_AUTH_URL: str = "https://api.stack-auth.com"
    NEON_AUTH_API_KEY: str = ""       # Stack Auth project ID
    NEON_AUTH_PUBLISHABLE_KEY: str = ""  # Stack Auth publishable client key (pck_...)
    NEON_AUTH_SECRET_SERVER_KEY: str = ""  # Stack Auth secret server key (ssk_...)

    # Local admin credentials (alternative to Stack Auth OAuth)
    ADMIN_EMAIL: str = ""             # Master admin email
    ADMIN_PASS: str = ""              # Master admin password (plaintext in .env — protect this file)
    JWT_SECRET: str = ""              # HS256 secret for local admin JWTs; generate: openssl rand -hex 32

    # Resend (transactional email)
    RESEND_API_KEY: str = ""
    RESEND_FROM_EMAIL: str = "noreply@notifications.kernelsecurity.tech"

    # Frontend
    FRONTEND_BASE_URL: str = "http://localhost:3000"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def system_pepper_bytes(self) -> bytes:
        return bytes.fromhex(self.KERNEL_SYSTEM_PEPPER)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
