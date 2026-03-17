from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings


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
    STORAGE_HMAC_SECRET: str = "change_in_production"
    ENV: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: str = ""

    # Neon Auth / Stack Auth (JWT session verification)
    NEON_AUTH_URL: str = "https://api.stack-auth.com"
    NEON_AUTH_API_KEY: str = ""       # Server-side Stack Auth API key
    ADMIN_USER_ID: str = ""           # User ID of the master admin

    # Resend (transactional email)
    RESEND_API_KEY: str = ""
    RESEND_FROM_EMAIL: str = "noreply@notifications.kernelsecurity.tech"

    # Frontend
    FRONTEND_BASE_URL: str = "http://localhost:3000"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def system_pepper_bytes(self) -> bytes:
        return bytes.fromhex(self.KERNEL_SYSTEM_PEPPER)
