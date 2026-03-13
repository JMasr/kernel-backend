from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CryptographicManifest:
    content_id: str
    content_hash_sha256: str    # hex string from SHA256.hexdigest()
    fingerprints_audio: list[str]   # list of 16-char hex strings
    fingerprints_video: list[str]   # list of 16-char hex strings
    author_id: str
    author_public_key: str      # PEM string
    created_at: str
    schema_version: int = 2

    def __post_init__(self) -> None:
        try:
            datetime.fromisoformat(self.created_at)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"created_at must be a valid ISO 8601 string, got {self.created_at!r}"
            ) from exc
