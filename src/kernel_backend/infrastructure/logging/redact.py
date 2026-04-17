"""Secret-scrubbing structlog processor.

Runs on every structured log event and every stdlib log record that the
ProcessorFormatter routes through the structlog chain. Two defences:

1. Key denylist — any event-dict key matching a known secret name has its
   value replaced with a fixed redaction marker.
2. Value regex — string values are scanned for recognisable secret shapes
   (`krnl_` API keys, JWTs, PEM blocks) and rewritten with the marker.

Recursion is capped at ``_MAX_DEPTH`` so a rogue deeply-nested payload cannot
turn a log line into a CPU hog.
"""
from __future__ import annotations

import re
from typing import Any, MutableMapping

REDACTED = "***REDACTED***"
_MAX_DEPTH = 3

_DENYLIST_KEYS: frozenset[str] = frozenset(
    k.lower()
    for k in (
        # Signing payloads
        "cert_json",
        "certificate_json",
        "private_key_pem",
        "private_key",
        "pkey",
        # Cryptographic material
        "pepper",
        "org_pepper_hex",
        "kernel_system_pepper",
        "system_pepper",
        # Auth / passwords
        "password",
        "pass",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "x-stack-access-token",
        "x-stack-secret-server-key",
        "api_key",
        "apikey",
        # Server secrets
        "neon_auth_secret_server_key",
        "storage_hmac_secret",
        "jwt_secret",
        "resend_api_key",
        "s3_secret_access_key",
        "minio_root_password",
        "redis_password",
    )
)

_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Kernel API keys — ``krnl_`` prefix plus at least 16 URL-safe chars.
    re.compile(r"krnl_[A-Za-z0-9_-]{16,}"),
    # Generic JWT (three base64url segments separated by dots).
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    # PEM-armoured blocks of any kind.
    re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]+?-----END [A-Z ]+-----"),
)


def scrub_string(value: str) -> str:
    for pattern in _VALUE_PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value


def _scrub(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH:
        return value
    if isinstance(value, str):
        return scrub_string(value)
    if isinstance(value, dict):
        return {k: _scrub_entry(k, v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        scrubbed = [_scrub(v, depth + 1) for v in value]
        return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


def _scrub_entry(key: Any, value: Any, depth: int) -> Any:
    if isinstance(key, str) and key.lower() in _DENYLIST_KEYS:
        return REDACTED
    return _scrub(value, depth)


class RedactSecretsProcessor:
    """structlog processor: scrub secrets from every log event."""

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        return {k: _scrub_entry(k, v, 1) for k, v in event_dict.items()}


def scrub_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Public helper for callers (e.g. Sentry ``before_send``) that need the same scrub."""
    return {k: _scrub_entry(k, v, 1) for k, v in payload.items()}
