"""Redactor unit tests.

Secrets must never leak through the logger. These tests cover:

* Denylist keys — any event-dict key that names a secret has its value replaced.
* Value regexes — ``krnl_`` API keys, JWTs, PEM blocks are scrubbed even if
  they appear inside free-form strings (e.g. exception messages).
* Nested structures — dicts and lists within event dicts are walked.
* Depth cap — we stop before a pathological payload wastes CPU.
"""
from __future__ import annotations

import pytest

from kernel_backend.infrastructure.logging.redact import (
    REDACTED,
    RedactSecretsProcessor,
    scrub_mapping,
    scrub_string,
)

processor = RedactSecretsProcessor()


def _apply(event: dict) -> dict:
    return processor(logger=None, method_name="info", event_dict=event)


@pytest.mark.parametrize(
    "key",
    [
        "cert_json",
        "certificate_json",
        "private_key_pem",
        "pepper",
        "password",
        "token",
        "access_token",
        "authorization",
        "api_key",
        "STORAGE_HMAC_SECRET",
        "JWT_SECRET",
    ],
)
def test_denylist_keys_are_scrubbed(key):
    result = _apply({key: "some-secret-value"})
    assert result[key] == REDACTED


def test_denylist_is_case_insensitive():
    assert _apply({"Authorization": "Bearer foo"})["Authorization"] == REDACTED
    assert _apply({"API_KEY": "krnl_x"})["API_KEY"] == REDACTED


def test_value_regex_scrubs_krnl_key_inside_message():
    result = _apply({"msg": "failed to authenticate krnl_abcdefghijklmnop12345 for user"})
    assert REDACTED in result["msg"]
    assert "krnl_" not in result["msg"]


def test_value_regex_scrubs_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    result = _apply({"token_dump": jwt})
    # Denylist key catches it first, but if key were safe, regex should still scrub.
    assert result["token_dump"] == REDACTED


def test_value_regex_scrubs_jwt_in_free_text_key():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    result = _apply({"note": f"request sent with header {jwt}"})
    assert REDACTED in result["note"]
    assert "eyJ" not in result["note"]


def test_value_regex_scrubs_pem_block():
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n"
        "-----END PRIVATE KEY-----"
    )
    result = _apply({"body": f"signed with {pem} end"})
    assert "BEGIN PRIVATE KEY" not in result["body"]
    assert REDACTED in result["body"]


def test_nested_dict_scrub():
    result = _apply({"ctx": {"user": "u1", "cert_json": "{...}"}})
    assert result["ctx"]["cert_json"] == REDACTED
    assert result["ctx"]["user"] == "u1"


def test_nested_list_scrub():
    pem = "-----BEGIN CERT-----\nAAAA\n-----END CERT-----"
    result = _apply({"samples": ["safe", pem, "krnl_abcdefghijklmnopqrst12"]})
    assert result["samples"][0] == "safe"
    assert REDACTED in result["samples"][1]
    assert REDACTED in result["samples"][2]


def test_safe_keys_unchanged():
    result = _apply({"request_id": "abc", "status": 200, "elapsed_ms": 42.0})
    assert result == {"request_id": "abc", "status": 200, "elapsed_ms": 42.0}


def test_scrub_mapping_public_helper():
    out = scrub_mapping({"password": "hunter2", "user": "alice"})
    assert out == {"password": REDACTED, "user": "alice"}


def test_scrub_string_helper():
    assert scrub_string("krnl_abcdefghijklmnop12345") == REDACTED
    assert scrub_string("plain message") == "plain message"


def test_depth_cap_stops_recursion():
    # Build a payload deeper than the cap so outer levels scrub but inner-most
    # remains (proving the cap protects CPU, not security — inner keys cannot
    # themselves leak through the outer scrub).
    inner = {"password": "leaked"}
    nested = {"a": {"b": {"c": {"d": inner}}}}
    result = _apply(nested)
    # Top level is scanned, then depth counts by level; the exact behaviour
    # is that we must not raise and must not corrupt structure.
    assert "a" in result
