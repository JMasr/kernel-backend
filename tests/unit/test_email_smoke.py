"""
Email adapter smoke tests.

Sends real emails via the Resend API using the verified domain
notifications.kernelsecurity.tech. Skipped automatically when
RESEND_API_KEY is not set in the environment.

Run manually:
    uv run python -m pytest tests/unit/test_email_smoke.py -v
"""
from __future__ import annotations

import pytest

from kernel_backend.config import Settings
from kernel_backend.infrastructure.email.resend_adapter import ResendEmailAdapter

_RECIPIENT = "moscaelectronica@gmail.com"
_ORG_NAME = "Kernel Security"
_FAKE_TOKEN = "00000000-0000-0000-0000-000000000001"
_FAKE_CONTENT_ID = "00000000-0000-0000-0000-000000000002"
_FAKE_FILENAME = "demo_video.mp4"


@pytest.fixture(scope="module")
def adapter() -> ResendEmailAdapter:
    settings = Settings()
    if not settings.RESEND_API_KEY:
        pytest.skip("RESEND_API_KEY not set — skipping email smoke tests")
    return ResendEmailAdapter(
        api_key=settings.RESEND_API_KEY,
        from_email=settings.RESEND_FROM_EMAIL,
        frontend_base_url=settings.FRONTEND_BASE_URL,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_send_invitation_email(adapter: ResendEmailAdapter) -> None:
    """Invitation email — contains org name and accept link."""
    await adapter.send_invitation(
        to_email=_RECIPIENT,
        org_name=_ORG_NAME,
        invite_token=_FAKE_TOKEN,
    )


async def test_send_job_complete_email(adapter: ResendEmailAdapter) -> None:
    """Job-complete notification — contains filename and content link."""
    await adapter.send_job_complete(
        to_email=_RECIPIENT,
        filename=_FAKE_FILENAME,
        content_id=_FAKE_CONTENT_ID,
    )


async def test_send_org_created_email(adapter: ResendEmailAdapter) -> None:
    """Org-created confirmation — sent to the admin who created the org."""
    await adapter.send_org_created(
        to_email=_RECIPIENT,
        org_name=_ORG_NAME,
    )


async def test_send_org_deleted_email(adapter: ResendEmailAdapter) -> None:
    """Org-deleted notification — sent when an org is hard-deleted."""
    await adapter.send_org_deleted(
        to_email=_RECIPIENT,
        org_name=_ORG_NAME,
    )
