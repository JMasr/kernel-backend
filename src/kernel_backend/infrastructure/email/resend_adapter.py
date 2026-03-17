"""
Email adapter using Resend.

Sends transactional emails for:
- Organization invitations
- Async job completion notifications
- Organization lifecycle events (created, deleted)
"""
from __future__ import annotations

from pathlib import Path

import resend
from jinja2 import Template


class ResendEmailAdapter:
    """Sends templated emails via the Resend API."""

    def __init__(self, api_key: str, from_email: str, frontend_base_url: str) -> None:
        resend.api_key = api_key
        self._from_email = from_email
        self._frontend_base_url = frontend_base_url
        self._templates_dir = Path(__file__).parent / "templates"

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    async def send_invitation(
        self,
        to_email: str,
        org_name: str,
        invite_token: str,
    ) -> None:
        """Send invitation email with accept link."""
        html = self._render("invitation.html", {
            "org_name": org_name,
            "invite_link": f"{self._frontend_base_url}/invite/{invite_token}",
        })
        self._send(
            to=to_email,
            subject=f"Invitación a {org_name} - Kernel Security",
            html=html,
        )

    async def send_job_complete(
        self,
        to_email: str,
        filename: str,
        content_id: str,
    ) -> None:
        """Notify user that an async signing job completed."""
        html = self._render("job_complete.html", {
            "filename": filename,
            "content_link": f"{self._frontend_base_url}/dashboard/content/{content_id}",
        })
        self._send(
            to=to_email,
            subject=f"Tu video {filename} ha sido firmado",
            html=html,
        )

    async def send_org_created(self, to_email: str, org_name: str) -> None:
        """Notify admin that an organization was created."""
        html = self._render("org_created.html", {"org_name": org_name})
        self._send(
            to=to_email,
            subject=f"Organización {org_name} creada",
            html=html,
        )

    async def send_org_deleted(self, to_email: str, org_name: str) -> None:
        """Notify user that their organization was deleted."""
        html = self._render("org_deleted.html", {"org_name": org_name})
        self._send(
            to=to_email,
            subject=f"Organización {org_name} eliminada",
            html=html,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render(self, filename: str, ctx: dict) -> str:
        path = self._templates_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Email template not found: {filename}")
        return Template(path.read_text()).render(**ctx)

    def _send(self, to: str, subject: str, html: str) -> None:
        resend.Emails.send({
            "from": self._from_email,
            "to": to,
            "subject": subject,
            "html": html,
        })
