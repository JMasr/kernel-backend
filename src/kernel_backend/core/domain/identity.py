from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Certificate:
    name: str
    institution: str
    author_id: str
    public_key_pem: str
    created_at: str
    # private_key_pem is NOT stored here — it is returned at generation time only
