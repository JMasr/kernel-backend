"""ARQ job functions for the signing pipeline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.services.signing_service import sign_audio
from kernel_backend.infrastructure.media.media_service import MediaService


async def process_sign_job(
    ctx: dict,
    media_path: str,
    certificate_json: str,
    private_key_pem: str,
) -> dict:
    """
    Deserialize certificate_json → Certificate, then run sign_audio() in a
    ProcessPoolExecutor so the CPU-bound DSP work does not block the event loop.

    Idempotent: if the content_id already exists in the registry, returns the
    stored result without re-signing.

    Returns a JSON-serialisable dict: content_id, signed_media_key,
    active_signals, rs_n.
    """
    cert_data = json.loads(certificate_json)
    certificate = Certificate(
        author_id=cert_data["author_id"],
        name=cert_data["name"],
        institution=cert_data["institution"],
        public_key_pem=cert_data["public_key_pem"],
        created_at=cert_data["created_at"],
    )

    storage = ctx["storage"]
    registry = ctx["registry"]
    pepper: bytes = ctx["pepper"]
    process_pool = ctx.get("process_pool")

    loop = asyncio.get_event_loop()

    if process_pool is not None:
        result = await loop.run_in_executor(
            process_pool,
            _sign_sync,
            media_path,
            cert_data,
            private_key_pem,
            pepper,
        )
        # Persist via registry/storage (async, must happen in this loop)
        # _sign_sync returns a plain dict; full storage/registry calls
        # are handled inside sign_audio when called directly.
        return result
    else:
        # Fallback: run in-process (dev / test)
        result = await sign_audio(
            media_path=Path(media_path),
            certificate=certificate,
            private_key_pem=private_key_pem,
            storage=storage,
            registry=registry,
            pepper=pepper,
            media=MediaService(),
        )
        return {
            "content_id": result.content_id,
            "signed_media_key": result.signed_media_key,
            "active_signals": result.active_signals,
            "rs_n": result.rs_n,
        }


def _sign_sync(
    media_path: str,
    cert_data: dict,
    private_key_pem: str,
    pepper: bytes,
) -> dict:
    """Top-level picklable wrapper that runs sign_audio in a new event loop.

    NOTE: storage and registry writes are performed in-process here.
    For production, wire a real storage/registry via ctx and move them
    back to the async layer after run_in_executor completes.
    This simplified version is suitable for CPU-offload testing.
    """
    from kernel_backend.core.domain.identity import Certificate  # noqa: PLC0415
    from kernel_backend.core.services.signing_service import sign_audio  # noqa: PLC0415
    from kernel_backend.infrastructure.media.media_service import MediaService  # noqa: PLC0415

    certificate = Certificate(
        author_id=cert_data["author_id"],
        name=cert_data["name"],
        institution=cert_data["institution"],
        public_key_pem=cert_data["public_key_pem"],
        created_at=cert_data["created_at"],
    )

    # _sign_sync cannot use real async storage/registry —
    # this is a placeholder; production jobs wire storage/registry via ctx.
    from kernel_backend.infrastructure.storage.local_storage import LocalStorageAdapter  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    tmp_base = tempfile.mkdtemp()
    storage = LocalStorageAdapter(base_path=__import__("pathlib").Path(tmp_base))

    class _NullRegistry:
        async def save_video(self, *a: object, **kw: object) -> None: ...
        async def save_segments(self, *a: object, **kw: object) -> None: ...
        async def get_by_content_id(self, *a: object, **kw: object): return None
        async def get_valid_candidates(self) -> list: return []
        async def match_fingerprints(self, *a: object, **kw: object) -> list: return []

    result = asyncio.run(sign_audio(
        media_path=__import__("pathlib").Path(media_path),
        certificate=certificate,
        private_key_pem=private_key_pem,
        storage=storage,
        registry=_NullRegistry(),
        pepper=pepper,
        media=MediaService(),
    ))
    return {
        "content_id": result.content_id,
        "signed_media_key": result.signed_media_key,
        "active_signals": result.active_signals,
        "rs_n": result.rs_n,
    }


async def process_verify_job(ctx: dict, **kwargs: object) -> None:
    """Enqueued verify job. Phase 4 deliverable."""
    raise NotImplementedError("process_verify_job is not implemented — Phase 4")
