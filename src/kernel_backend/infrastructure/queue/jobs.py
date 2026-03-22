"""ARQ job functions for the signing pipeline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.services.signing_service import (
    _persist_payload,
    _sign_audio_cpu,
    _sign_av_cpu,
    _sign_video_cpu,
    sign_audio,
    sign_av,
    sign_video,
)
from kernel_backend.infrastructure.media.media_service import MediaService


async def _set_job_status(redis: object, job_id: str, status: dict) -> None:
    """Write job status to Redis key job:{job_id}:status with 1-hour TTL."""
    await redis.set(  # type: ignore[union-attr]
        f"job:{job_id}:status",
        json.dumps(status),
        ex=3600,
    )


def _sign_sync(
    media_path: str,
    cert_data: dict,
    private_key_pem: str,
    pepper: bytes,
    org_id: str | None = None,
    original_filename: str = "",
) -> dict:
    """Top-level picklable function — runs the CPU phase of signing in a subprocess.

    Probes the media file, routes to the correct _sign_*_cpu helper, and returns
    a RawSigningPayload dict. No async I/O, no storage writes, no DB calls.

    The parent async loop (process_sign_job) runs _persist_payload() after this
    returns to upload the signed file and save metadata to storage + registry.
    """
    certificate = Certificate(
        author_id=cert_data["author_id"],
        name=cert_data["name"],
        institution=cert_data["institution"],
        public_key_pem=cert_data["public_key_pem"],
        created_at=cert_data["created_at"],
    )
    media = MediaService()
    media_path_obj = Path(media_path)
    profile = media.probe(media_path_obj)
    parsed_org_id: UUID | None = UUID(org_id) if org_id else None

    if profile.has_video and profile.has_audio:
        return _sign_av_cpu(media_path_obj, certificate, private_key_pem, pepper, media, parsed_org_id, original_filename)
    elif profile.has_video:
        return _sign_video_cpu(media_path_obj, certificate, private_key_pem, pepper, media, parsed_org_id, original_filename)
    else:
        return _sign_audio_cpu(media_path_obj, certificate, private_key_pem, pepper, media, parsed_org_id, original_filename)


async def process_sign_job(
    ctx: dict,
    media_path: str,
    certificate_json: str,
    private_key_pem: str,
    org_id: str | None = None,
    org_pepper_hex: str | None = None,
    original_filename: str = "",
) -> dict:
    """
    Deserialize certificate_json → Certificate, then run the CPU phase in a
    ProcessPoolExecutor so the CPU-bound DSP work does not block the event loop.
    After the subprocess returns, run the I/O phase (storage + registry) in the
    parent async loop using real adapters from ctx.

    Idempotent: if the content_id already exists in the registry, returns the
    stored result without re-signing.

    Returns a JSON-serialisable dict: content_id, signed_media_key,
    active_signals, rs_n.
    """
    redis = ctx.get("redis")
    arq_job_id: str = ctx.get("job_id", "unknown")

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
    # Use org-specific pepper when provided (Phase C); fall back to global system pepper
    pepper: bytes = bytes.fromhex(org_pepper_hex) if org_pepper_hex else ctx["pepper"]
    process_pool = ctx.get("process_pool")

    loop = asyncio.get_event_loop()
    parsed_org_id: UUID | None = UUID(org_id) if org_id else None

    try:
        # Progress: processing (0%)
        if redis is not None:
            await _set_job_status(redis, arq_job_id, {
                "job_id": arq_job_id, "status": "processing", "progress": 0,
            })

        if process_pool is not None:
            # Progress: 20% — about to start CPU work in subprocess
            if redis is not None:
                await _set_job_status(redis, arq_job_id, {
                    "job_id": arq_job_id, "status": "processing", "progress": 20,
                })

            # CPU phase — runs in subprocess, returns RawSigningPayload dict
            payload = await loop.run_in_executor(
                process_pool,
                _sign_sync,
                media_path,
                cert_data,
                private_key_pem,
                pepper,
                org_id,
                original_filename,
            )

            # I/O phase — runs in parent async loop with real storage + registry
            await _persist_payload(payload, storage, registry)

            result = {
                "content_id": payload["content_id"],
                "signed_media_key": payload["signed_media_key"],
                "active_signals": payload["active_signals"],
                "rs_n": payload["rs_n"],
            }

        else:
            # Progress: 20% — about to start in-process signing
            if redis is not None:
                await _set_job_status(redis, arq_job_id, {
                    "job_id": arq_job_id, "status": "processing", "progress": 20,
                })

            # Fallback: run in-process (dev / no process pool)
            # Routes to the correct signing function based on media type.
            media_svc = MediaService()
            profile = media_svc.probe(Path(media_path))

            if profile.has_video and profile.has_audio:
                signing_result = await sign_av(
                    media_path=Path(media_path),
                    certificate=certificate,
                    private_key_pem=private_key_pem,
                    storage=storage,
                    registry=registry,
                    pepper=pepper,
                    media=media_svc,
                    org_id=parsed_org_id,
                    original_filename=original_filename,
                )
            elif profile.has_video:
                signing_result = await sign_video(
                    media_path=Path(media_path),
                    certificate=certificate,
                    private_key_pem=private_key_pem,
                    storage=storage,
                    registry=registry,
                    pepper=pepper,
                    media=media_svc,
                    org_id=parsed_org_id,
                    original_filename=original_filename,
                )
            else:
                signing_result = await sign_audio(
                    media_path=Path(media_path),
                    certificate=certificate,
                    private_key_pem=private_key_pem,
                    storage=storage,
                    registry=registry,
                    pepper=pepper,
                    media=media_svc,
                    org_id=parsed_org_id,
                    original_filename=original_filename,
                )

            result = {
                "content_id": signing_result.content_id,
                "signed_media_key": signing_result.signed_media_key,
                "active_signals": signing_result.active_signals,
                "rs_n": signing_result.rs_n,
            }

        # Progress: completed (100%)
        if redis is not None:
            await _set_job_status(redis, arq_job_id, {
                "job_id": arq_job_id,
                "status": "completed",
                "progress": 100,
                "result": result,
            })

        return result

    except Exception as exc:
        if redis is not None:
            await _set_job_status(redis, arq_job_id, {
                "job_id": arq_job_id,
                "status": "failed",
                "progress": 0,
                "error": str(exc),
            })
        raise


async def process_verify_job(ctx: dict, **kwargs: object) -> None:
    """Enqueued verify job. Phase 4 deliverable."""
    raise NotImplementedError("process_verify_job is not implemented — Phase 4")
