import json
from typing import Optional
from uuid import UUID

from sqlalchemy import delete as sql_delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.watermark import (
    AudioEmbeddingParams,
    EmbeddingParams,
    VideoEmbeddingParams,
    SegmentFingerprint,
    VideoEntry,
    embedding_params_to_dict,
    embedding_params_from_dict,
)

_LEGACY_EMBEDDING_PARAMS = EmbeddingParams(
    audio=AudioEmbeddingParams(
        dwt_levels=(1, 2),
        chips_per_bit=256,
        psychoacoustic=False,
        safety_margin_db=3.0,
        target_snr_db=-14.0,
    ),
    video=VideoEmbeddingParams(
        jnd_adaptive=False,
        qim_step_base=64.0,
        qim_step_min=44.0,
        qim_step_max=128.0,
        qim_quantize_to=4.0,
    ),
)
from kernel_backend.core.ports.registry import RegistryPort
from kernel_backend.infrastructure.database.models import (
    AudioFingerprint,
    EmbeddingRecipe,
    Identity,
    PilotToneIndex,
    Video,
    VideoSegment,
)


class IdentityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, certificate: Certificate) -> None:
        """
        Persist the public fields of Certificate.
        Idempotent on author_id: if the author_id already exists, do nothing.
        """
        stmt = (
            insert(Identity)
            .values(
                author_id=certificate.author_id,
                name=certificate.name,
                institution=certificate.institution,
                public_key_pem=certificate.public_key_pem,
            )
            .on_conflict_do_nothing(index_elements=["author_id"])
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def create_with_org(self, certificate: Certificate, org_id: UUID) -> None:
        """Persist certificate linked to an organization. Idempotent on author_id."""
        stmt = (
            insert(Identity)
            .values(
                author_id=certificate.author_id,
                name=certificate.name,
                institution=certificate.institution,
                public_key_pem=certificate.public_key_pem,
                org_id=org_id,
            )
            .on_conflict_do_nothing(index_elements=["author_id"])
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def get_by_org_id(self, org_id: UUID) -> list[Certificate]:
        """Return all certificates belonging to an organization."""
        result = await self._session.execute(
            select(Identity).where(Identity.org_id == org_id)
        )
        return [
            Certificate(
                author_id=row.author_id,
                name=row.name,
                institution=row.institution,
                public_key_pem=row.public_key_pem,
                created_at=row.created_at.isoformat(),
            )
            for row in result.scalars().all()
        ]

    async def delete_by_author_id(self, author_id: str) -> bool:
        """Delete identity by author_id. Returns True if deleted, False if not found."""
        stmt = sql_delete(Identity).where(Identity.author_id == author_id)
        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount > 0

    async def get_by_author_id(self, author_id: str) -> Certificate | None:
        """
        Return Certificate domain object or None if not found.
        Maps ORM Identity → Certificate domain object.
        """
        result = await self._session.execute(
            select(Identity).where(Identity.author_id == author_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return Certificate(
            author_id=row.author_id,
            name=row.name,
            institution=row.institution,
            public_key_pem=row.public_key_pem,
            created_at=row.created_at.isoformat(),
        )


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _video_row_to_entry(row: Video) -> VideoEntry:
    ep = (
        embedding_params_from_dict(row.embedding_params)
        if row.embedding_params is not None
        else _LEGACY_EMBEDDING_PARAMS
    )
    return VideoEntry(
        content_id=row.content_id,
        author_id=row.author_id,
        author_public_key=row.author_public_key or "",
        active_signals=json.loads(row.active_signals_json or "[]"),
        rs_n=row.rs_n or 0,
        pilot_hash_48=row.pilot_hash_48 or 0,
        manifest_signature=row.manifest_signature or b"",
        embedding_params=ep,
        manifest_json=row.manifest_json or "",
        schema_version=row.schema_version,
        status=row.status or "VALID",
        org_id=row.org_id,
        signed_media_key=row.signed_storage_key or "",
    )


class VideoRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_video(self, entry: VideoEntry) -> None:
        stmt = (
            insert(Video)
            .values(
                content_id=entry.content_id,
                author_id=entry.author_id,
                author_public_key=entry.author_public_key,
                active_signals_json=json.dumps(entry.active_signals),
                rs_n=entry.rs_n,
                pilot_hash_48=entry.pilot_hash_48,
                manifest_signature=entry.manifest_signature,
                manifest_json=entry.manifest_json if entry.manifest_json else None,
                schema_version=entry.schema_version,
                status=entry.status,
                org_id=entry.org_id,
                signed_storage_key=entry.signed_media_key if entry.signed_media_key else None,
                embedding_params=embedding_params_to_dict(entry.embedding_params),
            )
            .on_conflict_do_nothing(index_elements=["content_id"])
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def get_by_content_id(self, content_id: str) -> VideoEntry | None:
        result = await self._session.execute(
            select(Video).where(Video.content_id == content_id)
        )
        row = result.scalar_one_or_none()
        return None if row is None else _video_row_to_entry(row)

    async def save_video_with_org(self, entry: VideoEntry, org_id: UUID) -> None:
        """Persist a VideoEntry linked to an organization. Idempotent on content_id."""
        stmt = (
            insert(Video)
            .values(
                content_id=entry.content_id,
                author_id=entry.author_id,
                author_public_key=entry.author_public_key,
                active_signals_json=json.dumps(entry.active_signals),
                rs_n=entry.rs_n,
                pilot_hash_48=entry.pilot_hash_48,
                manifest_signature=entry.manifest_signature,
                manifest_json=entry.manifest_json if entry.manifest_json else None,
                schema_version=entry.schema_version,
                status=entry.status,
                org_id=org_id,
                signed_storage_key=entry.signed_media_key if entry.signed_media_key else None,
                embedding_params=embedding_params_to_dict(entry.embedding_params),
            )
            .on_conflict_do_nothing(index_elements=["content_id"])
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def get_by_org_id(self, org_id: UUID) -> list[VideoEntry]:
        """Return all video entries belonging to an organization."""
        result = await self._session.execute(
            select(Video).where(Video.org_id == org_id)
        )
        return [_video_row_to_entry(r) for r in result.scalars().all()]

    async def get_valid_candidates(self) -> list[VideoEntry]:
        result = await self._session.execute(
            select(Video).where(Video.status == "VALID")
        )
        return [_video_row_to_entry(r) for r in result.scalars().all()]

    async def save_segments(
        self,
        content_id: str,
        segments: list[SegmentFingerprint],
        is_original: bool,
    ) -> None:
        for seg in segments:
            self._session.add(AudioFingerprint(
                content_id=content_id,
                time_offset_ms=seg.time_offset_ms,
                hash_hex=seg.hash_hex,
                is_original=is_original,
            ))
        await self._session.commit()

    async def match_fingerprints(
        self,
        hashes: list[str],
        max_hamming: int = 10,
        org_id: UUID | None = None,
    ) -> list[VideoEntry]:
        """Iterate all stored fingerprints and compute hamming distance in Python.

        When org_id is provided, only fingerprints belonging to that organization
        are considered (multi-tenant isolation).
        """
        if org_id is not None:
            stmt = (
                select(AudioFingerprint)
                .join(Video, AudioFingerprint.content_id == Video.content_id)
                .where(Video.org_id == org_id)
            )
        else:
            stmt = select(AudioFingerprint)

        result = await self._session.execute(stmt)
        all_fp = result.scalars().all()

        matching_content_ids: set[str] = set()
        for fp in all_fp:
            for qh in hashes:
                if _hamming(fp.hash_hex, qh) <= max_hamming:
                    matching_content_ids.add(fp.content_id)
                    break

        entries: list[VideoEntry] = []
        for cid in matching_content_ids:
            entry = await self.get_by_content_id(cid)
            if entry is not None:
                entries.append(entry)
        return entries

    async def list_by_org_id(
        self,
        org_id: UUID,
        author_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[tuple[VideoEntry, str | None, str | None]]:
        """List org content with pagination.

        Returns list of (VideoEntry, author_name, created_at_iso) tuples.
        author_name is None if no identity record exists for the author.
        created_at_iso is ISO-8601 string from the Video row.
        """
        stmt = (
            select(Video, Identity.name)
            .outerjoin(Identity, (Video.author_id == Identity.author_id) & (Video.org_id == Identity.org_id))
            .where(Video.org_id == org_id)
        )
        if author_id is not None:
            stmt = stmt.where(Video.author_id == author_id)
        stmt = stmt.order_by(Video.created_at.desc()).limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        rows = result.all()

        out: list[tuple[VideoEntry, str | None, str | None]] = []
        for video_row, author_name in rows:
            entry = _video_row_to_entry(video_row)
            created_at_iso = video_row.created_at.isoformat() if video_row.created_at else None
            out.append((entry, author_name, created_at_iso))
        return out

    async def delete_by_content_id(self, content_id: UUID, org_id: UUID) -> bool:
        """Delete a video entry and all related child rows, scoped to org.

        Deletes from child tables (audio_fingerprints, video_segments,
        embedding_recipes, pilot_tone_index) before the parent video row
        to satisfy foreign key constraints.
        Returns True if the video was deleted, False if not found.
        """
        cid = str(content_id)

        # Verify the video exists and belongs to this org before cascading
        row = await self._session.execute(
            select(Video.id).where(Video.content_id == cid).where(Video.org_id == org_id)
        )
        if row.scalar_one_or_none() is None:
            return False

        # Delete child rows referencing this content_id
        for child in (AudioFingerprint, VideoSegment, EmbeddingRecipe, PilotToneIndex):
            await self._session.execute(
                sql_delete(child).where(child.content_id == cid)
            )

        # Delete the parent video row
        await self._session.execute(
            sql_delete(Video).where(Video.content_id == cid).where(Video.org_id == org_id)
        )
        await self._session.commit()
        return True

    async def count_by_org_id(
        self,
        org_id: UUID,
        author_id: Optional[str] = None,
    ) -> int:
        """Count total video entries for an organization."""
        stmt = select(func.count()).select_from(Video).where(Video.org_id == org_id)
        if author_id is not None:
            stmt = stmt.where(Video.author_id == author_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()


class SessionFactoryRegistry(RegistryPort):
    """RegistryPort adapter that creates a fresh session per method call.

    Required for app.state.registry: VideoRepository is session-scoped
    (takes AsyncSession), so it cannot be stored directly in app.state.
    This wrapper holds the session factory and opens a new session for
    each port method invocation.
    """

    def __init__(self, session_factory: object) -> None:
        self._factory = session_factory

    async def save_video(self, entry: VideoEntry) -> None:
        async with self._factory() as session:  # type: ignore[operator]
            await VideoRepository(session).save_video(entry)

    async def get_by_content_id(self, content_id: str) -> VideoEntry | None:
        async with self._factory() as session:  # type: ignore[operator]
            return await VideoRepository(session).get_by_content_id(content_id)

    async def get_valid_candidates(self) -> list[VideoEntry]:
        async with self._factory() as session:  # type: ignore[operator]
            return await VideoRepository(session).get_valid_candidates()

    async def save_segments(
        self,
        content_id: str,
        segments: list[SegmentFingerprint],
        is_original: bool,
    ) -> None:
        async with self._factory() as session:  # type: ignore[operator]
            await VideoRepository(session).save_segments(content_id, segments, is_original)

    async def match_fingerprints(
        self,
        hashes: list[str],
        max_hamming: int = 10,
        org_id: UUID | None = None,
    ) -> list[VideoEntry]:
        async with self._factory() as session:  # type: ignore[operator]
            return await VideoRepository(session).match_fingerprints(hashes, max_hamming, org_id)
