import json

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from kernel_backend.core.domain.identity import Certificate
from kernel_backend.core.domain.watermark import SegmentFingerprint, VideoEntry
from kernel_backend.infrastructure.database.models import AudioFingerprint, Identity, Video


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
    return VideoEntry(
        content_id=row.content_id,
        author_id=row.author_id,
        author_public_key=row.author_public_key or "",
        active_signals=json.loads(row.active_signals_json or "[]"),
        rs_n=row.rs_n or 0,
        pilot_hash_48=row.pilot_hash_48 or 0,
        manifest_signature=row.manifest_signature or b"",
        manifest_json=row.manifest_json or "",
        schema_version=row.schema_version,
        status=row.status or "VALID",
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
    ) -> list[VideoEntry]:
        """Iterate all stored fingerprints and compute hamming distance in Python."""
        result = await self._session.execute(select(AudioFingerprint))
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
