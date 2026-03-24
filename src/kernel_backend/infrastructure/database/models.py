import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, LargeBinary, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Phase 6.A — Multi-tenancy
# ---------------------------------------------------------------------------

class OrgRecord(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    pepper_v1 = Column(String(64), nullable=True)
    current_pepper_version = Column(Integer, nullable=False, server_default="1")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ApiKeyRecord(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    key_hash = Column(String(64), nullable=False, unique=True)
    key_prefix = Column(String(12), nullable=False)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")


class OrgMemberRecord(Base):
    __tablename__ = "organization_members"
    __table_args__ = (UniqueConstraint("org_id", "user_id", name="uq_org_members_org_user"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False, server_default="member")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String, unique=True, nullable=False, index=True)
    content_hash_sha256 = Column(LargeBinary, nullable=True)
    author_id = Column(String, nullable=False, index=True)
    author_public_key = Column(Text, nullable=True)
    storage_key = Column(String, nullable=True)
    signed_storage_key = Column(String, nullable=True)
    is_signed = Column(Boolean, default=False, nullable=False)
    manifest_json = Column(Text, nullable=True)
    active_signals_json = Column(Text, nullable=True)       # JSON list[str]
    rs_n = Column(Integer, nullable=True)
    pilot_hash_48 = Column(BigInteger, nullable=True)        # 48-bit int
    manifest_signature = Column(LargeBinary, nullable=True)  # 64-byte Ed25519
    status = Column(String(20), nullable=False, server_default="VALID")
    schema_version = Column(Integer, server_default=text("2"), nullable=False)
    embedding_params = Column(JSON, nullable=True)  # JSONB on Postgres, JSON on SQLite
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True)


class EmbeddingRecipe(Base):
    __tablename__ = "embedding_recipes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String, ForeignKey("videos.content_id"), nullable=False, index=True)
    rs_n = Column(Integer, nullable=False)
    rs_k = Column(Integer, default=16, nullable=False)
    pilot_hash_48 = Column(LargeBinary, nullable=False)
    prng_seeds_json = Column(Text, nullable=False)
    band_configs_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class VideoSegment(Base):
    __tablename__ = "video_segments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String, ForeignKey("videos.content_id"), nullable=False, index=True)
    segment_index = Column(Integer, nullable=False)
    start_time_s = Column(Float, nullable=False)
    end_time_s = Column(Float, nullable=False)
    rs_codeword = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PilotToneIndex(Base):
    __tablename__ = "pilot_tone_index"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String, ForeignKey("videos.content_id"), nullable=False, index=True)
    pilot_hash_48 = Column(LargeBinary, nullable=False)
    signal_type = Column(String, nullable=False)  # "audio" | "video"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TransparencyLogEntry(Base):
    __tablename__ = "transparency_log_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String, nullable=False, index=True)
    author_id = Column(String, nullable=False)
    entry_hash = Column(LargeBinary, nullable=False)
    leaf_index = Column(Integer, nullable=False)
    manifest_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TransparencyLogRoot(Base):
    __tablename__ = "transparency_log_roots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tree_size = Column(Integer, nullable=False)
    root_hash = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Identity(Base):
    __tablename__ = "identities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    author_id = Column(String(16), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    institution = Column(String(255), nullable=False)
    public_key_pem = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True, index=True)


class InvitationRecord(Base):
    __tablename__ = "invitations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(UUID(as_uuid=True), nullable=False, unique=True, default=uuid.uuid4, index=True)
    email = Column(String(255), nullable=False, index=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(20), nullable=False, server_default="pending")  # pending|accepted|expired
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    accepted_at = Column(DateTime(timezone=True), nullable=True)


class AudioFingerprint(Base):
    __tablename__ = "audio_fingerprints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(String, ForeignKey("videos.content_id"), nullable=False, index=True)
    time_offset_ms = Column(Integer, nullable=False)
    hash_hex = Column(String(16), nullable=False)
    is_original = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
