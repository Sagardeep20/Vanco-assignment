"""Asset ORM model.

Single table storing every indexed media asset and its processing state.
One table covers images, videos, and PDFs (see ``width``/``height``/
``duration_seconds``); no per-type tables are needed.

Embedding dimension is defined once as :data:`EMBEDDING_DIMENSION` so it
can be changed in one place if the embedding model changes (the column
type change itself will need a migration).
"""

import uuid
from datetime import datetime
from enum import Enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base

# Dimension of the embedding vector. 512 matches CLIP ViT-B/32, a standard
# joint image-text embedding space suitable for image/video/document
# semantic search. Change this constant (plus a migration) to switch models.
EMBEDDING_DIMENSION = 512


def _enum_values(e: type[Enum]) -> list[str]:
    """Store lowercase enum values (not member names) in PostgreSQL."""
    return [m.value for m in e]


class FileType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    PDF = "pdf"
    OTHER = "other"


class AssetStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"
    UNSUPPORTED = "unsupported"


class Asset(Base):
    __tablename__ = "assets"

    # Identity and file information
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    # Path relative to the dataset root. Unique: one row per dataset file,
    # which is what incremental indexing looks up.
    relative_path: Mapped[str] = mapped_column(
        String(1024), nullable=False, unique=True
    )
    file_type: Mapped[FileType] = mapped_column(
        SAEnum(
            FileType,
            name="asset_file_type",
            native_enum=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        index=True,
    )
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # SHA-256 hex digest. Nullable (computed during indexing) and indexed
    # so duplicate detection is a fast lookup by hash. Deliberately not
    # unique: duplicate rows are kept with status="duplicate".
    file_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    # Filesystem timestamps of the source file (not row bookkeeping).
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # AI / content information
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Embedding vector. Nullable: assets exist before AI processing runs.
    # No vector similarity index yet; added later with the search step.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION), nullable=True
    )

    # Processing state
    status: Mapped[AssetStatus] = mapped_column(
        SAEnum(
            AssetStatus,
            name="asset_status",
            native_enum=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=AssetStatus.PENDING,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Optional media metadata (shared columns, no per-type tables)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"Asset(id={self.id}, relative_path={self.relative_path!r}, "
            f"file_type={self.file_type}, status={self.status})"
        )
