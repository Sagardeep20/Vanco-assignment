"""Basic Pydantic representations of an Asset.

Only creation input and read output are defined here; update/search
schemas will be added with the ingestion and search steps.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.asset import AssetStatus, FileType


class AssetCreate(BaseModel):
    """Fields accepted when registering a file for indexing."""

    filename: str
    original_path: str
    relative_path: str
    file_type: FileType
    mime_type: str | None = None
    file_size: int | None = None
    file_hash: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None


class AssetRead(BaseModel):
    """Full asset representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    original_path: str
    relative_path: str
    file_type: FileType
    mime_type: str | None = None
    file_size: int | None = None
    file_hash: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    description: str | None = None
    extracted_text: str | None = None
    status: AssetStatus
    error_message: str | None = None
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    last_indexed_at: datetime | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None


class OpenOriginalResponse(BaseModel):
    """Result of asking the server host to open the original file."""

    opened: bool
    message: str
    original_path: str
