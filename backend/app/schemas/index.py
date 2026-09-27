"""Schemas for the dataset indexing endpoints."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class IndexSummary(BaseModel):
    """Summary of one dataset scan + processing run."""

    model_config = ConfigDict(from_attributes=True)

    status: str
    dataset_path: str
    discovered: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    unsupported: int = 0
    failed: int = 0
    duplicates: int = 0
    # Processing phase counters (0 when nothing was pending).
    processed: int = 0
    processing_completed: int = 0
    processing_failed: int = 0


class IndexStatus(BaseModel):
    """In-memory indexing progress (no database table)."""

    model_config = ConfigDict(from_attributes=True)

    running: bool = False
    stage: str = "idle"
    dataset_path: str | None = None
    total_assets: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    duplicate: int = 0
    unsupported: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_error: str | None = None
