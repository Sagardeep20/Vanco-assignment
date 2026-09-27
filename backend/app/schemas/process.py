"""Schemas for the local AI processing endpoint."""

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Optional limit for one processing run."""

    limit: int = Field(default=100, ge=1, le=1000)


class ProcessSummary(BaseModel):
    """Outcome of one processing run (no background jobs in this step)."""

    status: str
    processed: int = 0
    completed: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)


class VideoProcessSummary(BaseModel):
    """Outcome of one video-processing run."""

    status: str
    total: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = Field(default_factory=list)
