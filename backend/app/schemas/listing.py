"""Schemas for asset listing / stats."""

from pydantic import BaseModel, Field

from app.schemas.asset import AssetRead


class AssetListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AssetRead] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total: int = 0
    images: int = 0
    videos: int = 0
    documents: int = 0
    pending: int = 0
    completed: int = 0
    duplicates: int = 0
    failed: int = 0
    unsupported: int = 0
