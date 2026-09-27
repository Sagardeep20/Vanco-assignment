"""Schemas for asset search."""

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.asset import AssetRead


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    asset: AssetRead
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    method: str
    count: int
    results: list[SearchResultItem] = Field(default_factory=list)
