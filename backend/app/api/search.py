"""Semantic search endpoint (pgvector first, keyword fallback)."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.asset import FileType
from app.schemas.search import SearchResponse, SearchResultItem
from app.services.search import search_assets

router = APIRouter()


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(default="", min_length=1, max_length=500),
    limit: int = Query(default=20, ge=1, le=100),
    file_type: FileType | None = None,
    db: Session = Depends(get_db),
):
    results, method = search_assets(db, q, limit=limit, file_type=file_type)
    return SearchResponse(
        query=q,
        method=method,
        count=len(results),
        results=[SearchResultItem(asset=asset, score=score) for asset, score in results],
    )
