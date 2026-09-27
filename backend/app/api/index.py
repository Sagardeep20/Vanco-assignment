"""Dataset indexing endpoints.

POST /api/index runs one full orchestration job (scan -> process pending
images/PDFs/videos -> embeddings -> summary) via the indexing service.
GET /api/index/status reports the in-memory job progress.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.index import IndexStatus, IndexSummary
from app.services.indexing import (
    IndexingAlreadyRunningError,
    get_index_status,
    run_indexing_job,
)
from app.services.scanner import resolve_dataset_dir

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/index/status", response_model=IndexStatus)
def indexing_status():
    return IndexStatus(**get_index_status())


@router.post("/index", response_model=IndexSummary)
def start_indexing(db: Session = Depends(get_db)):
    dataset_root = resolve_dataset_dir()
    if not dataset_root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset directory does not exist: {dataset_root}. "
                f"Configure DATASET_PATH (default './data')."
            ),
        )
    try:
        summary = run_indexing_job(db, dataset_root)
    except IndexingAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("Indexing failed with a database error")
        raise HTTPException(
            status_code=500, detail=f"Indexing failed: {exc}"
        ) from exc
    return IndexSummary(**{k: v for k, v in summary.items() if k != "errors"})
