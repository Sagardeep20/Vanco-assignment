"""Local AI processing endpoint.

POST /api/process runs deterministic local processing (PDF text,
descriptions, hashed embeddings) over pending rows. No model downloads;
optional Ollama is used only when already present locally.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.process import ProcessRequest, ProcessSummary, VideoProcessSummary
from app.services.ai import process_pending_assets, reembed_completed_assets
from app.services.video import process_pending_videos

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process", response_model=ProcessSummary)
def run_processing(body: ProcessRequest | None = None, db: Session = Depends(get_db)):
    limit = body.limit if body is not None else 100
    logger.info("AI processing started (limit=%d)", limit)
    try:
        outcome = process_pending_assets(db, limit=limit)
    except SQLAlchemyError as exc:
        logger.exception("Processing failed with a database error")
        return ProcessSummary(status="failed", errors=[str(exc)])
    logger.info(
        "AI processing completed: processed=%d completed=%d failed=%d",
        outcome["processed"], outcome["completed"], outcome["failed"],
    )
    return ProcessSummary(status="completed", **outcome)


@router.post("/process/videos", response_model=VideoProcessSummary)
def run_video_processing(body: ProcessRequest | None = None, db: Session = Depends(get_db)):
    """Process pending video assets via sampled frames.

    Never downloads models; uses ffmpeg/ffprobe when present and the
    existing vision-provider abstraction otherwise. Missing tooling marks
    videos failed with a clear message instead of crashing.
    """
    limit = body.limit if body is not None else 100
    logger.info("Video processing started (limit=%d)", limit)
    try:
        outcome = process_pending_videos(db, limit=limit)
    except SQLAlchemyError as exc:
        logger.exception("Video processing failed with a database error")
        return VideoProcessSummary(status="failed", errors=[str(exc)])
    logger.info(
        "Video processing completed: total=%d processed=%d failed=%d skipped=%d",
        outcome["total"], outcome["processed"], outcome["failed"], outcome["skipped"],
    )
    return VideoProcessSummary(status="completed", **outcome)


@router.post("/process/reembed", response_model=ProcessSummary)
def run_reembedding(body: ProcessRequest | None = None, db: Session = Depends(get_db)):
    """One-time re-embedding of completed assets with the current provider.

    Recomputes embeddings for completed image/PDF/video rows (e.g. after
    switching from deterministic to CLIP embeddings) without rescanning
    files or touching metadata/status. Only the ``embedding`` column is
    rewritten; failed/unsupported/duplicate/pending rows are preserved.
    Safe to re-run; normal incremental indexing is unchanged.
    """
    limit = body.limit if body is not None else 1000
    logger.info("Re-embedding started (limit=%d)", limit)
    try:
        outcome = reembed_completed_assets(db, limit=limit)
    except SQLAlchemyError as exc:
        logger.exception("Re-embedding failed with a database error")
        return ProcessSummary(status="failed", errors=[str(exc)])
    logger.info(
        "Re-embedding completed: processed=%d completed=%d failed=%d",
        outcome["processed"], outcome["completed"], outcome["failed"],
    )
    return ProcessSummary(status="completed", **outcome)
