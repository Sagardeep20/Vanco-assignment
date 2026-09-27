"""Indexing orchestration: scan -> process -> embed -> summary.

Runs the full local pipeline for one indexing job:

1. scan the dataset directory (new/changed/duplicate/unsupported files),
2. process pending images + PDFs via :func:`process_asset` and pending
   videos via :func:`process_video_asset` (descriptions + 512-d embeddings
   come from the shared embedding provider),
3. persist per-asset status, continue past individual failures,
4. return a final summary and keep it in the in-memory status.

Concurrency: a module-level lock guarantees at most one job per process.
A second ``POST /api/index`` while a job runs gets a clear 409 instead of
a second job. Progress lives in application memory only — no new tables.

Incremental by construction: the scanner leaves unchanged rows alone
(completed rows are never reprocessed, so descriptions/embeddings are
not regenerated), marks changed files pending again (stale AI outputs
cleared by the scanner), and only pending rows are processed here.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.asset import Asset, AssetStatus, FileType
from app.services.ai import process_asset
from app.services.scanner import resolve_dataset_dir, scan_dataset
from app.services.video import process_video_asset

logger = logging.getLogger(__name__)

PROCESS_BATCH_LIMIT = 1000


class IndexingAlreadyRunningError(RuntimeError):
    """Raised when a job is started while another one is running."""


_INDEX_LOCK = threading.RLock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _idle_status() -> dict:
    return {
        "running": False,
        "stage": "idle",
        "dataset_path": None,
        "total_assets": 0,
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "duplicate": 0,
        "unsupported": 0,
        "started_at": None,
        "completed_at": None,
        "last_error": None,
    }


_INDEX_STATUS: dict = _idle_status()


def reset_index_status() -> None:
    """Reset in-memory status to idle (used by tests)."""
    with _INDEX_LOCK:
        _INDEX_STATUS.clear()
        _INDEX_STATUS.update(_idle_status())


def get_index_status() -> dict:
    """Return a snapshot of the current indexing status (idle if never run)."""
    with _INDEX_LOCK:
        return dict(_INDEX_STATUS)


def _set_status(**fields) -> None:
    with _INDEX_LOCK:
        _INDEX_STATUS.update(fields)


def _count_by_status(db: Session) -> dict[str, int]:
    rows = (
        db.query(Asset.status, func.count(Asset.id))
        .group_by(Asset.status)
        .all()
    )
    counts = {status.value: 0 for status in AssetStatus}
    for status, count in rows:
        key = status.value if isinstance(status, AssetStatus) else str(status)
        counts[key] = count
    return counts


def _refresh_status_counts(db: Session) -> None:
    """Reconcile in-memory progress counters with the database."""
    counts = _count_by_status(db)
    total = sum(counts.values())
    _set_status(
        total_assets=total,
        pending=counts.get(AssetStatus.PENDING.value, 0),
        processing=counts.get(AssetStatus.PROCESSING.value, 0),
        completed=counts.get(AssetStatus.COMPLETED.value, 0),
        failed=counts.get(AssetStatus.FAILED.value, 0),
        skipped=counts.get(AssetStatus.SKIPPED.value, 0),
        duplicate=counts.get(AssetStatus.DUPLICATE.value, 0),
        unsupported=counts.get(AssetStatus.UNSUPPORTED.value, 0),
    )


def _is_ffmpeg_unavailable_failure(asset: Asset) -> bool:
    """Return True for failed VIDEO rows blocked by missing ffmpeg/ffprobe.

    Only matches failures whose persisted message shows the tooling was
    unavailable (e.g. "ffmpeg/ffprobe not available on this system").
    Corrupt-video failures (ffprobe parse errors, frame-extraction
    failures, ...) do not match and are never retried here.
    """
    if asset.file_type is not FileType.VIDEO:
        return False
    if asset.status is not AssetStatus.FAILED:
        return False
    message = (asset.error_message or "").lower()
    return "ffmpeg" in message and "not available" in message


def _requeue_ffmpeg_failed_videos(db: Session) -> int:
    """Reset previously ffmpeg-blocked VIDEO failures back to pending.

    Runs once per indexing job, after the scan and before pending assets
    are processed. Only requeues when ffmpeg/ffprobe are available now;
    otherwise the retry would fail identically. No schema change: reuses
    the existing status/error_message columns.
    """
    import app.services.video as video_service

    try:
        tooling_ready = bool(
            video_service.ffmpeg_available()
            and video_service.ffprobe_available()
        )
    except Exception:
        tooling_ready = False
    if not tooling_ready:
        return 0
    candidates = (
        db.query(Asset)
        .filter(
            Asset.status == AssetStatus.FAILED,
            Asset.file_type == FileType.VIDEO,
        )
        .all()
    )
    requeued = 0
    for asset in candidates:
        if not _is_ffmpeg_unavailable_failure(asset):
            continue
        asset.status = AssetStatus.PENDING
        asset.error_message = None
        requeued += 1
    if requeued:
        db.commit()
        logger.info(
            "Requeued %d failed video(s) previously blocked by "
            "missing ffmpeg/ffprobe",
            requeued,
        )
    return requeued


def _fail_asset_unexpected(db: Session, asset: Asset, exc: BaseException) -> str:
    """Record an unexpected per-asset failure; never raises."""
    message = f"{type(exc).__name__}: {exc}"
    try:
        db.rollback()
        asset.status = AssetStatus.FAILED
        asset.error_message = message[:2000]
        asset.processing_completed_at = _now()
        db.commit()
    except Exception as commit_exc:  # last resort: log and move on
        logger.exception(
            "Could not persist failure for %s: %s", asset.relative_path, commit_exc
        )
        try:
            db.rollback()
        except Exception:
            pass
    return message


def run_indexing_job(
    db: Session,
    dataset_path: str | Path | None = None,
    limit: int = PROCESS_BATCH_LIMIT,
) -> dict:
    """Run one full indexing job synchronously.

    Raises :class:`IndexingAlreadyRunningError` if a job is running,
    :class:`FileNotFoundError` for a missing dataset directory.
    Individual asset failures are collected, never fatal.
    """
    with _INDEX_LOCK:
        if _INDEX_STATUS.get("running"):
            raise IndexingAlreadyRunningError(
                "An indexing job is already running. "
                "Check GET /api/index/status and retry when it finishes."
            )
        _INDEX_STATUS.update(
            running=True,
            stage="starting",
            started_at=_now(),
            completed_at=None,
            last_error=None,
        )
    try:
        return _run_locked(db, dataset_path, limit)
    finally:
        with _INDEX_LOCK:
            _INDEX_STATUS["running"] = False


def _run_locked(db: Session, dataset_path: str | Path | None, limit: int) -> dict:
    root = resolve_dataset_dir(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    started = _now()
    _set_status(
        running=True,
        stage="scanning",
        dataset_path=str(root),
        started_at=started,
        completed_at=None,
        last_error=None,
        pending=0,
        processing=0,
    )
    logger.info("Indexing job started for dataset directory: %s", root)

    # --- 1. scan ---------------------------------------------------------
    scan = scan_dataset(db, root)
    logger.info(
        "Indexing job scan complete: discovered=%d added=%d updated=%d "
        "skipped=%d unsupported=%d failed=%d duplicates=%d",
        scan.discovered, scan.added, scan.updated, scan.skipped,
        scan.unsupported, scan.failed, scan.duplicates,
    )
    _set_status(stage="processing")
    _refresh_status_counts(db)

    # Retry videos previously failed only because ffmpeg/ffprobe was
    # missing, now that the tooling is available. Unrelated failures
    # (corrupt videos, bad images, ...) stay failed; incremental
    # behavior for everything else is unchanged.
    _requeue_ffmpeg_failed_videos(db)
    _refresh_status_counts(db)

    # --- 2. process pending assets (images/PDFs + videos) ----------------
    pending_rows = (
        db.query(Asset)
        .filter(Asset.status == AssetStatus.PENDING)
        .order_by(Asset.last_indexed_at.asc())
        .limit(limit)
        .all()
    )
    processed = 0
    processing_completed = 0
    processing_failed = 0
    errors: list[str] = list(scan.errors)

    total_pending = len(pending_rows)
    for i, asset in enumerate(pending_rows, start=1):
        kind = "video" if asset.file_type == FileType.VIDEO else asset.file_type.value
        try:
            if asset.file_type == FileType.VIDEO:
                result = process_video_asset(db, asset)
            else:
                result = process_asset(db, asset)
        except Exception as exc:  # defensive: processors already trap content errors
            message = _fail_asset_unexpected(db, asset, exc)
            logger.error(
                "Indexing job asset failed (%d/%d): %s: %s",
                i, total_pending, asset.relative_path, message,
            )
            processed += 1
            processing_failed += 1
            errors.append(f"{asset.relative_path}: {message}")
            _set_status(failed=_INDEX_STATUS.get("failed", 0) + 1,
                        pending=max(0, _INDEX_STATUS.get("pending", 1) - 1))
            continue

        if result.status == AssetStatus.COMPLETED:
            processing_completed += 1
            logger.info(
                "Indexing job progress %d/%d: completed %s (%s)",
                i, total_pending, result.relative_path, kind,
            )
        else:
            processing_failed += 1
            detail = result.error_message or "processing failed"
            logger.error(
                "Indexing job asset failed (%d/%d): %s: %s",
                i, total_pending, result.relative_path, detail,
            )
            errors.append(f"{result.relative_path}: {detail}")
        processed += 1
        _set_status(
            pending=max(0, _INDEX_STATUS.get("pending", 1) - 1),
            completed=_INDEX_STATUS.get("completed", 0)
            + (1 if result.status == AssetStatus.COMPLETED else 0),
            failed=_INDEX_STATUS.get("failed", 0)
            + (0 if result.status == AssetStatus.COMPLETED else 1),
        )

    # --- 3. finalize -----------------------------------------------------
    _set_status(stage="finalizing")
    _refresh_status_counts(db)
    finished = _now()
    _set_status(running=False, stage="completed", completed_at=finished)
    final_counts = _count_by_status(db)

    logger.info(
        "Indexing job completed: discovered=%d processed=%d completed=%d "
        "failed=%d (scan_failed=%d) duplicates=%d unsupported=%d total=%d",
        scan.discovered, processed, processing_completed, processing_failed,
        scan.failed, scan.duplicates, scan.unsupported, sum(final_counts.values()),
    )
    return {
        "status": "completed",
        "dataset_path": scan.dataset_path,
        "discovered": scan.discovered,
        "added": scan.added,
        "updated": scan.updated,
        "skipped": scan.skipped,
        "unsupported": scan.unsupported,
        "failed": scan.failed,
        "duplicates": scan.duplicates,
        "processed": processed,
        "processing_completed": processing_completed,
        "processing_failed": processing_failed,
        "errors": errors,
    }
