"""Asset listing + stats + local file access endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.asset import Asset, AssetStatus, FileType
from app.schemas.asset import AssetRead, OpenOriginalResponse
from app.schemas.listing import AssetListResponse, StatsResponse
from app.services import files as file_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/assets", response_model=AssetListResponse)
def list_assets(
    file_type: FileType | None = None,
    status: AssetStatus | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Asset)
    if file_type is not None:
        q = q.filter(Asset.file_type == file_type)
    if status is not None:
        q = q.filter(Asset.status == status)
    total = q.count()
    items = q.order_by(Asset.filename.asc()).offset(offset).limit(limit).all()
    return AssetListResponse(total=total, limit=limit, offset=offset, items=items)


@router.get("/assets/stats", response_model=StatsResponse)
def asset_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(Asset.id)).scalar() or 0

    def count_by(**filters) -> int:
        q = db.query(func.count(Asset.id))
        for column, value in filters.items():
            q = q.filter(getattr(Asset, column) == value)
        return q.scalar() or 0

    return StatsResponse(
        total=total,
        images=count_by(file_type=FileType.IMAGE),
        videos=count_by(file_type=FileType.VIDEO),
        documents=count_by(file_type=FileType.PDF),
        pending=count_by(status=AssetStatus.PENDING),
        completed=count_by(status=AssetStatus.COMPLETED),
        duplicates=count_by(status=AssetStatus.DUPLICATE),
        failed=count_by(status=AssetStatus.FAILED),
        unsupported=count_by(status=AssetStatus.UNSUPPORTED),
    )


@router.get("/assets/{asset_id}", response_model=AssetRead)
def get_asset(asset_id: UUID, db: Session = Depends(get_db)):
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


def _load_asset_or_404(asset_id: UUID, db: Session) -> Asset:
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.api_route("/assets/{asset_id}/file", methods=["GET", "HEAD"])
def get_asset_file(asset_id: UUID, db: Session = Depends(get_db)):
    """Serve the indexed file for inline preview.

    The asset is looked up by database ID only; the stored path must
    resolve inside the dataset directory (no arbitrary path access).
    """
    asset = _load_asset_or_404(asset_id, db)
    try:
        path = file_service.resolve_dataset_file(asset.original_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = file_service.guess_media_type(asset.filename, asset.mime_type)
    # "inline" (not the FileResponse default "attachment") so browsers render
    # images/video/PDFs inside <img>/<video>/<iframe> instead of downloading.
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=asset.filename,
        content_disposition_type="inline",
    )


@router.post("/assets/{asset_id}/open", response_model=OpenOriginalResponse)
def open_asset_original(asset_id: UUID, db: Session = Depends(get_db)):
    """Ask the server host to open the original file with its OS app.

    Local-desktop convenience only: works when the browser user and the
    server share the same machine. The path is validated exactly like the
    preview endpoint; failures return opened=false with a message.
    """
    asset = _load_asset_or_404(asset_id, db)
    try:
        path = file_service.resolve_dataset_file(asset.original_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        file_service.open_in_os(path)
    except Exception as exc:
        logger.info("Could not open %s in OS: %s", asset.relative_path, exc)
        return OpenOriginalResponse(
            opened=False,
            message=(
                "Could not open the file automatically on this host "
                f"({exc}). Copy the original path instead."
            ),
            original_path=asset.original_path,
        )
    return OpenOriginalResponse(
        opened=True,
        message="Opened with the operating system's default application.",
        original_path=asset.original_path,
    )
