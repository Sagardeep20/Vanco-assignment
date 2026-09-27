"""Dataset scanner: discover files and upsert Asset rows.

Walks the configured dataset directory recursively, extracts basic file
metadata plus a SHA-256 hash for every supported file, and creates or
updates the corresponding :class:`Asset` row.

Later steps hook into the clearly marked extension points:
- image AI processing -> use ``extract_image_size`` / add new helpers
- PDF text extraction -> add ``extract_pdf_text`` next to ``probe_video``
- video frame processing / embeddings / semantic search -> services layer

The scan is idempotent: rows are looked up by ``relative_path`` and only
rewritten when the content hash changed. It is also crash-safe per file:
one bad file is recorded as ``failed`` (or ``unsupported``) without
aborting the whole run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.asset import Asset, AssetStatus, FileType

logger = logging.getLogger(__name__)

# Extension (lowercase, with dot) -> asset file type.
SUPPORTED_EXTENSIONS: dict[str, FileType] = {
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".png": FileType.IMAGE,
    ".webp": FileType.IMAGE,
    ".mp4": FileType.VIDEO,
    ".mov": FileType.VIDEO,
    ".mkv": FileType.VIDEO,
    ".webm": FileType.VIDEO,
    ".pdf": FileType.PDF,
}

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def classify_file(path: Path) -> FileType | None:
    """Return the asset file type for a path, or None if unsupported."""
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_utc_aware(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


@dataclass
class FileMetadata:
    """Basic metadata extracted for any supported file."""

    filename: str
    original_path: str
    relative_path: str
    file_type: FileType
    mime_type: str | None
    file_size: int
    file_hash: str
    created_at: datetime
    modified_at: datetime
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None


def extract_file_metadata(path: Path, dataset_root: Path) -> FileMetadata:
    """Extract basic metadata + SHA-256 hash for a supported file."""
    file_type = classify_file(path)
    if file_type is None:
        raise ValueError(f"Unsupported file type: {path.suffix or '(no extension)'}")
    stat = path.stat()
    absolute = path.resolve()
    return FileMetadata(
        filename=path.name,
        original_path=str(absolute),
        relative_path=absolute.relative_to(dataset_root.resolve()).as_posix(),
        file_type=file_type,
        mime_type=mimetypes.guess_type(path.name)[0],
        file_size=stat.st_size,
        file_hash=compute_sha256(path),
        created_at=_as_utc_aware(stat.st_ctime),
        modified_at=_as_utc_aware(stat.st_mtime),
    )


def extract_image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image. Raises on unreadable files."""
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


def probe_video(path: Path) -> tuple[int | None, int | None, float | None]:
    """Return (width, height, duration_seconds) for a video.

    Uses ffprobe when available. Returns all None when ffprobe is missing
    or the metadata cannot be read; never raises for metadata problems.
    """
    if shutil.which("ffprobe") is None:
        logger.warning("ffprobe not found; skipping video metadata for %s", path.name)
        return None, None, None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            logger.warning(
                "ffprobe failed for %s: %s", path.name, completed.stderr.strip()
            )
            return None, None, None
        info = json.loads(completed.stdout or "{}")
        streams = info.get("streams") or [{}]
        stream = streams[0]
        duration = stream.get("duration") or (info.get("format") or {}).get("duration")
        return (
            stream.get("width"),
            stream.get("height"),
            float(duration) if duration is not None else None,
        )
    except (subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Could not read video metadata for %s: %s", path.name, exc)
        return None, None, None


def resolve_dataset_dir(configured: str | Path | None = None) -> Path:
    """Resolve the dataset directory to scan.

    Relative paths are resolved against the current working directory;
    when that does not exist (e.g. backend started from ``backend/`` with
    the default ``./data``), fall back to the project root sibling of the
    backend package directory.
    """
    raw = Path(configured if configured is not None else settings.dataset_path)
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.is_dir():
        return cwd_candidate
    project_root_candidate = (
        Path(__file__).resolve().parent.parent.parent.parent / raw
    ).resolve()
    if project_root_candidate.is_dir():
        return project_root_candidate
    return cwd_candidate


@dataclass
class ScanResult:
    """Outcome counters for one dataset scan."""

    dataset_path: str
    discovered: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    unsupported: int = 0
    failed: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


def _hash_exists_elsewhere(
    db: Session, file_hash: str, relative_path: str
) -> bool:
    """Return True if another asset row already stores this hash."""
    return (
        db.query(Asset.id)
        .filter(Asset.file_hash == file_hash, Asset.relative_path != relative_path)
        .first()
        is not None
    )


def _record_failure(
    db: Session,
    relative_path: str,
    filename: str,
    original_path: str,
    file_type: FileType,
    message: str,
    result: ScanResult,
) -> None:
    logger.error("Failed %s: %s", relative_path, message)
    result.failed += 1
    result.errors.append(f"{relative_path}: {message}")
    now = datetime.now(timezone.utc)
    existing = db.query(Asset).filter(Asset.relative_path == relative_path).first()
    if existing is None:
        db.add(
            Asset(
                filename=filename,
                original_path=original_path,
                relative_path=relative_path,
                file_type=file_type,
                status=AssetStatus.FAILED,
                error_message=message,
                last_indexed_at=now,
            )
        )
    else:
        existing.status = AssetStatus.FAILED
        existing.error_message = message
        existing.last_indexed_at = now
    db.commit()


def _index_supported_file(
    db: Session, path: Path, dataset_root: Path, result: ScanResult
) -> None:
    """Index one supported file; records failure instead of raising."""
    relative = path.resolve().relative_to(dataset_root.resolve()).as_posix()
    try:
        meta = extract_file_metadata(path, dataset_root)
    except OSError as exc:
        _record_failure(
            db, relative, path.name, str(path.resolve()),
            classify_file(path) or FileType.OTHER,
            f"Could not read file: {exc}", result,
        )
        return

    if meta.file_type is FileType.IMAGE:
        try:
            meta.width, meta.height = extract_image_size(path)
        except Exception as exc:  # corrupt/unreadable image data
            _record_failure(
                db, meta.relative_path, meta.filename, meta.original_path,
                meta.file_type, f"Could not read image: {exc}", result,
            )
            return
    elif meta.file_type is FileType.VIDEO:
        # Missing/unreadable video metadata never fails the file itself.
        meta.width, meta.height, meta.duration_seconds = probe_video(path)
    # PDFs need only basic file metadata at this stage; text extraction
    # will be added later.

    now = datetime.now(timezone.utc)
    existing = db.query(Asset).filter(Asset.relative_path == meta.relative_path).first()

    if existing is not None and existing.file_hash == meta.file_hash:
        # Same content: refresh trivial fields, keep status/AI outputs.
        existing.filename = meta.filename
        existing.original_path = meta.original_path
        existing.file_size = meta.file_size
        existing.modified_at = meta.modified_at
        db.commit()
        logger.info("Skipped (unchanged): %s", meta.relative_path)
        result.skipped += 1
        return

    is_duplicate = _hash_exists_elsewhere(db, meta.file_hash, meta.relative_path)
    status = AssetStatus.DUPLICATE if is_duplicate else AssetStatus.PENDING

    if existing is None:
        db.add(
            Asset(
                filename=meta.filename,
                original_path=meta.original_path,
                relative_path=meta.relative_path,
                file_type=meta.file_type,
                mime_type=meta.mime_type,
                file_size=meta.file_size,
                file_hash=meta.file_hash,
                created_at=meta.created_at,
                modified_at=meta.modified_at,
                width=meta.width,
                height=meta.height,
                duration_seconds=meta.duration_seconds,
                status=status,
                last_indexed_at=now,
            )
        )
        db.commit()
        logger.info("Added: %s (status=%s)", meta.relative_path, status.value)
        result.added += 1
    else:
        existing.filename = meta.filename
        existing.original_path = meta.original_path
        existing.file_type = meta.file_type
        existing.mime_type = meta.mime_type
        existing.file_size = meta.file_size
        existing.file_hash = meta.file_hash
        existing.created_at = meta.created_at
        existing.modified_at = meta.modified_at
        existing.width = meta.width
        existing.height = meta.height
        existing.duration_seconds = meta.duration_seconds
        # Content changed: stale AI outputs must not survive.
        existing.description = None
        existing.extracted_text = None
        existing.embedding = None
        existing.status = status
        existing.error_message = None
        existing.last_indexed_at = now
        db.commit()
        logger.info("Updated: %s (status=%s)", meta.relative_path, status.value)
        result.updated += 1

    if is_duplicate:
        logger.info("Duplicate content: %s", meta.relative_path)
        result.duplicates += 1


def _index_unsupported_file(
    db: Session, path: Path, dataset_root: Path, result: ScanResult
) -> None:
    """Record a visible unsupported file as status=unsupported (idempotent)."""
    absolute = path.resolve()
    relative = absolute.relative_to(dataset_root.resolve()).as_posix()
    existing = db.query(Asset).filter(Asset.relative_path == relative).first()
    if existing is not None:
        logger.info("Skipped (unchanged): %s", relative)
        result.skipped += 1
        return
    try:
        stat = path.stat()
        file_hash = compute_sha256(path)
        created = _as_utc_aware(stat.st_ctime)
        modified = _as_utc_aware(stat.st_mtime)
    except OSError:
        created = modified = None
        file_hash = None
        stat = None
    db.add(
        Asset(
            filename=path.name,
            original_path=str(absolute),
            relative_path=relative,
            file_type=FileType.OTHER,
            mime_type=mimetypes.guess_type(path.name)[0],
            file_size=stat.st_size if stat else None,
            file_hash=file_hash,
            created_at=created,
            modified_at=modified,
            status=AssetStatus.UNSUPPORTED,
            last_indexed_at=datetime.now(timezone.utc),
        )
    )
    db.commit()
    logger.info("Unsupported: %s", relative)
    result.unsupported += 1


def scan_dataset(
    db: Session, dataset_path: str | Path | None = None
) -> ScanResult:
    """Recursively scan the dataset directory and upsert Asset rows."""
    dataset_root = resolve_dataset_dir(dataset_path)
    result = ScanResult(dataset_path=str(dataset_root))
    if not dataset_root.is_dir():
        message = f"Dataset directory does not exist: {dataset_root}"
        logger.error(message)
        result.errors.append(message)
        raise FileNotFoundError(message)

    logger.info("Scanning dataset directory: %s", dataset_root)
    for path in sorted(p for p in dataset_root.rglob("*") if p.is_file()):
        if path.name.startswith("."):
            # Hidden bookkeeping files (.gitkeep, .DS_Store, ...).
            logger.debug("Skipped (hidden): %s", path.name)
            result.skipped += 1
            continue
        result.discovered += 1
        try:
            if classify_file(path) is None:
                _index_unsupported_file(db, path, dataset_root, result)
            else:
                _index_supported_file(db, path, dataset_root, result)
        except Exception as exc:  # one bad path must not abort the scan
            # E.g. a symlink escaping the dataset root (relative_to fails),
            # an unreadable file, or an ffprobe exec problem. The scan
            # continues; the job still finishes and reports the failure.
            try:
                db.rollback()
            except Exception:
                pass
            message = f"{path.name}: unexpected scan error ({exc})"
            logger.error("Failed %s: %s", path.name, exc)
            result.failed += 1
            result.errors.append(message)

    logger.info(
        "Scan finished: discovered=%d added=%d updated=%d skipped=%d "
        "unsupported=%d failed=%d duplicates=%d",
        result.discovered, result.added, result.updated, result.skipped,
        result.unsupported, result.failed, result.duplicates,
    )
    return result
