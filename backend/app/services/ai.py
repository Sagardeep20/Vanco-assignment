"""Local-only AI processing: descriptions, embeddings, PDF text.

No model downloads. All defaults are deterministic and dependency-free
(stdlib + Pillow/pypdf already in the stack):

- PDF text via pypdf (pure Python, graceful fallback to None).
- Descriptions via deterministic templates from file metadata.
- Text embeddings via the shared provider (CLIP when available,
  hashed bag-of-words fallback), 512-d, L2-normalized, so pgvector
  cosine search works.
- Image embeddings from actual pixels via the CLIP image tower, and
  video embeddings mean-pooled over sampled frames (see
  :mod:`app.services.video`) — all in the same joint space as text
  queries, which is what enables visual semantic search. PDFs stay
  text-based. When no image encoder is available, processing falls
  back to text embeddings so assets still complete.

Optional Ollama override: if ``AI_PROVIDER=ollama`` AND the configured
model is already present in the local Ollama instance, it is used for
descriptions. This path NEVER pulls/downloads models — if the model is
missing, unreachable, or slow, it falls back to local templates.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.asset import EMBEDDING_DIMENSION, Asset, AssetStatus, FileType
from app.services.embeddings import deterministic_embedding, embed_image, embed_text
from app.services.files import resolve_dataset_file

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PDF_TEXT_LIMIT = 8000
_OLLAMA_TIMEOUT_SECONDS = 8.0


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for hashed embeddings."""
    return _TOKEN_RE.findall(text.lower())


def generate_embedding(text: str, dim: int = EMBEDDING_DIMENSION) -> list[float] | None:
    """Deterministic hashed bag-of-words embedding.

    Kept for backwards compatibility (tests, callers); delegates to the
    deterministic embedding provider. New code should use
    :func:`app.services.embeddings.embed_text` so queries and stored
    vectors share one provider. Returns None for blank input.
    """
    return deterministic_embedding(text, dim)


def extract_pdf_text(path: Path, limit: int = _PDF_TEXT_LIMIT) -> tuple[str | None, int | None]:
    """Extract text + page count from a PDF with pypdf.

    Returns (None, None) when pypdf is missing, the file is unreadable,
    or no text is found. Never raises for content problems.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed; skipping PDF text for %s", path.name)
        return None, None
    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        chunks: list[str] = []
        total = 0
        for page in reader.pages:
            try:
                chunk = page.extract_text() or ""
            except Exception as exc:  # per-page failure must not abort
                logger.warning("PDF page extract failed for %s: %s", path.name, exc)
                continue
            chunk = chunk.strip()
            if not chunk:
                continue
            if total + len(chunk) > limit:
                chunks.append(chunk[: limit - total])
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= limit:
                break
        text = "\n".join(chunks).strip()
        return (text or None, pages)
    except Exception as exc:
        logger.warning("Could not read PDF %s: %s", path.name, exc)
        return None, None


def humanize_stem(filename: str) -> str:
    """Turn 'sunset-beach_01.png' into 'sunset beach 01'."""
    stem = Path(filename).stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem or Path(filename).stem


def _format_size(file_size: int | None) -> str:
    if not file_size:
        return "unknown size"
    if file_size < 1024:
        return f"{file_size} bytes"
    return f"{file_size / 1024:.1f} KB"


def build_searchable_text(
    filename: str,
    description: str | None,
    extracted_text: str | None = None,
) -> str:
    """Text that embeddings are built from (filename + AI + PDF text)."""
    parts = [filename, humanize_stem(filename)]
    if description:
        parts.append(description)
    if extracted_text:
        parts.append(extracted_text[:2000])
    return "\n".join(p for p in parts if p)


def _ollama_model_present(base_url: str, model: str) -> bool:
    """Check local Ollama tags WITHOUT pulling anything."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
        names = {m.get("name", "") for m in payload.get("models", [])}
        return model in names or any(n.startswith(model) for n in names)
    except Exception as exc:
        logger.info("Ollama not reachable at %s: %s", base_url, exc)
        return False


def _ollama_describe(base_url: str, model: str, prompt: str) -> str | None:
    """Ask local Ollama for a description. Never triggers a pull."""
    try:
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
        text = (payload.get("response") or "").strip()
        return text or None
    except Exception as exc:
        logger.info("Ollama generate failed, using local fallback: %s", exc)
        return None


def maybe_ollama_description(prompt: str) -> str | None:
    """Return an Ollama description only if provider=ollama and model exists.

    Explicitly never pulls. Any failure -> None -> local template wins.
    """
    provider = (getattr(settings, "ai_provider", "local") or "local").lower()
    if provider != "ollama":
        return None
    base_url = getattr(settings, "ollama_url", "http://localhost:11434")
    model = getattr(settings, "ollama_model", "")
    if not model:
        return None
    if not _ollama_model_present(base_url, model):
        logger.info("Ollama model '%s' not present locally; no pull attempted.", model)
        return None
    return _ollama_describe(base_url, model, prompt)


def generate_description(
    filename: str,
    file_type: FileType,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: float | None = None,
    file_size: int | None = None,
    extracted_text: str | None = None,
    pdf_pages: int | None = None,
) -> str:
    """Deterministic human-readable description (no model needed)."""
    title = humanize_stem(filename)
    size = _format_size(file_size)
    prompt_hint = f"Describe this {file_type.value} file named {filename}."
    ollama_text = maybe_ollama_description(prompt_hint)
    if ollama_text:
        return ollama_text

    if file_type is FileType.IMAGE:
        dims = f"{width}x{height}px" if width and height else "unknown dimensions"
        return (
            f"Image '{title}' ({dims}, {size}). "
            f"Local placeholder description from file metadata; "
            f"connect a vision model later for real captions."
        )
    if file_type is FileType.VIDEO:
        dims = f"{width}x{height}px" if width and height else "unknown resolution"
        dur = f"{duration_seconds:.1f}s" if duration_seconds else "unknown duration"
        return (
            f"Video '{title}' ({dims}, duration {dur}, {size}). "
            f"Local placeholder description from container metadata."
        )
    if file_type is FileType.PDF:
        pages = f"{pdf_pages} pages" if pdf_pages else "unknown page count"
        snippet = (extracted_text or "").strip().replace("\n", " ")[:220]
        if snippet:
            return (
                f"Document '{title}' ({pages}, {size}). "
                f"Content starts: {snippet}..."
            )
        return f"Document '{title}' ({pages}, {size}). No extractable text found."
    return f"File '{title}' ({file_type.value}, {size})."


def _embed_image_asset(asset: Asset, fallback_text: str) -> list[float] | None:
    """CLIP embedding from the asset's actual pixels (dataset-confined).

    Loads the file via :func:`app.services.files.resolve_dataset_file`
    (path-traversal protection) and encodes the pixels with the shared
    image tower. Any failure — file outside the dataset, missing file,
    unreadable pixels, or no image encoder — falls back to a text
    embedding so the asset still completes.
    """
    try:
        from PIL import Image

        path = resolve_dataset_file(asset.original_path)
        with Image.open(path) as img:
            vec = embed_image(img.convert("RGB"))
        if vec is not None:
            return vec
    except Exception as exc:
        logger.info(
            "Image embedding unavailable for %s (%s); using text embedding.",
            asset.relative_path, exc,
        )
    return embed_text(fallback_text)


def process_asset(db: Session, asset: Asset) -> Asset:
    """Run local AI processing for one asset row (idempotent-ish).

    - PDFs: extract text from disk when possible; embedding stays text-based.
    - Images: description from metadata, embedding from actual pixels
      (CLIP image tower; text fallback when pixels/encoder unavailable).
    - Videos: handled by :func:`app.services.video.process_video_asset`
      (frame pixels, mean-pooled); direct calls here keep the text path.
    - pending/failed -> completed; duplicates/unsupported left untouched
      by the batch driver (callers may still process one explicitly).
    Never raises for content issues; records failure on the row instead.
    """
    asset.status = AssetStatus.PROCESSING
    asset.processing_started_at = datetime.now(timezone.utc)
    asset.error_message = None
    db.commit()

    try:
        extracted: str | None = asset.extracted_text
        pdf_pages: int | None = None
        if asset.file_type is FileType.PDF and not extracted:
            disk_path: Path | None = None
            try:
                candidate = Path(asset.original_path)
                if candidate.is_file():
                    disk_path = candidate
            except (OSError, ValueError):
                disk_path = None
            if disk_path is not None:
                extracted, pdf_pages = extract_pdf_text(disk_path)
                asset.extracted_text = extracted

        description = generate_description(
            filename=asset.filename,
            file_type=asset.file_type,
            width=asset.width,
            height=asset.height,
            duration_seconds=asset.duration_seconds,
            file_size=asset.file_size,
            extracted_text=extracted,
            pdf_pages=pdf_pages,
        )
        asset.description = description
        searchable = build_searchable_text(asset.filename, description, extracted)
        if asset.file_type is FileType.IMAGE:
            # Genuine visual embedding (same joint space as text queries).
            asset.embedding = _embed_image_asset(asset, searchable)
        else:
            # Same provider as search queries (see services.embeddings).
            asset.embedding = embed_text(searchable)
        asset.status = AssetStatus.COMPLETED
        asset.processing_completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:  # never let one asset kill a batch
        logger.exception("AI processing failed for %s", asset.relative_path)
        asset.status = AssetStatus.FAILED
        asset.error_message = str(exc)[:2000]
        asset.processing_completed_at = datetime.now(timezone.utc)
        db.commit()
    return asset


def process_pending_assets(db: Session, limit: int = 100) -> dict:
    """Process up to ``limit`` pending rows. Skips duplicates/unsupported."""
    rows = (
        db.query(Asset)
        .filter(Asset.status == AssetStatus.PENDING)
        .order_by(Asset.last_indexed_at.asc())
        .limit(limit)
        .all()
    )
    completed = 0
    failed = 0
    errors: list[str] = []
    for asset in rows:
        result = process_asset(db, asset)
        if result.status == AssetStatus.COMPLETED:
            completed += 1
        else:
            failed += 1
            errors.append(f"{result.relative_path}: {result.error_message or 'failed'}")
    return {
        "processed": len(rows),
        "completed": completed,
        "failed": failed,
        "errors": errors,
    }


def _reembed_video(asset: Asset) -> list[float] | None:
    """Re-embed a completed video from freshly sampled frame pixels.

    Falls back to a text embedding of the stored description when the
    file/tooling is unavailable, so the run preserves the row instead
    of failing it. Status and all other columns are never touched.
    """
    try:
        # Local import: video.py imports from this module (cycle otherwise).
        from app.services.video import embed_video_file

        path = resolve_dataset_file(asset.original_path)
        vec = embed_video_file(path, duration=asset.duration_seconds)
        if vec is not None:
            return vec
    except Exception as exc:
        logger.info(
            "Video frame embedding unavailable for %s (%s); using text embedding.",
            asset.relative_path, exc,
        )
    searchable = build_searchable_text(
        asset.filename, asset.description, asset.extracted_text
    )
    return embed_text(searchable)


def reembed_completed_assets(db: Session, limit: int = 1000) -> dict:
    """One-time re-embedding of completed assets with the current provider.

    Regenerates ``Asset.embedding`` per media type:

    - images: from actual pixels via the CLIP image tower;
    - videos: mean-pooled over freshly sampled frames (text fallback
      when the file or ffmpeg is unavailable);
    - PDFs: from extracted/searchable text (text-based, unchanged).

    All vectors are 512-d and L2-normalized in the one shared CLIP
    joint space, matching text search queries.

    Safe to run on a live database:

    - no rescan: hashes never recomputed, rows never created/deleted;
    - only the ``embedding`` column is written — description, extracted
      text, metadata, status, and timestamps are left untouched;
    - only ``completed`` image/PDF/video rows are touched: failed,
      pending, duplicate, skipped, and unsupported rows are preserved;
    - per-asset commit: an interrupted run keeps progress and can simply
      be re-run (idempotent).
    """
    rows = (
        db.query(Asset)
        .filter(
            Asset.status == AssetStatus.COMPLETED,
            Asset.file_type.in_([FileType.IMAGE, FileType.PDF, FileType.VIDEO]),
        )
        .order_by(Asset.last_indexed_at.asc())
        .limit(limit)
        .all()
    )
    reembedded = 0
    failed = 0
    errors: list[str] = []
    for asset in rows:
        try:
            if asset.file_type is FileType.IMAGE:
                searchable = build_searchable_text(
                    asset.filename, asset.description, asset.extracted_text
                )
                asset.embedding = _embed_image_asset(asset, searchable)
            elif asset.file_type is FileType.VIDEO:
                asset.embedding = _reembed_video(asset)
            else:  # PDFs stay text-based.
                searchable = build_searchable_text(
                    asset.filename, asset.description, asset.extracted_text
                )
                asset.embedding = embed_text(searchable)
            db.commit()
            reembedded += 1
        except Exception as exc:  # one bad row must not abort the run
            try:
                db.rollback()
            except Exception:
                pass
            logger.exception(
                "Re-embedding failed for %s", asset.relative_path
            )
            failed += 1
            errors.append(f"{asset.relative_path}: {exc}")
    return {
        "processed": len(rows),
        "completed": reembedded,
        "failed": failed,
        "errors": errors,
    }
