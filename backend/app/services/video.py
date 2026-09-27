"""Video processing: frame sampling + vision descriptions.

Local-only, no model downloads. For each pending video asset:

- verify the file exists on disk,
- determine duration (stored metadata, else ffprobe),
- sample frames approximately every ``FRAME_INTERVAL_SECONDS`` seconds
  (capped at ``MAX_FRAMES`` for long videos) into a temporary directory,
- describe each frame via the existing vision-provider abstraction
  (:func:`app.services.ai.maybe_ollama_description` with a deterministic
  local fallback — never pulls/downloads models),
- combine the frame notes into one video description saved to
  ``Asset.description``.

State: pending -> processing -> completed; on failure
processing -> failed with ``error_message``. One video's failure never
stops the batch. If ffmpeg/ffprobe is unavailable, videos are marked
failed with a clear message instead of crashing. Extracted frames are
temporary only and always cleaned up.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.asset import Asset, AssetStatus, FileType
from app.services.ai import humanize_stem, maybe_ollama_description
from app.services.embeddings import (
    EmbeddingUnavailableError,
    embed_image,
    embed_text,
    l2_normalize,
)

logger = logging.getLogger(__name__)

# Sample roughly one frame every 10s; at most 6 frames per video so long
# videos stay cheap. Both are module constants so tests can monkeypatch.
FRAME_INTERVAL_SECONDS = 10.0
MAX_FRAMES = 6
_FFMPEG_TIMEOUT_SECONDS = 60


def ffmpeg_available() -> bool:
    """Return True if the ``ffmpeg`` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    """Return True if the ``ffprobe`` binary is on PATH."""
    return shutil.which("ffprobe") is not None


def get_video_duration(path: Path, stored: float | None = None) -> float | None:
    """Return video duration in seconds.

    Prefers the stored scanner metadata; falls back to ffprobe.
    Returns None when unknown. Never raises for metadata problems.
    """
    if stored:
        return stored
    if not ffprobe_available():
        return None
    try:
        from app.services.scanner import probe_video

        _, _, duration = probe_video(path)
        return duration
    except Exception as exc:  # defensive: scanner helper is best-effort
        logger.warning("Could not determine duration for %s: %s", path.name, exc)
        return None


def compute_frame_timestamps(
    duration: float | None,
    interval: float = FRAME_INTERVAL_SECONDS,
    max_frames: int = MAX_FRAMES,
) -> list[float]:
    """Timestamps (seconds) to sample: ~one per ``interval``, capped."""
    if not duration or duration <= 0:
        return [1.0]
    count = int(duration // interval) or 1
    count = max(1, min(max_frames, count))
    step = duration / count
    return [round((i + 0.5) * step, 2) for i in range(count)]


def extract_frames(video_path: Path, timestamps: list[float], tmpdir: Path) -> list[Path]:
    """Extract one JPEG frame per timestamp into ``tmpdir``.

    Raises RuntimeError when ffmpeg is missing or no frame could be
    extracted. Per-frame failures are logged and skipped; at least one
    success is required.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not available on this system; "
            "install ffmpeg to enable video frame sampling."
        )
    extracted: list[Path] = []
    for i, ts in enumerate(timestamps):
        out = tmpdir / f"frame_{i:03d}.jpg"
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    str(ts),
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(out),
                ],
                capture_output=True,
                text=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("ffmpeg failed for %s at %ss: %s", video_path.name, ts, exc)
            continue
        if completed.returncode != 0 or not out.is_file():
            logger.warning(
                "ffmpeg could not extract frame at %ss for %s: %s",
                ts,
                video_path.name,
                (completed.stderr or "").strip(),
            )
            continue
        extracted.append(out)
    if not extracted:
        raise RuntimeError(
            f"No frames could be extracted from {video_path.name} "
            f"({len(timestamps)} timestamp(s) attempted)."
        )
    return extracted


def describe_frame(frame_path: Path, index: int, timestamp: float, asset: Asset) -> str:
    """Describe one sampled frame via the existing vision abstraction.

    Uses :func:`maybe_ollama_description` when a local Ollama model is
    already configured/present (never pulls); otherwise a deterministic
    local note from the frame's readable metadata. Never raises for
    content problems — unreadable frames get a plain placeholder note.
    """
    label = f"frame {index + 1} at {timestamp:.1f}s of video '{asset.filename}'"
    try:
        vision_text = maybe_ollama_description(f"Describe this video frame: {label}.")
    except Exception as exc:  # abstraction must never break the batch
        logger.info("Vision provider failed for %s: %s", label, exc)
        vision_text = None
    if vision_text:
        return f"Frame {index + 1} at {timestamp:.1f}s: {vision_text}"

    dims = "unknown dimensions"
    try:
        from PIL import Image

        with Image.open(frame_path) as img:
            dims = f"{img.width}x{img.height}px"
    except Exception:
        dims = "unreadable frame data"
    return (
        f"Frame {index + 1} at {timestamp:.1f}s ({dims}). "
        f"Local frame note; no vision model used."
    )


def combine_frame_descriptions(
    filename: str,
    duration: float | None,
    interval: float,
    frame_descriptions: list[str],
) -> str:
    """Combine per-frame notes into one final video description."""
    title = humanize_stem(filename)
    dur = f"{duration:.1f}s" if duration else "unknown duration"
    header = (
        f"Video '{title}' ({dur}, "
        f"{len(frame_descriptions)} frame(s) sampled ~every {interval:g}s)."
    )
    body = " ".join(f"[{i + 1}] {d}" for i, d in enumerate(frame_descriptions))
    return f"{header} Frame observations: {body}"


def embed_frame_files(frames: list[Path]) -> list[float] | None:
    """Aggregate CLIP image embeddings over sampled frame files.

    Each frame's actual pixels go through the shared image tower; the
    per-frame vectors are mean-pooled and L2-normalized into one
    512-d vector. Returns None when no frame yields an embedding or
    the provider has no image encoder (caller falls back to text).
    """
    from PIL import Image

    vectors: list[list[float]] = []
    for frame in frames:
        try:
            with Image.open(frame) as img:
                vec = embed_image(img.convert("RGB"))
        except EmbeddingUnavailableError:
            logger.info("Image encoder unavailable; using text embedding instead.")
            return None
        except Exception as exc:
            logger.warning("Skipping frame %s for embedding: %s", frame.name, exc)
            continue
        if vec:
            vectors.append(vec)
    if not vectors:
        return None
    mean = [sum(col) / len(vectors) for col in zip(*vectors)]
    return l2_normalize(mean)


def embed_video_file(video_path: Path, duration: float | None = None) -> list[float] | None:
    """Sample a video file's frames and return one aggregated image embedding.

    Returns None when frame tooling/files fail (caller falls back to
    text). Never raises for content or tooling problems.
    """
    if not ffmpeg_available() or not ffprobe_available():
        return None
    try:
        timestamps = compute_frame_timestamps(
            get_video_duration(video_path, stored=duration)
        )
        with tempfile.TemporaryDirectory(prefix="dam-reembed-frames-") as tmp:
            frames = extract_frames(video_path, timestamps, Path(tmp))
            return embed_frame_files(frames)
    except Exception as exc:
        logger.info("Video frame embedding failed for %s: %s", video_path.name, exc)
        return None


def _mark_failed(
    db: Session, asset: Asset, message: str
) -> Asset:
    logger.error("Video processing failed for %s: %s", asset.relative_path, message)
    asset.status = AssetStatus.FAILED
    asset.error_message = message[:2000]
    asset.processing_completed_at = datetime.now(timezone.utc)
    db.commit()
    return asset


def process_video_asset(db: Session, asset: Asset) -> Asset:
    """Process one video asset: sample frames, describe, save description.

    - Non-video or already-completed rows are returned untouched (the
      batch driver counts them as skipped).
    - Only ``Asset.description``/``Asset.embedding`` + status bookkeeping
      are written; frames live in a temporary directory that is removed.
    - Never raises for content/tooling problems; records failure instead.
    """
    if asset.file_type is not FileType.VIDEO:
        return asset
    if asset.status == AssetStatus.COMPLETED:
        return asset

    try:
        video_path = Path(asset.original_path)
    except (OSError, ValueError, TypeError):
        video_path = None
    if video_path is None or not video_path.is_file():
        asset.status = AssetStatus.PROCESSING
        asset.processing_started_at = datetime.now(timezone.utc)
        asset.error_message = None
        db.commit()
        return _mark_failed(
            db, asset, f"Video file not found: {asset.original_path}"
        )

    if not ffmpeg_available() or not ffprobe_available():
        asset.status = AssetStatus.PROCESSING
        asset.processing_started_at = datetime.now(timezone.utc)
        asset.error_message = None
        db.commit()
        return _mark_failed(
            db,
            asset,
            "ffmpeg/ffprobe not available on this system; "
            "install ffmpeg to enable video processing.",
        )

    asset.status = AssetStatus.PROCESSING
    asset.processing_started_at = datetime.now(timezone.utc)
    asset.error_message = None
    db.commit()

    try:
        duration = get_video_duration(video_path, stored=asset.duration_seconds)
        timestamps = compute_frame_timestamps(duration)
        with tempfile.TemporaryDirectory(prefix="dam-frames-") as tmp:
            frames = extract_frames(video_path, timestamps, Path(tmp))
            notes = [
                describe_frame(frame, i, ts, asset)
                for i, (frame, ts) in enumerate(zip(frames, timestamps))
            ]
            frame_vec = embed_frame_files(frames)
        asset.description = combine_frame_descriptions(
            asset.filename, duration, FRAME_INTERVAL_SECONDS, notes
        )
        if frame_vec is not None:
            # Genuine visual embedding: mean-pooled CLIP image tower
            # features over the sampled frames (same joint space as
            # text queries).
            asset.embedding = frame_vec
        else:
            # Same embedding provider as asset processing + search queries.
            asset.embedding = embed_text(
                f"{asset.filename}\n{humanize_stem(asset.filename)}\n{asset.description}"
            )
        asset.status = AssetStatus.COMPLETED
        asset.processing_completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:  # one bad video must not stop the batch
        logger.exception("Video processing failed for %s", asset.relative_path)
        return _mark_failed(db, asset, str(exc) or "Video processing failed")
    return asset


def process_pending_videos(db: Session, limit: int = 100) -> dict:
    """Process video assets: pending -> completed/failed, rest skipped."""
    candidates = (
        db.query(Asset)
        .filter(Asset.file_type == FileType.VIDEO)
        .order_by(Asset.last_indexed_at.asc())
        .limit(limit)
        .all()
    )
    processed = 0
    failed = 0
    skipped = 0
    errors: list[str] = []
    for asset in candidates:
        if asset.status == AssetStatus.COMPLETED:
            skipped += 1
            continue
        if asset.status != AssetStatus.PENDING:
            skipped += 1
            continue
        result = process_video_asset(db, asset)
        if result.status == AssetStatus.COMPLETED:
            processed += 1
        elif result.status == AssetStatus.FAILED:
            failed += 1
            errors.append(
                f"{result.relative_path}: {result.error_message or 'failed'}"
            )
        else:  # defensive; should not happen
            skipped += 1
    return {
        "total": len(candidates),
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }
