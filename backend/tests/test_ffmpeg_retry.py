"""Regression test: retry videos failed due to missing ffmpeg/ffprobe.

A video stored as failed with "ffmpeg/ffprobe not available" must be
requeued to pending on the next Start Indexing run once the tooling is
available. Unrelated failures must stay failed.
"""

from datetime import datetime, timezone
from pathlib import Path

import app.services.video as video_svc
from app.core.config import settings
from app.models import Asset, AssetStatus, FileType
from app.services.indexing import reset_index_status
from app.services.scanner import compute_sha256

import pytest


@pytest.fixture(autouse=True)
def _fresh_status():
    reset_index_status()
    yield
    reset_index_status()


def _make_fake_png(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color="green").save(path)
    return path


def _patch_video_success(monkeypatch):
    monkeypatch.setattr(video_svc, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(video_svc, "ffprobe_available", lambda: True)
    monkeypatch.setattr(
        video_svc,
        "get_video_duration",
        lambda path, stored=None: stored if stored else 25.0,
    )

    def fake_extract(video_path, timestamps, tmpdir):
        return [_make_fake_png(tmpdir / f"frame_{i:03d}.jpg") for i, _ in enumerate(timestamps)]

    monkeypatch.setattr(video_svc, "extract_frames", fake_extract)


def _seed_failed_asset(db, path: Path, dataset_root: Path, *, error: str, file_type: FileType):
    absolute = path.resolve()
    relative = absolute.relative_to(dataset_root.resolve()).as_posix()
    asset = Asset(
        filename=path.name,
        original_path=str(absolute),
        relative_path=relative,
        file_type=file_type,
        mime_type="video/mp4" if file_type is FileType.VIDEO else "image/png",
        file_size=path.stat().st_size,
        file_hash=compute_sha256(path),
        status=AssetStatus.FAILED,
        error_message=error,
        last_indexed_at=datetime.now(timezone.utc),
        duration_seconds=25.0 if file_type is FileType.VIDEO else None,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def test_retry_ffmpeg_unavailable_video_after_tooling_available(
    client, db, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))
    _patch_video_success(monkeypatch)

    retry_path = tmp_path / "videos" / "retry.mp4"
    retry_path.parent.mkdir(parents=True, exist_ok=True)
    retry_path.write_bytes(b"fake-video-bytes-retry")

    corrupt_path = tmp_path / "videos" / "corrupt.mp4"
    corrupt_path.write_bytes(b"fake-video-bytes-corrupt")

    bad_image_path = tmp_path / "images" / "bad.png"
    bad_image_path.parent.mkdir(parents=True, exist_ok=True)
    bad_image_path.write_bytes(b"not-image-but-failed-row")

    _seed_failed_asset(
        db,
        retry_path,
        tmp_path,
        error="ffmpeg/ffprobe not available on this system; "
        "install ffmpeg to enable video processing.",
        file_type=FileType.VIDEO,
    )
    _seed_failed_asset(
        db,
        corrupt_path,
        tmp_path,
        error="moov atom not found: Invalid data found when processing input",
        file_type=FileType.VIDEO,
    )
    _seed_failed_asset(
        db,
        bad_image_path,
        tmp_path,
        error="mocked processing boom",
        file_type=FileType.IMAGE,
    )

    response = client.post("/api/index")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"

    retry_row = db.query(Asset).filter(Asset.relative_path == "videos/retry.mp4").one()
    assert retry_row.status is AssetStatus.COMPLETED
    assert retry_row.error_message is None
    assert retry_row.description is not None

    # Unrelated failures must not be blindly retried.
    corrupt_row = db.query(Asset).filter(Asset.relative_path == "videos/corrupt.mp4").one()
    assert corrupt_row.status is AssetStatus.FAILED
    assert "moov atom not found" in (corrupt_row.error_message or "")

    bad_row = db.query(Asset).filter(Asset.relative_path == "images/bad.png").one()
    assert bad_row.status is AssetStatus.FAILED
