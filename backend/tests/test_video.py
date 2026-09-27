"""Tests for video processing (mocked tooling/vision, no AI model).

ffmpeg/ffprobe and the vision provider are always mocked: these tests
must pass on machines without ffmpeg and must never download a model.
"""

from pathlib import Path

import app.services.video as video_svc
from app.models import Asset, AssetStatus, FileType
from app.services.video import (
    combine_frame_descriptions,
    compute_frame_timestamps,
    process_pending_videos,
    process_video_asset,
)


def make_video_asset(db, tmp_path, name="clip.mp4", status=AssetStatus.PENDING,
                     duration=25.0, content=b"fake-video-bytes"):
    """Create a pending video Asset backed by a real (fake-content) file."""
    target = tmp_path / "videos" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    asset = Asset(
        filename=name,
        original_path=str(target.resolve()),
        relative_path=f"videos/{name}",
        file_type=FileType.VIDEO,
        mime_type="video/mp4",
        file_size=target.stat().st_size,
        status=status,
        duration_seconds=duration,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def make_fake_png(path: Path, size=(32, 24), color="green") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def patch_success(monkeypatch, record=None):
    """Mock tooling: available, fixed duration, real PNGs as 'frames'."""
    monkeypatch.setattr(video_svc, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(video_svc, "ffprobe_available", lambda: True)
    monkeypatch.setattr(
        video_svc, "get_video_duration",
        lambda path, stored=None: stored if stored else 25.0,
    )

    def fake_extract(video_path, timestamps, tmpdir):
        if record is not None:
            record["tmpdir"] = str(tmpdir)
        frames = []
        for i, _ in enumerate(timestamps):
            frames.append(make_fake_png(tmpdir / f"frame_{i:03d}.jpg"))
        return frames

    monkeypatch.setattr(video_svc, "extract_frames", fake_extract)


# --- timestamp sampling ------------------------------------------------------


def test_timestamps_unknown_duration_single_frame():
    assert compute_frame_timestamps(None) == [1.0]
    assert compute_frame_timestamps(0) == [1.0]


def test_timestamps_short_video_single_frame():
    assert len(compute_frame_timestamps(5.0)) == 1


def test_timestamps_long_video_capped():
    stamps = compute_frame_timestamps(300.0)
    assert len(stamps) == video_svc.MAX_FRAMES
    assert stamps == sorted(stamps)
    assert all(0 < t < 300.0 for t in stamps)


# --- processing --------------------------------------------------------------


def test_successful_video_processing(db, tmp_path, monkeypatch):
    record = {}
    patch_success(monkeypatch, record)
    asset = make_video_asset(db, tmp_path)

    result = process_video_asset(db, asset)

    assert result.status is AssetStatus.COMPLETED
    assert result.processing_started_at is not None
    assert result.processing_completed_at is not None
    assert result.error_message is None
    assert result.description is not None
    assert "clip" in result.description.lower()
    assert "Frame observations" in result.description
    # Frames are temporary only: the temp dir is removed after processing.
    assert not Path(record["tmpdir"]).exists()


def test_description_saved_to_db(db, tmp_path, monkeypatch):
    patch_success(monkeypatch)
    asset = make_video_asset(db, tmp_path, name="saved.mp4")
    process_video_asset(db, asset)

    db.expire_all()
    row = db.query(Asset).filter(Asset.relative_path == "videos/saved.mp4").one()
    assert row.status is AssetStatus.COMPLETED
    assert row.description and "saved" in row.description.lower()


def test_failed_video_processing_does_not_stop_batch(db, tmp_path, monkeypatch):
    monkeypatch.setattr(video_svc, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(video_svc, "ffprobe_available", lambda: True)
    monkeypatch.setattr(video_svc, "get_video_duration", lambda p, stored=None: 20.0)
    good = make_video_asset(db, tmp_path, name="good.mp4")
    bad = make_video_asset(db, tmp_path, name="bad.mp4", content=b"different-bytes")

    def selective_extract(video_path, timestamps, tmpdir):
        if "bad.mp4" in str(video_path):
            raise RuntimeError("boom-extract")
        return [make_fake_png(tmpdir / "frame_000.jpg")]

    monkeypatch.setattr(video_svc, "extract_frames", selective_extract)

    outcome = process_pending_videos(db, limit=10)

    assert outcome["processed"] == 1
    assert outcome["failed"] == 1
    assert outcome["total"] == 2
    db.refresh(good)
    db.refresh(bad)
    assert good.status is AssetStatus.COMPLETED
    assert bad.status is AssetStatus.FAILED
    assert "boom-extract" in (bad.error_message or "")


def test_missing_video_file_failed(db, tmp_path, monkeypatch):
    monkeypatch.setattr(video_svc, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(video_svc, "ffprobe_available", lambda: True)
    asset = Asset(
        filename="gone.mp4",
        original_path=str((tmp_path / "videos" / "gone.mp4").resolve()),
        relative_path="videos/gone.mp4",
        file_type=FileType.VIDEO,
        status=AssetStatus.PENDING,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)

    result = process_video_asset(db, asset)

    assert result.status is AssetStatus.FAILED
    assert "not found" in (result.error_message or "").lower()
    assert result.processing_completed_at is not None


def test_already_completed_video_skipped(db, tmp_path, monkeypatch):
    patch_success(monkeypatch)
    asset = make_video_asset(
        db, tmp_path, name="done.mp4", status=AssetStatus.COMPLETED,
    )
    asset.description = "Original description"
    db.commit()

    result = process_video_asset(db, asset)

    assert result.status is AssetStatus.COMPLETED
    assert result.description == "Original description"
    assert result.processing_started_at is None  # untouched

    outcome = process_pending_videos(db, limit=10)
    assert outcome == {
        "total": 1, "processed": 0, "failed": 0, "skipped": 1, "errors": [],
    }


def test_non_video_asset_untouched(db, tmp_path):
    target = tmp_path / "images" / "a.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"png-bytes")
    asset = Asset(
        filename="a.png",
        original_path=str(target.resolve()),
        relative_path="images/a.png",
        file_type=FileType.IMAGE,
        status=AssetStatus.PENDING,
    )
    db.add(asset)
    db.commit()

    result = process_video_asset(db, asset)
    assert result.status is AssetStatus.PENDING
    assert result.description is None


def test_ffmpeg_unavailable_marks_failed_gracefully(db, tmp_path, monkeypatch):
    monkeypatch.setattr(video_svc, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(video_svc, "ffprobe_available", lambda: False)
    make_video_asset(db, tmp_path, name="one.mp4")
    make_video_asset(db, tmp_path, name="two.mp4", content=b"other-bytes")

    outcome = process_pending_videos(db, limit=10)

    assert outcome["total"] == 2
    assert outcome["processed"] == 0
    assert outcome["failed"] == 2
    assert outcome["skipped"] == 0
    rows = db.query(Asset).all()
    assert all(r.status is AssetStatus.FAILED for r in rows)
    assert all("ffmpeg" in (r.error_message or "").lower() for r in rows)


def test_combine_frame_descriptions_format():
    text = combine_frame_descriptions("beach-day.mp4", 25.0, 10.0, ["sunny", "crowd"])
    assert "beach day" in text.lower()
    assert "25.0s" in text
    assert "[1] sunny" in text and "[2] crowd" in text


# --- endpoint -----------------------------------------------------------------


def test_videos_endpoint_statistics(client, db, tmp_path, monkeypatch):
    patch_success(monkeypatch)
    make_video_asset(db, tmp_path, name="a.mp4", duration=12.0)
    make_video_asset(db, tmp_path, name="b.mp4", duration=30.0,
                     content=b"other-bytes")
    done = make_video_asset(db, tmp_path, name="c.mp4",
                            status=AssetStatus.COMPLETED, content=b"third-bytes")
    done.description = "Already done"
    db.commit()

    # A non-video row must not affect video statistics.
    img = tmp_path / "images" / "x.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"img")
    db.add(Asset(
        filename="x.png", original_path=str(img.resolve()),
        relative_path="images/x.png", file_type=FileType.IMAGE,
        status=AssetStatus.PENDING,
    ))
    db.commit()

    response = client.post("/api/process/videos", json={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["total"] == 3
    assert body["processed"] == 2
    assert body["failed"] == 0
    assert body["skipped"] == 1
    assert body["errors"] == []
