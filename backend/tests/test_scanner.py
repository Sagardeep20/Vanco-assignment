"""Tests for the file ingestion/scanning layer."""

import hashlib
from pathlib import Path

import pytest

from app.core.config import settings
from app.models import Asset, AssetStatus, FileType
from app.services.scanner import (
    classify_file,
    compute_sha256,
    extract_file_metadata,
    extract_image_size,
    probe_video,
    scan_dataset,
)


def make_png(path: Path, size=(64, 48), color="red") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def make_dataset(root: Path) -> dict[str, Path]:
    """Build a small nested dataset mirroring data/images|videos|documents."""
    files = {
        "image": make_png(root / "images" / "photo.png"),
        "nested_image": make_png(root / "images" / "nested" / "pic.jpg", size=(32, 16)),
        "video": root / "videos" / "clip.mp4",
        "pdf": root / "documents" / "doc.pdf",
        "unsupported": root / "notes.txt",
        "hidden": root / "images" / ".gitkeep",
    }
    files["video"].parent.mkdir(parents=True, exist_ok=True)
    files["video"].write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024)
    files["pdf"].parent.mkdir(parents=True, exist_ok=True)
    files["pdf"].write_bytes(b"%PDF-1.4\n%EOF\n")
    files["unsupported"].write_text("plain notes")
    files["hidden"].write_text("")
    return files


# --- supported / unsupported detection ------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.jpg", FileType.IMAGE),
        ("a.JPEG", FileType.IMAGE),
        ("a.png", FileType.IMAGE),
        ("a.webp", FileType.IMAGE),
        ("a.mp4", FileType.VIDEO),
        ("a.mov", FileType.VIDEO),
        ("a.mkv", FileType.VIDEO),
        ("a.webm", FileType.VIDEO),
        ("a.pdf", FileType.PDF),
    ],
)
def test_classify_supported(name, expected):
    assert classify_file(Path(name)) is expected


@pytest.mark.parametrize("name", ["a.txt", "a.exe", "a", "a.tiff", ".gitkeep"])
def test_classify_unsupported(name):
    assert classify_file(Path(name)) is None


# --- hashing / metadata ----------------------------------------------------


def test_compute_sha256(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"hello-assets")
    assert compute_sha256(target) == hashlib.sha256(b"hello-assets").hexdigest()


def test_extract_image_size(tmp_path):
    target = make_png(tmp_path / "img.png", size=(64, 48))
    assert extract_image_size(target) == (64, 48)


def test_probe_video_handles_missing_ffprobe(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.scanner.shutil.which", lambda _: None)
    assert probe_video(tmp_path / "clip.mp4") == (None, None, None)


def test_probe_video_handles_garbage_file(tmp_path):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"not a video")
    # Must not raise even though ffprobe (if present) cannot parse it.
    assert probe_video(target) == (None, None, None)


def test_extract_file_metadata(tmp_path):
    target = make_png(tmp_path / "images" / "a.png")
    meta = extract_file_metadata(target, tmp_path)
    assert meta.filename == "a.png"
    assert meta.relative_path == "images/a.png"
    assert meta.file_type is FileType.IMAGE
    assert meta.mime_type == "image/png"
    assert meta.file_size == target.stat().st_size
    assert meta.file_hash == compute_sha256(target)
    assert meta.created_at is not None and meta.modified_at is not None


# --- full scan against Postgres --------------------------------------------


def test_scan_temp_dataset(db, tmp_path):
    make_dataset(tmp_path)
    result = scan_dataset(db, tmp_path)

    assert result.discovered == 5  # hidden .gitkeep is skipped, not discovered
    assert result.added == 4
    assert result.unsupported == 1
    assert result.failed == 0
    assert result.skipped == 1

    rows = {a.relative_path: a for a in db.query(Asset).all()}
    assert set(rows) == {
        "images/photo.png",
        "images/nested/pic.jpg",
        "videos/clip.mp4",
        "documents/doc.pdf",
        "notes.txt",
    }

    photo = rows["images/photo.png"]
    assert photo.file_type is FileType.IMAGE
    assert photo.status is AssetStatus.PENDING
    assert (photo.width, photo.height) == (64, 48)
    assert photo.mime_type == "image/png"
    assert photo.embedding is None

    nested = rows["images/nested/pic.jpg"]
    assert (nested.width, nested.height) == (32, 16)

    pdf = rows["documents/doc.pdf"]
    assert pdf.file_type is FileType.PDF
    assert pdf.width is None and pdf.duration_seconds is None
    assert pdf.file_hash == compute_sha256(tmp_path / "documents" / "doc.pdf")

    video = rows["videos/clip.mp4"]
    assert video.file_type is FileType.VIDEO
    assert video.status is AssetStatus.PENDING  # no crash without ffprobe data

    unsupported = rows["notes.txt"]
    assert unsupported.file_type is FileType.OTHER
    assert unsupported.status is AssetStatus.UNSUPPORTED


def test_scan_is_idempotent(db, tmp_path):
    make_dataset(tmp_path)
    first = scan_dataset(db, tmp_path)
    second = scan_dataset(db, tmp_path)

    assert (first.added, first.updated) == (4, 0)
    assert (second.added, second.updated, second.failed) == (0, 0, 0)
    assert second.skipped == 6  # 4 supported + 1 unsupported + 1 hidden
    assert db.query(Asset).count() == 5


def test_scan_updates_changed_file(db, tmp_path):
    files = make_dataset(tmp_path)
    scan_dataset(db, tmp_path)

    make_png(files["image"], size=(10, 10), color="blue")  # change content
    result = scan_dataset(db, tmp_path)

    assert result.updated == 1
    row = db.query(Asset).filter(Asset.relative_path == "images/photo.png").one()
    assert (row.width, row.height) == (10, 10)
    assert row.status is AssetStatus.PENDING
    assert row.embedding is None


def test_scan_marks_duplicate_content(db, tmp_path):
    files = make_dataset(tmp_path)
    scan_dataset(db, tmp_path)

    duplicate = tmp_path / "images" / "photo-copy.png"
    duplicate.write_bytes(files["image"].read_bytes())
    result = scan_dataset(db, tmp_path)

    assert result.duplicates == 1
    row = db.query(Asset).filter(Asset.relative_path == "images/photo-copy.png").one()
    assert row.status is AssetStatus.DUPLICATE
    assert row.file_hash == compute_sha256(files["image"])


def test_scan_missing_directory(db, tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(db, tmp_path / "does-not-exist")


# --- API endpoint -----------------------------------------------------------


def test_index_endpoint(client, db, tmp_path, monkeypatch):
    make_dataset(tmp_path)
    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))

    response = client.post("/api/index")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["added"] == 4
    assert body["unsupported"] == 1
    assert body["failed"] == 0
    assert db.query(Asset).count() == 5


def test_index_endpoint_missing_directory(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "dataset_path", str(tmp_path / "missing"))
    response = client.post("/api/index")
    assert response.status_code == 400
