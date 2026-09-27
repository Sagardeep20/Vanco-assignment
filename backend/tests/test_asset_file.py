"""Tests for asset file serving + open-original (no real OS opens)."""

import uuid
from pathlib import Path

from app.core.config import settings
from app.models import Asset, AssetStatus, FileType
from app.services import files as file_service


def make_png(path: Path, size=(32, 24), color="green") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def add_asset(db, target: Path, relative: str, file_type: FileType,
              mime: str | None = None, status: AssetStatus = AssetStatus.COMPLETED,
              original: str | None = None) -> Asset:
    asset = Asset(
        filename=target.name if original is None else Path(original).name,
        original_path=str(target.resolve()) if original is None else original,
        relative_path=relative,
        file_type=file_type,
        mime_type=mime,
        file_size=target.stat().st_size if target.exists() else 123,
        status=status,
        description="Test description",
        extracted_text=None,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def use_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))
    return tmp_path


# --- valid asset file ----------------------------------------------------------


def test_serve_valid_image(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = make_png(root / "images" / "a.png")
    asset = add_asset(db, target, "images/a.png", FileType.IMAGE, mime="image/png")

    response = client.get(f"/api/assets/{asset.id}/file")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == target.read_bytes()


def test_serve_uses_extension_guess_without_stored_mime(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = root / "documents" / "doc.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"%PDF-1.4\n%EOF\n")
    asset = add_asset(db, target, "documents/doc.pdf", FileType.PDF, mime=None)

    response = client.get(f"/api/assets/{asset.id}/file")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_serve_video_media_type(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = root / "videos" / "c.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    asset = add_asset(db, target, "videos/c.mp4", FileType.VIDEO, mime="video/mp4")

    response = client.get(f"/api/assets/{asset.id}/file")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"


# --- missing asset / missing file -----------------------------------------------


def test_missing_asset_returns_404(client, db):
    response = client.get(f"/api/assets/{uuid.uuid4()}/file")
    assert response.status_code == 404


def test_missing_physical_file_returns_404(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    ghost = root / "images" / "gone.png"  # never created
    asset = add_asset(db, ghost, "images/gone.png", FileType.IMAGE, mime="image/png")

    response = client.get(f"/api/assets/{asset.id}/file")

    assert response.status_code == 404


# --- path traversal protection -----------------------------------------------------


def test_path_traversal_refused(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("top secret")
    asset = add_asset(
        db, outside, "images/evil.png", FileType.IMAGE,
        mime="image/png", original=str(outside.resolve()),
    )
    assert Path(asset.original_path).is_file()  # file exists, but outside dataset

    response = client.get(f"/api/assets/{asset.id}/file")

    assert response.status_code == 400
    assert "outside" in response.json()["detail"].lower()


def test_no_user_controlled_path_parameter(client, db, tmp_path, monkeypatch):
    """There is no endpoint that serves an arbitrary path string."""
    use_dataset(monkeypatch, tmp_path)
    response = client.get("/api/assets/../../etc/passwd/file")
    assert response.status_code in (404, 422)


# --- detail endpoint carries preview fields ------------------------------------------


def test_asset_detail_has_preview_fields(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = make_png(root / "images" / "a.png")
    asset = add_asset(db, target, "images/a.png", FileType.IMAGE, mime="image/png")

    body = client.get(f"/api/assets/{asset.id}").json()

    for field in ("filename", "file_type", "file_size", "relative_path",
                  "original_path", "description", "extracted_text", "status",
                  "mime_type"):
        assert field in body
    assert body["original_path"] == str(target.resolve())


# --- open-original ---------------------------------------------------------------------


def test_open_original_success_mocked(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = make_png(root / "images" / "a.png")
    asset = add_asset(db, target, "images/a.png", FileType.IMAGE, mime="image/png")

    opened = {}
    monkeypatch.setattr(
        file_service, "open_in_os",
        lambda path: opened.setdefault("path", str(path)),
    )

    response = client.post(f"/api/assets/{asset.id}/open")

    assert response.status_code == 200
    body = response.json()
    assert body["opened"] is True
    assert opened["path"] == str(target.resolve())


def test_open_original_missing_file_404(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    ghost = root / "images" / "gone.png"
    asset = add_asset(db, ghost, "images/gone.png", FileType.IMAGE)

    response = client.post(f"/api/assets/{asset.id}/open")

    assert response.status_code == 404


def test_open_original_failure_returns_not_opened(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = make_png(root / "images" / "a.png")
    asset = add_asset(db, target, "images/a.png", FileType.IMAGE)

    def boom(path):
        raise OSError("no desktop here")

    monkeypatch.setattr(file_service, "open_in_os", boom)

    response = client.post(f"/api/assets/{asset.id}/open")

    assert response.status_code == 200
    assert response.json()["opened"] is False
