"""Reliability tests: duplicates, isolation, deleted files, API errors.

Covers the failure modes not already exercised by the per-area suites.
No models, no network, no new framework.
"""

import os

import pytest

import app.services.ai as ai_svc
from app.core.config import settings
from app.models import Asset, AssetStatus
from app.services.embeddings import deterministic_embedding


def make_png(path, size=(64, 48), color="red"):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def use_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))
    return tmp_path


# --- 1. duplicates ---------------------------------------------------------------


def test_identical_files_all_visible_job_completes(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "a.png", size=(64, 48), color="red")
    for name in ("b.png", "c.png"):
        (root / "images" / name).write_bytes((root / "images" / "a.png").read_bytes())

    body = client.post("/api/index").json()

    assert body["status"] == "completed"
    assert body["duplicates"] == 2
    rows = db.query(Asset).order_by(Asset.relative_path).all()
    assert len(rows) == 3  # every duplicate remains visible
    assert [r.status for r in rows].count(AssetStatus.DUPLICATE) == 2
    assert [r.status for r in rows].count(AssetStatus.COMPLETED) == 1
    listed = client.get("/api/assets").json()
    assert listed["total"] == 3


# --- 3. failure isolation -----------------------------------------------------------


def test_corrupt_image_isolated_in_full_job(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    (root / "images" / "bad.png").parent.mkdir(parents=True, exist_ok=True)
    (root / "images" / "bad.png").write_bytes(b"this is not image data")
    make_png(root / "images" / "good.png", size=(16, 16), color="blue")

    body = client.post("/api/index").json()

    assert body["status"] == "completed"
    good = db.query(Asset).filter(Asset.relative_path == "images/good.png").one()
    bad = db.query(Asset).filter(Asset.relative_path == "images/bad.png").one()
    assert good.status is AssetStatus.COMPLETED
    assert bad.status is AssetStatus.FAILED
    assert bad.error_message  # useful message persisted
    assert bad.last_indexed_at is not None
    # Failed asset stays visible in listing + stats.
    assert client.get("/api/assets").json()["total"] == 2
    assert client.get("/api/assets/stats").json()["failed"] == 1


def test_embedding_failure_isolated(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "good.png", size=(64, 48), color="red")
    make_png(root / "images" / "bad.png", size=(32, 16), color="blue")

    def flaky_embed(text, provider_name=None):
        if "bad" in text:
            raise RuntimeError("mocked embedding boom")
        return deterministic_embedding(text)

    def flaky_embed_image(image, provider_name=None):
        # Images embed from pixels; fail the 32x16 "bad" image so the
        # text fallback (also flaky for "bad") records the failure.
        if tuple(image.size) == (32, 16):
            raise RuntimeError("mocked image embedding boom")
        return deterministic_embedding(f"img-{image.size[0]}x{image.size[1]}")

    monkeypatch.setattr(ai_svc, "embed_text", flaky_embed)
    monkeypatch.setattr(ai_svc, "embed_image", flaky_embed_image)
    body = client.post("/api/index").json()

    assert body["status"] == "completed"
    assert body["processing_failed"] == 1
    good = db.query(Asset).filter(Asset.relative_path == "images/good.png").one()
    bad = db.query(Asset).filter(Asset.relative_path == "images/bad.png").one()
    assert good.status is AssetStatus.COMPLETED and good.embedding is not None
    assert bad.status is AssetStatus.FAILED
    assert "mocked embedding boom" in (bad.error_message or "")
    assert bad.processing_started_at is not None
    assert bad.processing_completed_at is not None


def test_pdf_extraction_failure_isolated(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    pdf = root / "documents" / "doc.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    make_png(root / "images" / "good.png")

    def boom(path, limit=8000):
        raise RuntimeError("mocked pdf boom")

    monkeypatch.setattr(ai_svc, "extract_pdf_text", boom)
    body = client.post("/api/index").json()

    assert body["status"] == "completed"
    doc = db.query(Asset).filter(Asset.relative_path == "documents/doc.pdf").one()
    good = db.query(Asset).filter(Asset.relative_path == "images/good.png").one()
    assert doc.status is AssetStatus.FAILED
    assert "mocked pdf boom" in (doc.error_message or "")
    assert good.status is AssetStatus.COMPLETED


def test_video_failure_isolated_in_full_job(client, db, tmp_path, monkeypatch):
    import app.services.video as video_svc

    root = use_dataset(monkeypatch, tmp_path)
    clip = root / "videos" / "clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    make_png(root / "images" / "good.png")
    # Deterministic regardless of host tooling.
    monkeypatch.setattr(video_svc, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(video_svc, "ffprobe_available", lambda: False)

    body = client.post("/api/index").json()

    assert body["status"] == "completed"
    clip_row = db.query(Asset).filter(Asset.relative_path == "videos/clip.mp4").one()
    good = db.query(Asset).filter(Asset.relative_path == "images/good.png").one()
    assert clip_row.status is AssetStatus.FAILED
    assert "ffmpeg" in (clip_row.error_message or "").lower()
    assert good.status is AssetStatus.COMPLETED


def test_unexpected_per_file_error_does_not_abort_scan(db, tmp_path, monkeypatch):
    import app.services.scanner as scanner

    make_png(tmp_path / "images" / "good.png")
    make_png(tmp_path / "images" / "weird.png", size=(16, 16), color="blue")
    original = scanner._index_supported_file

    def flaky(db_session, path, root, result):
        if path.name == "weird.png":
            raise ValueError("mocked surprise")
        return original(db_session, path, root, result)

    monkeypatch.setattr(scanner, "_index_supported_file", flaky)
    result = scanner.scan_dataset(db, tmp_path)

    assert result.failed == 1
    assert any("weird.png" in e for e in result.errors)
    good = db.query(Asset).filter(Asset.relative_path == "images/good.png").one()
    assert good.status is AssetStatus.PENDING


def test_symlink_escape_does_not_crash_job(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    outside = tmp_path.parent / "outside-secret.png"
    outside.write_bytes(b"not an image")
    link = root / "images" / "link.png"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(outside), str(link))
    except OSError:
        pytest.skip("symlinks not permitted on this host")
    make_png(root / "images" / "good.png")

    body = client.post("/api/index").json()

    assert body["status"] == "completed"
    good = db.query(Asset).filter(Asset.relative_path == "images/good.png").one()
    assert good.status is AssetStatus.COMPLETED


# --- 7. deleted files ------------------------------------------------------------------


def test_deleted_file_row_persists_job_stays_green(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = make_png(root / "images" / "a.png")
    make_png(root / "images" / "b.png", size=(16, 16), color="blue")
    assert client.post("/api/index").status_code == 200

    target.unlink()  # file deleted from the dataset afterwards
    body = client.post("/api/index").json()

    assert body["status"] == "completed"  # no crash, no auto-delete
    rows = {r.relative_path: r for r in db.query(Asset).all()}
    assert set(rows) == {"images/a.png", "images/b.png"}
    assert client.get("/api/assets").json()["total"] == 2
    # Preview reports the missing file cleanly instead of crashing.
    file_res = client.get(f"/api/assets/{rows['images/a.png'].id}/file")
    assert file_res.status_code == 404


# --- 9. API error codes -------------------------------------------------------------------


def test_api_error_codes_are_clean(client, db, tmp_path, monkeypatch):
    import uuid

    use_dataset(monkeypatch, tmp_path)

    res = client.get("/api/search", params={"file_type": "not-a-type", "q": "x"})
    assert res.status_code == 422

    res = client.get(f"/api/assets/{uuid.uuid4()}")
    assert res.status_code == 404
    assert "detail" in res.json()

    res = client.get(f"/api/assets/{uuid.uuid4()}/file")
    assert res.status_code == 404

    res = client.post("/api/index")
    assert res.status_code == 200  # empty dataset is valid, not an error
