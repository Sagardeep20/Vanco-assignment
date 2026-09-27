"""Tests for the local-only AI pipeline (no model downloads)."""

import math
from pathlib import Path

from app.models import Asset, AssetStatus, FileType
from app.models.asset import EMBEDDING_DIMENSION
from app.services.ai import (
    build_searchable_text,
    extract_pdf_text,
    generate_description,
    generate_embedding,
    process_asset,
    process_pending_assets,
)
from app.services.scanner import scan_dataset


def make_png(path: Path, size=(64, 48), color="red") -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def make_pdf(path: Path, text: str = "hello asset world") -> Path:
    from pypdf import PdfWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # pypdf writer alone has no text; write minimal PDF with text via raw stream
    # Simplest reliable: create with reportlab if present, else blank (no text).
    try:
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(path))
        c.drawString(50, 150, text)
        c.save()
    except ImportError:
        with open(path, "wb") as f:
            writer.write(f)
    return path


# --- embeddings -------------------------------------------------------------


def test_embedding_dim_and_normalized():
    vec = generate_embedding("sunset beach photo")
    assert vec is not None
    assert len(vec) == EMBEDDING_DIMENSION == 512
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_embedding_deterministic():
    assert generate_embedding("same text here") == generate_embedding("same text here")


def test_embedding_blank_returns_none():
    assert generate_embedding("") is None
    assert generate_embedding("   ") is None


def test_embedding_similar_texts_closer():
    from app.services.search import vector_search  # noqa: F401 (ensures import ok)

    a = generate_embedding("sunset beach ocean photo")
    b = generate_embedding("sunset beach vacation photo")
    c = generate_embedding("quantum database replication index")
    assert a is not None and b is not None and c is not None

    def cosine(u, v):
        return sum(x * y for x, y in zip(u, v))

    assert cosine(a, b) > cosine(a, c)


# --- descriptions ------------------------------------------------------------


def test_description_image_mentions_dimensions():
    desc = generate_description("sunset.png", FileType.IMAGE, width=64, height=48, file_size=1234)
    assert "64x48" in desc
    assert "sunset" in desc.lower()


def test_description_video_mentions_duration():
    desc = generate_description("clip.mp4", FileType.VIDEO, width=1280, height=720,
                                duration_seconds=12.5, file_size=2048)
    assert "12.5" in desc
    assert "clip" in desc.lower()


def test_description_pdf_with_text_snippet():
    desc = generate_description("report.pdf", FileType.PDF, file_size=512,
                                extracted_text="annual revenue grew", pdf_pages=3)
    assert "3 pages" in desc
    assert "annual revenue" in desc


def test_build_searchable_text_combines_fields():
    text = build_searchable_text("photo.png", "A nice photo", "extra words here")
    assert "photo.png" in text and "A nice photo" in text and "extra words" in text


# --- PDF extraction ----------------------------------------------------------


def test_extract_pdf_text_garbage_returns_none(tmp_path):
    target = tmp_path / "bad.pdf"
    target.write_bytes(b"not a pdf at all")
    text, pages = extract_pdf_text(target)
    assert text is None


def test_extract_pdf_text_missing_pypdf_graceful(tmp_path, monkeypatch):
    import builtins

    target = tmp_path / "a.pdf"
    target.write_bytes(b"%PDF-1.4\n%EOF\n")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    text, pages = extract_pdf_text(target)
    assert text is None and pages is None


# --- processing --------------------------------------------------------------


def test_process_asset_completes_image(db, tmp_path):
    target = make_png(tmp_path / "images" / "sunset.png")
    scan_dataset(db, tmp_path)
    asset = db.query(Asset).filter(Asset.relative_path == "images/sunset.png").one()
    assert asset.status is AssetStatus.PENDING

    process_asset(db, asset)

    assert asset.status is AssetStatus.COMPLETED
    assert asset.description and "sunset" in asset.description.lower()
    assert asset.embedding is not None and len(asset.embedding) == 512
    assert asset.processing_completed_at is not None


def test_process_pending_assets_batch(db, tmp_path):
    make_png(tmp_path / "images" / "a.png", size=(64, 48), color="red")
    make_png(tmp_path / "images" / "b.png", size=(32, 16), color="blue")
    scan_dataset(db, tmp_path)

    outcome = process_pending_assets(db, limit=10)
    assert outcome["processed"] == 2
    assert outcome["completed"] == 2
    assert outcome["failed"] == 0
    assert db.query(Asset).filter(Asset.status == AssetStatus.COMPLETED).count() == 2


def test_process_skips_duplicates_and_unsupported(db, tmp_path):
    target = make_png(tmp_path / "images" / "a.png")
    (tmp_path / "notes.txt").write_text("plain notes")
    scan_dataset(db, tmp_path)
    dup = tmp_path / "images" / "copy.png"
    dup.write_bytes(target.read_bytes())
    scan_dataset(db, tmp_path)

    outcome = process_pending_assets(db, limit=50)
    # Only the original pending image is processed; duplicate + unsupported stay.
    assert outcome["processed"] == 1
    dup_row = db.query(Asset).filter(Asset.relative_path == "images/copy.png").one()
    assert dup_row.status is AssetStatus.DUPLICATE
    unsup = db.query(Asset).filter(Asset.relative_path == "notes.txt").one()
    assert unsup.status is AssetStatus.UNSUPPORTED


def test_rescan_clears_stale_ai_outputs(db, tmp_path):
    target = make_png(tmp_path / "images" / "a.png", size=(64, 48))
    scan_dataset(db, tmp_path)
    asset = db.query(Asset).filter(Asset.relative_path == "images/a.png").one()
    process_asset(db, asset)
    assert asset.embedding is not None

    make_png(target, size=(10, 10), color="blue")  # change content
    scan_dataset(db, tmp_path)
    db.refresh(asset)
    assert asset.status is AssetStatus.PENDING
    assert asset.description is None
    assert asset.embedding is None


# --- API endpoints -----------------------------------------------------------


def test_assets_list_and_filter(client, db, tmp_path):
    from app.core.config import settings

    make_png(tmp_path / "images" / "a.png")
    (tmp_path / "videos" / "c.mp4").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "videos" / "c.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    scan_dataset(db, tmp_path)

    res = client.get("/api/assets")
    assert res.status_code == 200
    assert res.json()["total"] == 2

    res = client.get("/api/assets", params={"file_type": "image"})
    assert res.json()["total"] == 1

    res = client.get("/api/assets/stats")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2 and body["images"] == 1 and body["videos"] == 1


def test_process_endpoint(client, db, tmp_path):
    make_png(tmp_path / "images" / "a.png")
    scan_dataset(db, tmp_path)

    res = client.post("/api/process", json={"limit": 10})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "completed"
    assert body["completed"] == 1


def test_search_keyword_fallback_before_processing(client, db, tmp_path):
    make_png(tmp_path / "images" / "sunset-beach.png")
    scan_dataset(db, tmp_path)

    res = client.get("/api/search", params={"q": "sunset"})
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 1
    assert "keyword" in body["method"] or body["method"] == "vector"


def test_search_vector_after_processing(client, db, tmp_path):
    make_png(tmp_path / "images" / "sunset-beach.png")
    make_png(tmp_path / "images" / "report-scan.png", color="blue")
    scan_dataset(db, tmp_path)
    process_pending_assets(db, limit=10)

    res = client.get("/api/search", params={"q": "sunset beach"})
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 1
    assert body["method"] == "vector"
    top_path = body["results"][0]["asset"]["relative_path"]
    assert top_path == "images/sunset-beach.png"
