"""Tests for the indexing orchestration layer (no models, no new tables)."""

import pytest

import app.services.indexing as indexing
from app.core.config import settings
from app.models import Asset, AssetStatus
from app.services.indexing import (
    IndexingAlreadyRunningError,
    get_index_status,
    reset_index_status,
    run_indexing_job,
)


@pytest.fixture(autouse=True)
def _fresh_status():
    reset_index_status()
    yield
    reset_index_status()


def make_png(path, size=(64, 48), color="red"):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def use_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))
    return tmp_path


# --- 1. starting an indexing job ----------------------------------------------


def test_start_indexing_job(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "a.png")
    (root / "notes.txt").write_text("plain notes")

    response = client.post("/api/index")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["added"] == 1
    assert body["unsupported"] == 1
    assert body["processed"] == 1
    assert body["processing_completed"] == 1
    assert body["processing_failed"] == 0
    row = db.query(Asset).filter(Asset.relative_path == "images/a.png").one()
    assert row.status is AssetStatus.COMPLETED
    assert row.description is not None and row.embedding is not None


# --- 2. status endpoint ----------------------------------------------------------


def test_status_endpoint_after_job(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "a.png")
    (root / "notes.txt").write_text("plain notes")
    client.post("/api/index")

    response = client.get("/api/index/status")

    assert response.status_code == 200
    body = response.json()
    assert body["running"] is False
    assert body["stage"] == "completed"
    assert body["total_assets"] == 2
    assert body["completed"] == 1
    assert body["unsupported"] == 1
    assert body["pending"] == 0
    assert body["failed"] == 0
    assert body["dataset_path"] == str(root.resolve())
    assert body["started_at"] is not None
    assert body["completed_at"] is not None


# --- 3. duplicate job prevention --------------------------------------------------


def test_duplicate_job_prevention_endpoint(client, db, tmp_path, monkeypatch):
    use_dataset(monkeypatch, tmp_path)
    indexing._set_status(running=True, stage="processing")
    try:
        response = client.post("/api/index")
    finally:
        reset_index_status()

    assert response.status_code == 409
    assert "already running" in response.json()["detail"].lower()
    assert get_index_status()["running"] is False


def test_duplicate_job_prevention_service(db, tmp_path, monkeypatch):
    use_dataset(monkeypatch, tmp_path)
    indexing._set_status(running=True, stage="processing")
    try:
        with pytest.raises(IndexingAlreadyRunningError, match="already running"):
            run_indexing_job(db)
    finally:
        reset_index_status()

    # Flag was cleared: a real job still runs afterwards.
    summary = run_indexing_job(db)
    assert summary["status"] == "completed"


# --- 4. empty dataset ------------------------------------------------------------


def test_empty_dataset(client, db, tmp_path, monkeypatch):
    use_dataset(monkeypatch, tmp_path)

    response = client.post("/api/index")

    assert response.status_code == 200
    body = response.json()
    assert body["discovered"] == 0
    assert body["processed"] == 0
    status = client.get("/api/index/status").json()
    assert status["running"] is False
    assert status["stage"] == "completed"
    assert status["total_assets"] == 0


# --- 5. incremental indexing -------------------------------------------------------


def test_incremental_indexing_skips_unchanged(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "a.png")
    first = client.post("/api/index").json()
    assert first["processing_completed"] == 1

    row = db.query(Asset).filter(Asset.relative_path == "images/a.png").one()
    description, embedding = row.description, list(row.embedding)

    second = client.post("/api/index").json()

    assert (second["added"], second["updated"], second["failed"]) == (0, 0, 0)
    assert second["processed"] == 0
    assert second["processing_completed"] == 0
    db.refresh(row)
    assert row.status is AssetStatus.COMPLETED
    assert row.description == description  # not regenerated
    assert list(row.embedding) == embedding  # not regenerated


# --- 6. changed file reprocessing ---------------------------------------------------


def test_changed_file_reprocessed(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    target = make_png(root / "images" / "a.png", size=(64, 48))
    client.post("/api/index")
    row = db.query(Asset).filter(Asset.relative_path == "images/a.png").one()
    old_description, old_embedding = row.description, list(row.embedding)

    make_png(target, size=(10, 10), color="blue")  # change content
    body = client.post("/api/index").json()

    assert body["updated"] == 1
    assert body["processed"] == 1
    assert body["processing_completed"] == 1
    db.refresh(row)
    assert row.status is AssetStatus.COMPLETED
    assert row.description != old_description
    assert list(row.embedding) != old_embedding


# --- 7. one failed asset does not stop the job ---------------------------------------


def test_failed_asset_does_not_stop_job(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "good.png", size=(64, 48), color="red")
    make_png(root / "images" / "bad.png", size=(32, 16), color="blue")

    original = indexing.process_asset

    def flaky(db_session, asset):
        if asset.filename == "bad.png":
            raise RuntimeError("mocked processing boom")
        return original(db_session, asset)

    monkeypatch.setattr(indexing, "process_asset", flaky)
    body = client.post("/api/index").json()

    assert body["status"] == "completed"
    assert body["processed"] == 2
    assert body["processing_completed"] == 1
    assert body["processing_failed"] == 1
    good = db.query(Asset).filter(Asset.relative_path == "images/good.png").one()
    bad = db.query(Asset).filter(Asset.relative_path == "images/bad.png").one()
    assert good.status is AssetStatus.COMPLETED
    assert bad.status is AssetStatus.FAILED
    assert "mocked processing boom" in (bad.error_message or "")


# --- 8. final statistics -----------------------------------------------------------------


def test_final_statistics(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "a.png", size=(64, 48), color="red")
    make_png(root / "images" / "b.png", size=(32, 16), color="blue")
    (root / "images" / "copy.png").write_bytes(
        (root / "images" / "a.png").read_bytes()
    )
    (root / "documents" / "doc.pdf").parent.mkdir(parents=True, exist_ok=True)
    (root / "documents" / "doc.pdf").write_bytes(b"%PDF-1.4\n%EOF\n")
    (root / "notes.txt").write_text("plain notes")

    body = client.post("/api/index").json()

    assert body["discovered"] == 5
    assert body["unsupported"] == 1
    assert body["duplicates"] == 1
    assert body["processed"] == 3  # duplicate + unsupported are not pending
    assert body["processing_completed"] == 3
    assert body["processing_failed"] == 0

    status = client.get("/api/index/status").json()
    assert status["total_assets"] == 5
    assert status["completed"] == 3
    assert status["duplicate"] == 1
    assert status["unsupported"] == 1
    assert status["pending"] == 0
    assert status["failed"] == 0


# --- 9. idle status ------------------------------------------------------------------------


def test_idle_status(client):
    response = client.get("/api/index/status")

    assert response.status_code == 200
    body = response.json()
    assert body["running"] is False
    assert body["stage"] == "idle"
    assert body["dataset_path"] is None
    assert body["started_at"] is None
    assert body["completed_at"] is None
    for field in ("total_assets", "pending", "processing", "completed",
                  "failed", "skipped", "duplicate", "unsupported"):
        assert body[field] == 0


# --- 10. existing endpoints still work -------------------------------------------------------


def test_existing_endpoints_still_work(client, db, tmp_path, monkeypatch):
    root = use_dataset(monkeypatch, tmp_path)
    make_png(root / "images" / "sunset-beach.png")
    assert client.post("/api/index").status_code == 200

    assets = client.get("/api/assets")
    assert assets.status_code == 200 and assets.json()["total"] >= 1

    stats = client.get("/api/assets/stats")
    assert stats.status_code == 200 and stats.json()["total"] >= 1

    search = client.get("/api/search", params={"q": "sunset"})
    assert search.status_code == 200 and search.json()["count"] >= 1

    process = client.post("/api/process", json={"limit": 10})
    assert process.status_code == 200 and process.json()["status"] == "completed"

    videos = client.post("/api/process/videos", json={"limit": 10})
    assert videos.status_code == 200 and videos.json()["status"] == "completed"
