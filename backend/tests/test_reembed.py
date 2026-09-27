"""Focused tests for one-time re-embedding of completed assets.

Uses a stubbed embedding entrypoint (no weights needed): what matters
here is row selection, preservation guarantees, and that the shared
``embed_text`` path is used — not the CLIP weights themselves.
"""

from __future__ import annotations

import pytest

import app.services.ai as ai_svc
from app.models import Asset, AssetStatus, FileType
from app.services.ai import build_searchable_text, reembed_completed_assets
from app.services.embeddings import deterministic_embedding

STUB_VEC = [1.0] + [0.0] * 511


@pytest.fixture()
def stub_embed(monkeypatch):
    calls = []

    def fake_embed(text, provider_name=None):
        calls.append(text)
        return list(STUB_VEC)

    monkeypatch.setattr(ai_svc, "embed_text", fake_embed)
    return calls


def _add(db, relative_path, file_type, status, **kwargs):
    kwargs.setdefault("filename", relative_path.rsplit("/", 1)[-1])
    kwargs.setdefault("original_path", f"/data/{relative_path}")
    asset = Asset(
        relative_path=relative_path,
        file_type=file_type,
        status=status,
        **kwargs,
    )
    db.add(asset)
    db.commit()
    return asset


@pytest.fixture()
def seeded(db):
    """Completed image/pdf/video with stale deterministic embeddings,
    plus rows that must be preserved untouched."""
    old_img = deterministic_embedding("old image text")
    old_pdf = deterministic_embedding("old pdf text")
    old_vid = deterministic_embedding("old video text")
    img = _add(
        db, "images/beach.png", FileType.IMAGE, AssetStatus.COMPLETED,
        description="Image 'beach' (64x48px, 1.0 KB). Local placeholder.",
        embedding=list(old_img),
    )
    pdf = _add(
        db, "docs/report.pdf", FileType.PDF, AssetStatus.COMPLETED,
        description="Document 'report' (3 pages, 2.0 KB). Content starts: hello...",
        extracted_text="hello asset world",
        embedding=list(old_pdf),
    )
    vid = _add(
        db, "videos/clip.mp4", FileType.VIDEO, AssetStatus.COMPLETED,
        description="Video 'clip' (10.0s, 2 frame(s) sampled). Frame observations: [1] ...",
        embedding=list(old_vid),
    )
    failed = _add(
        db, "images/broken.png", FileType.IMAGE, AssetStatus.FAILED,
        error_message="Could not read image: broken",
    )
    pending = _add(db, "images/new.png", FileType.IMAGE, AssetStatus.PENDING)
    duplicate = _add(
        db, "images/beach-copy.png", FileType.IMAGE, AssetStatus.DUPLICATE,
        embedding=list(old_img),
    )
    unsupported = _add(
        db, "notes/readme.txt", FileType.OTHER, AssetStatus.UNSUPPORTED,
    )
    return {
        "img": img, "pdf": pdf, "vid": vid, "failed": failed,
        "pending": pending, "duplicate": duplicate, "unsupported": unsupported,
        "old": {"img": old_img, "pdf": old_pdf, "vid": old_vid},
    }


def test_reembed_updates_only_completed_media(db, seeded, stub_embed):
    before = {
        key: (
            seeded[key].description, seeded[key].extracted_text,
            seeded[key].status, seeded[key].file_hash,
        )
        for key in ("img", "pdf", "vid", "failed", "pending", "duplicate", "unsupported")
    }
    total_before = db.query(Asset).count()

    outcome = reembed_completed_assets(db)

    assert outcome["processed"] == 3
    assert outcome["completed"] == 3
    assert outcome["failed"] == 0
    assert outcome["errors"] == []
    assert db.query(Asset).count() == total_before  # nothing added/deleted

    for key in ("img", "pdf", "vid"):
        asset = db.get(Asset, seeded[key].id)
        assert list(asset.embedding) == pytest.approx(STUB_VEC)
        # Metadata/status untouched; only the embedding column changed.
        assert (
            asset.description, asset.extracted_text,
            asset.status, asset.file_hash,
        ) == before[key]

    # Failed/pending/duplicate/unsupported rows preserved verbatim.
    for key in ("failed", "pending", "duplicate", "unsupported"):
        asset = db.get(Asset, seeded[key].id)
        assert (
            asset.description, asset.extracted_text,
            asset.status, asset.file_hash,
        ) == before[key]
        if key == "duplicate":
            assert list(asset.embedding) == pytest.approx(list(seeded["old"]["img"]))
        else:
            assert asset.embedding is None

    # Same searchable text as normal processing/search path.
    expected = {
        build_searchable_text(a.filename, a.description, a.extracted_text)
        for a in (seeded["img"], seeded["pdf"], seeded["vid"])
    }
    assert set(stub_embed) == expected


def test_reembed_respects_limit(db, seeded, stub_embed):
    outcome = reembed_completed_assets(db, limit=1)
    assert outcome["processed"] == 1
    assert outcome["completed"] == 1
    assert len(stub_embed) == 1


def test_reembed_failure_does_not_abort(db, seeded, monkeypatch):
    def flaky_embed(text, provider_name=None):
        if "report.pdf" in text:
            raise RuntimeError("provider blew up")
        return list(STUB_VEC)

    monkeypatch.setattr(ai_svc, "embed_text", flaky_embed)
    outcome = reembed_completed_assets(db)

    assert outcome["processed"] == 3
    assert outcome["completed"] == 2
    assert outcome["failed"] == 1
    assert len(outcome["errors"]) == 1 and "report.pdf" in outcome["errors"][0]
    # Failed row keeps its old embedding and completed status.
    pdf = db.get(Asset, seeded["pdf"].id)
    assert list(pdf.embedding) == pytest.approx(list(seeded["old"]["pdf"]))
    assert pdf.status == AssetStatus.COMPLETED
    assert list(db.get(Asset, seeded["img"].id).embedding) == pytest.approx(STUB_VEC)
    assert list(db.get(Asset, seeded["vid"].id).embedding) == pytest.approx(STUB_VEC)


def test_reembed_endpoint(db, seeded, stub_embed, client):
    response = client.post("/api/process/reembed", json={"limit": 1000})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["processed"] == 3
    assert body["completed"] == 3
    assert body["failed"] == 0
    assert list(db.get(Asset, seeded["vid"].id).embedding) == pytest.approx(STUB_VEC)
