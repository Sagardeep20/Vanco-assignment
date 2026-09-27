"""Focused tests for genuine CLIP cross-modal image embeddings.

- Structural tests (512-d, normalized, deterministic) run against the
  real cached OpenCLIP weights when available, else skip.
- Provider/routing tests (pixels-not-description, fallbacks,
  frame aggregation) are weight-free via stubs.
"""

from __future__ import annotations

import math

import pytest

import app.services.ai as ai_svc
import app.services.video as video_svc
from app.core.config import settings
from app.models import Asset, AssetStatus, FileType
from app.models.asset import EMBEDDING_DIMENSION
from app.services.ai import process_asset, reembed_completed_assets
from app.services.embeddings import (
    DeterministicProvider,
    EmbeddingUnavailableError,
    clear_provider_cache,
    embed_image,
)


@pytest.fixture(autouse=True)
def _clean_provider_cache():
    clear_provider_cache()
    yield
    clear_provider_cache()


def _norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def _needs_clip():
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    pytest.importorskip("PIL.Image")
    from app.services.embeddings import LocalSemanticProvider

    provider = LocalSemanticProvider()
    try:
        provider._load()
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"cached ViT-B-32/openai weights unavailable offline: {exc}")
    return provider


def _image(color="green", size=(64, 48)):
    from PIL import Image

    return Image.new("RGB", size, color=color)


# --- real CLIP image tower ----------------------------------------------------


def test_clip_image_embedding_512_and_normalized():
    _needs_clip()
    vec = embed_image(_image())
    assert vec is not None
    assert len(vec) == EMBEDDING_DIMENSION == 512
    assert abs(_norm(vec) - 1.0) < 1e-5


def test_clip_image_embedding_deterministic():
    _needs_clip()
    assert embed_image(_image("green")) == pytest.approx(embed_image(_image("green")))


def test_clip_image_embeddings_distinguish_pixel_content():
    """Different pixels -> different vectors (i.e. content, not boilerplate)."""
    _needs_clip()
    green = embed_image(_image("green"))
    red = embed_image(_image("red"))

    def cosine(u, v):
        return sum(a * b for a, b in zip(u, v))

    assert cosine(green, red) < 0.999


def test_clip_cross_modal_text_query_matches_image():
    """A visually grounded caption scores above an unrelated one.

    Genuine joint-space check: no keywords are injected anywhere.
    """
    _needs_clip()
    from app.services.embeddings import embed_text

    img_vec = embed_image(_image("green", size=(128, 128)))

    def cosine(u, v):
        return sum(a * b for a, b in zip(u, v))

    related = embed_text("a photo of a lush green forest")
    unrelated = embed_text("a red fire engine")
    assert cosine(img_vec, related) > cosine(img_vec, unrelated)


# --- provider interface / fallback (weight-free) ------------------------------


def test_embed_image_none_returns_none():
    assert embed_image(None) is None


def test_deterministic_provider_has_no_image_encoder():
    assert DeterministicProvider().embed_image(None) is None
    with pytest.raises(EmbeddingUnavailableError):
        DeterministicProvider().embed_image(_image())


# --- processing routes pixels, not descriptions -------------------------------


def _add_image_asset(db, path, status=AssetStatus.PENDING):
    asset = Asset(
        filename=path.name,
        original_path=str(path.resolve()),
        relative_path=path.name,
        file_type=FileType.IMAGE,
        file_size=path.stat().st_size,
        status=status,
        width=64,
        height=48,
    )
    db.add(asset)
    db.commit()
    return asset


def test_process_asset_embeds_actual_pixels(db, tmp_path, monkeypatch):
    """process_asset must pass the decoded image (not the description)."""
    from PIL import Image

    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))
    img_path = tmp_path / "meadow.png"
    Image.new("RGB", (64, 48), color="green").save(img_path)
    asset = _add_image_asset(db, img_path)

    seen = {}
    marker = [0.0] * 511 + [1.0]

    def fake_embed_image(image, provider_name=None):
        seen["size"] = image.size
        seen["mode"] = image.mode
        return list(marker)

    monkeypatch.setattr(ai_svc, "embed_image", fake_embed_image)
    result = process_asset(db, asset)

    assert result.status is AssetStatus.COMPLETED
    assert seen["size"] == (64, 48)  # real decoded pixels arrived
    assert seen["mode"] == "RGB"
    assert list(result.embedding) == pytest.approx(marker)


def test_process_asset_image_falls_back_to_text(db, tmp_path, monkeypatch):
    """Missing/unreadable pixels (or no encoder) still complete via text."""
    from app.services.embeddings import deterministic_embedding

    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))
    # File does not exist inside the dataset -> confinement check fails.
    asset = Asset(
        filename="ghost.png",
        original_path="/elsewhere/ghost.png",
        relative_path="ghost.png",
        file_type=FileType.IMAGE,
        status=AssetStatus.PENDING,
    )
    db.add(asset)
    db.commit()

    def no_encoder(image, provider_name=None):
        raise EmbeddingUnavailableError("no image encoder here")

    monkeypatch.setattr(ai_svc, "embed_image", no_encoder)
    monkeypatch.setattr(ai_svc, "embed_text", deterministic_embedding)
    result = process_asset(db, asset)

    assert result.status is AssetStatus.COMPLETED
    assert list(result.embedding) == pytest.approx(
        deterministic_embedding(
            ai_svc.build_searchable_text(
                result.filename, result.description, result.extracted_text
            )
        )
    )


def test_embed_frame_files_mean_pool_and_normalize(tmp_path, monkeypatch):
    """Frame vectors are mean-pooled then L2-normalized into one 512-d vector."""
    from PIL import Image

    frames = []
    for i, color in enumerate(("red", "green")):
        path = tmp_path / f"frame_{i:03d}.jpg"
        Image.new("RGB", (16, 16), color=color).save(path)
        frames.append(path)

    vecs = [[3.0, 0.0] + [0.0] * 510, [0.0, 4.0] + [0.0] * 510]
    calls = {"n": 0}

    def fake_embed_image(image, provider_name=None):
        vec = vecs[calls["n"]]
        calls["n"] += 1
        return list(vec)

    monkeypatch.setattr(video_svc, "embed_image", fake_embed_image)
    out = video_svc.embed_frame_files(frames)

    assert out is not None and len(out) == 512
    # mean([3,0],[0,4]) = [1.5,2] -> norm 2.5 -> [0.6,0.8,...]
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(0.8)
    assert abs(_norm(out) - 1.0) < 1e-9


def test_embed_frame_files_returns_none_without_encoder(tmp_path, monkeypatch):
    from PIL import Image

    path = tmp_path / "frame_000.jpg"
    Image.new("RGB", (16, 16), color="blue").save(path)

    def no_encoder(image, provider_name=None):
        raise EmbeddingUnavailableError("deterministic")

    monkeypatch.setattr(video_svc, "embed_image", no_encoder)
    assert video_svc.embed_frame_files([path]) is None


def test_reembed_image_uses_pixels_pdf_uses_text(db, tmp_path, monkeypatch):
    """Re-embed: images from pixels, PDFs from searchable text."""
    from PIL import Image

    monkeypatch.setattr(settings, "dataset_path", str(tmp_path))
    img_path = tmp_path / "ridge.png"
    Image.new("RGB", (32, 24), color="green").save(img_path)
    img = Asset(
        filename="ridge.png",
        original_path=str(img_path.resolve()),
        relative_path="ridge.png",
        file_type=FileType.IMAGE,
        status=AssetStatus.COMPLETED,
        description="Image 'ridge' (32x24px). Local placeholder.",
        embedding=[0.0] * 511 + [1.0],
    )
    pdf = Asset(
        filename="notes.pdf",
        original_path="/elsewhere/notes.pdf",
        relative_path="notes.pdf",
        file_type=FileType.PDF,
        status=AssetStatus.COMPLETED,
        description="Document 'notes'.",
        extracted_text="alpine meadow report",
        embedding=[0.0] * 511 + [1.0],
    )
    db.add_all([img, pdf])
    db.commit()

    pixel_marker = [1.0] + [0.0] * 511
    text_calls = []

    def fake_embed_image(image, provider_name=None):
        assert image.size == (32, 24)
        return list(pixel_marker)

    def fake_embed_text(text, provider_name=None):
        text_calls.append(text)
        return [0.5] * 512

    monkeypatch.setattr(ai_svc, "embed_image", fake_embed_image)
    monkeypatch.setattr(ai_svc, "embed_text", fake_embed_text)

    outcome = reembed_completed_assets(db)

    assert outcome == {"processed": 2, "completed": 2, "failed": 0, "errors": []}
    assert list(db.get(Asset, img.id).embedding) == pytest.approx(pixel_marker)
    assert db.get(Asset, img.id).description.startswith("Image 'ridge'")
    assert db.get(Asset, img.id).status is AssetStatus.COMPLETED
    # PDF went through the text path with its searchable text.
    assert text_calls == [
        ai_svc.build_searchable_text(pdf.filename, pdf.description, pdf.extracted_text)
    ]
