"""Tests for the embedding provider abstraction (mocked, no downloads).

The local semantic model is always mocked: these tests never require
torch, sentence-transformers, transformers, cached weights, or network.
"""

import math

import pytest

import app.services.ai as ai_svc
import app.services.search as search_svc
from app.core.config import settings
from app.models import Asset, AssetStatus
from app.models.asset import EMBEDDING_DIMENSION
from app.services.embeddings import (
    DeterministicProvider,
    EmbeddingUnavailableError,
    LocalSemanticProvider,
    clear_provider_cache,
    deterministic_embedding,
    embed_text,
    get_provider,
    l2_normalize,
)
from app.services.scanner import scan_dataset


def norm(vec):
    return math.sqrt(sum(v * v for v in vec))


@pytest.fixture(autouse=True)
def _clean_provider_cache():
    clear_provider_cache()
    yield
    clear_provider_cache()


def make_png(path, size=(64, 48), color="red"):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


class FakeModel:
    """Stand-in for a loaded CLIP encoder (no weights, no torch)."""

    def __init__(self):
        self.encode_calls = []


# --- provider interface -------------------------------------------------------


def test_provider_interface():
    for cls, expected_name in (
        (DeterministicProvider, "deterministic"),
        (LocalSemanticProvider, "local"),
    ):
        provider = cls()
        assert provider.name == expected_name
        assert provider.dim == EMBEDDING_DIMENSION == 512
        assert callable(provider.embed)


def test_deterministic_provider_exactly_512_and_normalized():
    vec = DeterministicProvider().embed("sunset beach photo")
    assert vec is not None and len(vec) == 512
    assert abs(norm(vec) - 1.0) < 1e-6


def test_deterministic_provider_blank_returns_none():
    provider = DeterministicProvider()
    assert provider.embed("") is None
    assert provider.embed("   ") is None


def test_l2_normalize_unit_norm():
    vec = l2_normalize([3.0, 4.0] + [0.0] * 510)
    assert len(vec) == 512
    assert vec[0] == pytest.approx(0.6)
    assert vec[1] == pytest.approx(0.8)
    assert abs(norm(vec) - 1.0) < 1e-9


def test_unknown_provider_name_raises():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        get_provider("clip-cloud")


# --- selection / fallback ------------------------------------------------------


def test_auto_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setattr(
        LocalSemanticProvider, "_load",
        lambda self: (_ for _ in ()).throw(
            EmbeddingUnavailableError("mocked: no local weights")),
    )
    monkeypatch.setattr(settings, "embedding_provider", "auto")

    provider = get_provider()
    assert isinstance(provider, DeterministicProvider)
    assert embed_text("hello world") == deterministic_embedding("hello world")


def test_deterministic_always_selected(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "deterministic")
    assert isinstance(get_provider(), DeterministicProvider)


def test_local_unavailable_returns_clear_error(monkeypatch):
    monkeypatch.setattr(
        LocalSemanticProvider, "_load",
        lambda self: (_ for _ in ()).throw(
            EmbeddingUnavailableError("mocked: no local weights")),
    )
    with pytest.raises(EmbeddingUnavailableError, match="local"):
        get_provider("local")


def test_local_provider_selection_and_normalization(monkeypatch):
    raw = [3.0, 4.0] + [0.0] * 510
    monkeypatch.setattr(LocalSemanticProvider, "_load", lambda self: FakeModel())
    monkeypatch.setattr(
        LocalSemanticProvider, "_encode", lambda self, model, text: list(raw))

    provider = get_provider("local")
    assert isinstance(provider, LocalSemanticProvider)
    vec = provider.embed("a photo of a beach")
    assert vec is not None and len(vec) == 512
    assert vec[0] == pytest.approx(0.6)
    assert vec[1] == pytest.approx(0.8)
    assert abs(norm(vec) - 1.0) < 1e-9


def test_auto_prefers_local_when_available(monkeypatch):
    raw = [1.0] * 512
    monkeypatch.setattr(LocalSemanticProvider, "_load", lambda self: FakeModel())
    monkeypatch.setattr(
        LocalSemanticProvider, "_encode", lambda self, model, text: list(raw))
    monkeypatch.setattr(settings, "embedding_provider", "auto")

    assert isinstance(get_provider(), LocalSemanticProvider)


# --- persistence + search use the same provider ---------------------------------


def test_asset_processing_uses_provider(db, tmp_path, monkeypatch):
    """process_asset stores whatever the configured provider returns."""
    make_png(tmp_path / "images" / "a.png")
    scan_dataset(db, tmp_path)

    seen = {}
    stub_vec = [0.5] * 512  # already unit-norm-ish; stored verbatim

    def fake_embed(text, provider_name=None):
        seen["text"] = text
        seen["provider_name"] = provider_name
        return list(stub_vec)

    monkeypatch.setattr(ai_svc, "embed_text", fake_embed)
    asset = db.query(Asset).filter(Asset.relative_path == "images/a.png").one()

    from app.services.ai import process_asset

    process_asset(db, asset)

    assert asset.status is AssetStatus.COMPLETED
    assert asset.description and "a" in asset.description.lower()
    assert "a" in seen["text"].lower()  # searchable text carries the description
    assert list(asset.embedding) == stub_vec


def test_asset_embedding_persisted_512_normalized(db, tmp_path):
    """Default path stores a real 512-d normalized vector."""
    make_png(tmp_path / "images" / "beach.png")
    scan_dataset(db, tmp_path)
    asset = db.query(Asset).filter(Asset.relative_path == "images/beach.png").one()

    from app.services.ai import process_asset

    process_asset(db, asset)

    assert asset.embedding is not None and len(asset.embedding) == 512
    assert abs(norm(list(asset.embedding)) - 1.0) < 1e-6


def test_search_query_uses_same_provider(db, tmp_path, monkeypatch):
    """The search query is embedded through the shared provider entrypoint."""
    make_png(tmp_path / "images" / "sunset-beach.png")
    scan_dataset(db, tmp_path)
    from app.services.ai import process_pending_assets

    process_pending_assets(db, limit=10)

    recorded = {}

    def recording_embed(text, provider_name=None):
        recorded["query"] = text
        return deterministic_embedding(text)

    monkeypatch.setattr(search_svc, "embed_text", recording_embed)

    from app.services.search import search_assets

    results, method = search_assets(db, "sunset beach")
    assert recorded["query"] == "sunset beach"
    assert method == "vector"
    assert results and results[0][0].relative_path == "images/sunset-beach.png"


def test_existing_search_behavior_preserved(db, tmp_path):
    """End-to-end vector search still ranks shared-vocabulary hits first."""
    make_png(tmp_path / "images" / "sunset-beach.png")
    make_png(tmp_path / "images" / "zzz-report.png", size=(16, 16))
    scan_dataset(db, tmp_path)
    from app.services.ai import process_pending_assets

    process_pending_assets(db, limit=10)

    from app.services.search import search_assets

    results, method = search_assets(db, "sunset beach")
    assert method == "vector"
    assert [r.relative_path for r, _ in results][0] == "images/sunset-beach.png"
