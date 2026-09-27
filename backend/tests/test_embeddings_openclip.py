"""Focused tests for open_clip-based local provider selection.

Unit tests run without weights/torch. Integration tests use the already
installed open_clip package and already cached ViT-B-32/openai weights
(offline only, no downloads) and skip when genuinely unavailable.
"""

from __future__ import annotations

import math
import sys

import pytest

from app.core.config import settings
from app.services.embeddings import (
    DeterministicProvider,
    EmbeddingUnavailableError,
    LocalSemanticProvider,
    clear_provider_cache,
    get_provider,
)


@pytest.fixture(autouse=True)
def _clean_provider_cache():
    clear_provider_cache()
    yield
    clear_provider_cache()


def _norm(vec):
    return math.sqrt(sum(v * v for v in vec))


# --- model-ref resolution (no weights needed) ---------------------------------


@pytest.mark.parametrize(
    "ref", ["clip-ViT-B-32", "ViT-B-32", "ViT-B/32", "CLIP-ViT-B-32", ""]
)
def test_resolve_default_refs_use_cached_vit_b32_openai(ref):
    assert LocalSemanticProvider._resolve_open_clip_model(ref) == (
        "ViT-B-32",
        "openai",
    )


@pytest.mark.parametrize(
    "ref", ["ViT-L-14", "clip-ViT-L-14", "all-MiniLM-L6-v2", "bert-base-uncased"]
)
def test_resolve_unknown_refs_returns_none_never_downloads_other_models(ref):
    # Unknown names must NOT be handed to open_clip (which would download).
    assert LocalSemanticProvider._resolve_open_clip_model(ref) is None


def test_resolve_explicit_weight_file(tmp_path, monkeypatch):
    weight_file = tmp_path / "ViT-B-32.pt"
    weight_file.write_bytes(b"fake")
    monkeypatch.setattr(settings, "local_embedding_model_path", str(weight_file))
    assert LocalSemanticProvider._resolve_open_clip_model("clip-ViT-B-32") == (
        "ViT-B-32",
        str(weight_file),
    )


# --- fallback preserved when open_clip is genuinely unavailable ----------------


def test_try_open_clip_returns_none_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", None)
    assert LocalSemanticProvider()._try_open_clip("clip-ViT-B-32") is None


def test_auto_falls_back_when_no_backend_available(monkeypatch):
    monkeypatch.setattr(LocalSemanticProvider, "_try_open_clip", lambda self, ref: None)
    monkeypatch.setattr(
        LocalSemanticProvider, "_try_sentence_transformers", lambda self, ref: None
    )
    monkeypatch.setattr(
        LocalSemanticProvider, "_try_transformers_clip", lambda self, ref: None
    )
    monkeypatch.setattr(settings, "embedding_provider", "auto")
    assert isinstance(get_provider(), DeterministicProvider)
    with pytest.raises(EmbeddingUnavailableError):
        get_provider("local")


# --- integration with the real cached weights (skipped if unavailable) ---------


def _require_cached_open_clip():
    open_clip = pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    provider = LocalSemanticProvider()
    try:
        provider._load()
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"cached ViT-B-32/openai weights unavailable offline: {exc}")
    return provider


def test_auto_selects_local_when_cached_weights_present(monkeypatch):
    _require_cached_open_clip()
    monkeypatch.setattr(settings, "embedding_provider", "auto")
    assert isinstance(get_provider(), LocalSemanticProvider)


def test_local_embedding_is_512_and_normalized():
    provider = _require_cached_open_clip()
    vec = provider.embed("a photo of a beach")
    assert vec is not None and len(vec) == 512
    assert abs(_norm(vec) - 1.0) < 1e-5
    assert provider.embed("   ") is None
    # Deterministic on CPU: same text -> same vector.
    again = provider.embed("a photo of a beach")
    assert again == pytest.approx(vec)
