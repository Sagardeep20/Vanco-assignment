"""Regression tests for search ranking with missing/degenerate embeddings.

Guards the reported bug: rows with no usable vector similarity (NULL
embeddings, zero-norm vectors yielding NaN cosine distance) must never
sort ahead of genuinely scored results. Weight-free: deterministic
vectors stand in for CLIP embeddings; only ranking is under test.
"""

from __future__ import annotations

import math

import pytest

import app.services.search as search_svc
from app.models import Asset, AssetStatus, FileType
from app.services.embeddings import deterministic_embedding
from app.services.search import order_results, search_assets, vector_search


@pytest.fixture()
def stub_query(monkeypatch):
    """Fix the query embedding so similarities are fully controlled."""

    def fake_embed(text, provider_name=None):
        return deterministic_embedding("dog")

    monkeypatch.setattr(search_svc, "embed_text", fake_embed)
    return deterministic_embedding("dog")


def _add(db, relative_path, file_type, embedding, **kwargs):
    kwargs.setdefault("filename", relative_path.rsplit("/", 1)[-1])
    kwargs.setdefault("original_path", f"/data/{relative_path}")
    asset = Asset(
        relative_path=relative_path,
        file_type=file_type,
        status=AssetStatus.COMPLETED,
        embedding=embedding,
        **kwargs,
    )
    db.add(asset)
    db.commit()
    return asset


@pytest.fixture()
def ranked_rows(db):
    """A top hit, a weak hit, a NULL-embedding row, and a zero vector."""
    good = _add(
        db, "images/dog.jpg", FileType.IMAGE,
        list(deterministic_embedding("dog")),
    )
    weak = _add(
        db, "images/car.jpg", FileType.IMAGE,
        list(deterministic_embedding("quantum database replication index")),
    )
    missing = _add(db, "docs/sample.pdf", FileType.PDF, None)
    zero = _add(db, "images/blank.png", FileType.IMAGE, [0.0] * 512)
    return {"good": good, "weak": weak, "missing": missing, "zero": zero}


def test_scored_result_ranks_before_missing_and_zero_embeddings(
    db, ranked_rows, stub_query
):
    results, _ = vector_search(db, "dog")
    assert results, "expected vector hits"
    paths = [asset.relative_path for asset, _ in results]
    # The high-scoring hit is first; every returned score is finite.
    assert paths[0] == "images/dog.jpg"
    assert all(score == score and score not in (float("inf"), float("-inf"))
               for _, score in results)
    # NULL/degenerate rows never appear ahead of (or among) scored rows.
    assert "docs/sample.pdf" not in paths
    assert "images/blank.png" not in paths
    # Genuine weak hit still ranks after the top hit.
    assert paths.index("images/dog.jpg") < paths.index("images/car.jpg")


def test_search_assets_vector_path_orders_valid_first(
    db, ranked_rows, stub_query
):
    results, method = search_assets(db, "dog")
    assert method == "vector"
    assert results[0][0].relative_path == "images/dog.jpg"
    # Fused relevance: finite, and strictly above the runner-up.
    assert math.isfinite(results[0][1])
    assert results[0][1] > results[1][1]


def test_order_results_pushes_unscored_last():
    scored = [("b", 0.2), ("a", 0.9)]
    unscored = [("c", None), ("d", float("nan"))]
    ordered = order_results(
        [(a, s) for a, s in scored] + [(a, s) for a, s in unscored]  # type: ignore[arg-type]
    )
    assert [name for name, _ in ordered] == ["a", "b", "c", "d"]


def test_file_type_filter_and_limit_preserved(db, ranked_rows, stub_query):
    results, _ = vector_search(db, "dog", file_type=FileType.PDF)
    assert results == []
    results, _ = vector_search(db, "dog", limit=1)
    assert [a.relative_path for a, _ in results] == ["images/dog.jpg"]
    results, method = search_assets(db, "dog", file_type=FileType.IMAGE)
    assert method == "vector"
    assert {a.relative_path for a, _ in results} == {"images/dog.jpg", "images/car.jpg"}


def test_keyword_fallback_preserved_without_embeddings(db, stub_query):
    _add(db, "images/dog.jpg", FileType.IMAGE, None)
    results, method = search_assets(db, "dog")
    assert method == "keyword (no embeddings yet)"
    assert [a.relative_path for a, _ in results] == ["images/dog.jpg"]
    assert results[0][1] is None
