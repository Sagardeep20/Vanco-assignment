"""Regression tests for hybrid lexical + semantic ranking.

Covers the reported failure: unrelated PDFs with inflated raw
text-text cosine (0.55-0.85 band) outranking genuinely relevant
images (0.15-0.30 text-image band). Vectors are crafted to reproduce
those exact bands with precise cosines — deterministic and
weight-free. No production rule mentions any specific word.
"""

from __future__ import annotations

import math

import pytest

import app.services.search as search_svc
from app.models import Asset, AssetStatus, FileType
from app.services.embeddings import deterministic_embedding
from app.services.search import (
    hybrid_rank,
    lexical_score,
    query_terms,
    search_assets,
    vector_search,
)


def _sim_vec(q: list[float], sim: float, basis: int) -> list[float]:
    """Unit vector with exact cosine ``sim`` to unit vector ``q``."""
    rest = math.sqrt(max(0.0, 1.0 - sim * sim))
    return [sim * qi + (rest if i == basis else 0.0) for i, qi in enumerate(q)]


@pytest.fixture()
def query_q(monkeypatch):
    """Fixed unit query vector for every search in this module."""
    q = deterministic_embedding("query seed words")
    assert q is not None
    basis = next(i for i, v in enumerate(q) if v == 0.0)

    def fake_embed(text, provider_name=None):
        return list(q)

    monkeypatch.setattr(search_svc, "embed_text", fake_embed)
    return q, basis


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


def _pdf_text():
    return {
        "description": "Document placeholder.",
        "extracted_text": "quarterly budget numbers",
    }


# --- A. exact filename beats higher raw-scoring PDFs ---------------------------


@pytest.fixture()
def dog_fixture(db, query_q):
    q, basis = query_q
    dog = _add(db, "images/dog.jpg", FileType.IMAGE, _sim_vec(q, 0.30, basis),
               description="Image placeholder.")
    other = _add(db, "images/other.jpg", FileType.IMAGE, _sim_vec(q, 0.20, basis),
                 description="Image placeholder.")
    pdfs = [
        _add(db, f"documents/p{i}.pdf", FileType.PDF, _sim_vec(q, s, basis),
             **_pdf_text())
        for i, s in enumerate((0.99, 0.95, 0.90))
    ]
    return {"dog": dog, "other": other, "pdfs": pdfs}


def test_exact_filename_outranks_higher_raw_pdfs(db, dog_fixture):
    results, method = search_assets(db, "dog")
    assert method == "vector"
    paths = [a.relative_path for a, _ in results]
    assert paths[0] == "images/dog.jpg"
    assert all(math.isfinite(s) for _, s in results)
    # Every unrelated PDF ranks below despite raw 0.90-0.99 > 0.30.
    assert paths.index("images/dog.jpg") < min(
        paths.index(f"documents/p{i}.pdf") for i in range(3)
    )


def test_exact_match_survives_small_limit(db, dog_fixture):
    results, _ = search_assets(db, "dog", limit=2)
    assert len(results) == 2
    assert results[0][0].relative_path == "images/dog.jpg"


# --- B/C. semantic retrieval without filename match ----------------------------


@pytest.fixture()
def forest_fixture(db, query_q):
    q, basis = query_q
    valley = _add(
        db, "images/still-water.jpg", FileType.IMAGE, _sim_vec(q, 0.25, basis),
        description="Image placeholder.",
    )
    mid = [
        _add(db, f"images/photo{i}.jpg", FileType.IMAGE, _sim_vec(q, s, basis),
             description="Image placeholder.")
        for i, s in enumerate((0.18, 0.17))
    ]
    pdfs = [
        _add(db, f"documents/r{i}.pdf", FileType.PDF, _sim_vec(q, s, basis),
             **_pdf_text())
        for i, s in enumerate((0.70, 0.65, 0.60))
    ]
    vids = [
        _add(db, f"videos/clip{i}.mp4", FileType.VIDEO, _sim_vec(q, s, basis),
             description="Video placeholder.")
        for i, s in enumerate((0.26, 0.15))
    ]
    return {"valley": valley, "mid": mid, "pdfs": pdfs, "vids": vids}


def test_semantic_match_wins_without_filename_terms(db, forest_fixture):
    results, method = search_assets(db, "forest")
    assert method == "vector"
    assert results[0][0].relative_path == "images/still-water.jpg"


def test_descriptive_query_retrieves_semantic_match(db, forest_fixture):
    # None of "lush/green/forest/landscape" appears in any stored
    # filename or text here; retrieval is by content similarity.
    results, _ = search_assets(db, "a lush green forest landscape")
    paths = [a.relative_path for a, _ in results]
    assert paths[0] == "images/still-water.jpg"


def test_stem_term_match_boosts_but_semantic_decides(db, query_q):
    q, basis = query_q
    scenic = _add(
        db, "images/green-landscape.jpg", FileType.IMAGE,
        _sim_vec(q, 0.24, basis), description="Image placeholder.",
    )
    plain = _add(
        db, "images/plain.jpg", FileType.IMAGE,
        _sim_vec(q, 0.26, basis), description="Image placeholder.",
    )
    pdf = _add(db, "documents/r.pdf", FileType.PDF, _sim_vec(q, 0.70, basis),
               **_pdf_text())
    results, _ = search_assets(db, "a lush green forest landscape")
    paths = [a.relative_path for a, _ in results]
    # Stem-term "landscape" hit wins despite lower raw than plain.jpg.
    assert paths[0] == "images/green-landscape.jpg"
    assert scenic is not None and plain is not None and pdf is not None


# --- D. invalid embeddings never outrank ----------------------------------------


def test_null_zero_invalid_never_outrank(db, query_q):
    q, basis = query_q
    good = _add(db, "images/good.jpg", FileType.IMAGE, _sim_vec(q, 0.40, basis),
                description="Image placeholder.")
    _add(db, "documents/blank.pdf", FileType.PDF, None)
    _add(db, "images/zero.png", FileType.IMAGE, [0.0] * 512)
    results, method = search_assets(db, "good")
    assert method == "vector"
    assert results[0][0].id == good.id
    assert all(math.isfinite(s) for _, s in results)
    paths = [a.relative_path for a, _ in results]
    assert "documents/blank.pdf" not in paths
    assert "images/zero.png" not in paths


# --- E/F/G. every media type stays searchable ------------------------------------


def test_all_media_types_searchable(db, query_q):
    q, basis = query_q
    _add(db, "images/pic.jpg", FileType.IMAGE, _sim_vec(q, 0.30, basis),
         description="Image placeholder.")
    _add(db, "documents/note.pdf", FileType.PDF, _sim_vec(q, 0.60, basis),
         **_pdf_text())
    _add(db, "videos/movie.mp4", FileType.VIDEO, _sim_vec(q, 0.25, basis),
         description="Video placeholder.")
    results, method = search_assets(db, "holiday")
    assert method == "vector"
    assert {a.file_type for a, _ in results} == {
        FileType.IMAGE, FileType.PDF, FileType.VIDEO,
    }


# --- H/I. filters and limits ------------------------------------------------------


def test_file_type_filters(db, dog_fixture):
    results, _ = search_assets(db, "dog", file_type=FileType.PDF)
    assert results and all(a.file_type == FileType.PDF for a, _ in results)
    results, _ = search_assets(db, "dog", file_type=FileType.IMAGE)
    assert results and all(a.file_type == FileType.IMAGE for a, _ in results)
    assert results[0][0].relative_path == "images/dog.jpg"


def test_limit_respected_and_best_first(db, dog_fixture):
    results, _ = search_assets(db, "dog", limit=3)
    assert len(results) == 3
    assert results[0][0].relative_path == "images/dog.jpg"
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# --- lexical unit behavior (generic, term-based) -----------------------------------


def _asset(**kwargs):
    kwargs.setdefault("filename", "dog.jpg")
    kwargs.setdefault("relative_path", "images/dog.jpg")
    kwargs.setdefault("description", "Image placeholder.")
    kwargs.setdefault("extracted_text", None)
    return Asset(file_type=FileType.IMAGE, status=AssetStatus.COMPLETED, **kwargs)


def test_query_terms_drop_filler_words():
    assert query_terms("dog") == ["dog"]
    assert query_terms("a lush green forest landscape") == [
        "lush", "green", "forest", "landscape",
    ]
    assert query_terms("  ") == []


def test_lexical_tiers_are_generic():
    assert lexical_score("dog", _asset()) == 1.0  # stem-exact
    assert lexical_score("DOG", _asset()) == 1.0  # case-insensitive
    partial = _asset(filename="mydog-photo.jpg",
                     relative_path="images/mydog-photo.jpg")
    assert 0.0 < lexical_score("dog", partial) < 1.0  # substring tier
    desc = _asset(filename="x.jpg", relative_path="images/x.jpg",
                  description="A dog runs across the yard")
    assert lexical_score("dog", desc) == pytest.approx(0.5)
    text = _asset(filename="x.jpg", relative_path="images/x.jpg",
                  description="Image placeholder.",
                  extracted_text="the dog budget report")
    assert lexical_score("dog", text) == pytest.approx(0.35)
    assert lexical_score("forest", _asset()) == 0.0  # no match, no keywords
    assert lexical_score("", _asset()) == 0.0


def test_hybrid_rank_prefers_lexical_then_semantic():
    scenic = _asset(filename="oak.jpg", relative_path="images/oak.jpg")
    other = _asset(filename="b.jpg", relative_path="images/b.jpg")
    # Same raw semantic scale: the filename hit must win.
    ranked = hybrid_rank("oak", [(scenic, 0.2), (other, 0.9)])
    assert [a.filename for a, _ in ranked] == ["oak.jpg", "b.jpg"]
    # No lexical anywhere: higher raw semantic wins.
    ranked = hybrid_rank("zzz", [(scenic, 0.2), (other, 0.9)])
    assert [a.filename for a, _ in ranked] == ["b.jpg", "oak.jpg"]
