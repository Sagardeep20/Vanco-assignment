"""Asset search: hybrid lexical + semantic ranking, keyword fallback.

The query embedding comes from :func:`app.services.embeddings.embed_text`,
the SAME provider used when storing asset vectors. No model download is
ever triggered.

Why hybrid: raw CLIP cosine scores are NOT comparable across asset
types. Text embeddings live in a narrow cone, so unrelated PDF
(text-text) similarities routinely land in the 0.55-0.85 band while
correct image (text-image) matches score 0.15-0.30. Sorting raw
cosine therefore buries relevant images under unrelated PDFs.
Instead each candidate gets:

- a generic lexical score from query-term matches against filename,
  path, description, and extracted text (exact filename match wins),
- a semantic score: raw cosine normalized per asset type (z-score),
  which removes the per-modality band gap without assuming bands,

fused as ``LEXICAL_WEIGHT * lexical + semantic_z``. Exact matches
dominate; pure semantic decides when nothing matches lexically.
"""

from __future__ import annotations

import logging
import math
import re

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.asset import Asset, FileType
from app.services.embeddings import embed_text

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Generic English stopwords: matching "a"/"the"/"of" carries no topical
# signal (deterministic descriptions contain such words everywhere).
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "it", "its", "this",
    "that", "be", "are", "was", "were",
})

# Exact lexical matches must outrank any purely semantic gap: cosine
# bands span ~1 unit while a full lexical hit contributes this much.
LEXICAL_WEIGHT = 4.0


def _is_rankable(score: float | None) -> bool:
    """True only for a real similarity score (excludes None/NaN/inf)."""
    return isinstance(score, float) and math.isfinite(score)


# Extra candidates rescued by lexical match: the raw-distance SQL pool
# (``LIMIT``) can otherwise cut an exact filename hit that scores low
# on raw cosine before fusion ever sees it. Bounded and cheap.
LEXICAL_CANDIDATE_CAP = 200


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _lexical_filter(query: str, terms: list[str]):
    """SQL OR-filter matching the same fields as :func:`lexical_score`.

    Term-level (so multi-word queries still rescue candidates) plus a
    whole-query filename match. Terms are alphanumeric (LIKE-safe);
    the raw query string is escaped.
    """
    clauses = []
    if query.strip():
        pattern = f"%{_like_escape(query.strip().lower())}%"
        clauses.append(Asset.filename.ilike(pattern, escape="\\"))
    for term in terms:
        pattern = f"%{term}%"
        clauses.append(Asset.filename.ilike(pattern))
        clauses.append(Asset.description.ilike(pattern))
        clauses.append(Asset.extracted_text.ilike(pattern))
        clauses.append(Asset.relative_path.ilike(pattern))
    if not clauses:
        return None
    return or_(*clauses)


def query_terms(query: str) -> list[str]:
    """Topical search terms: lowercase alphanumeric tokens.

    Drops single characters and generic stopwords so filler words
    (e.g. the "a" in "a lush green forest") never count as relevance.
    """
    terms: list[str] = []
    for token in _TOKEN_RE.findall((query or "").lower()):
        if len(token) < 2 or token in _STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _field_tokens(text: str | None) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def lexical_score(query: str, asset: Asset) -> float:
    """Generic lexical relevance of an asset for a query, in [0, 1].

    Purely term-based over filename stem, filename, relative path,
    description, and extracted text. No per-word, per-file, or
    per-dataset rules: any query behaves the same way.
    """
    terms = query_terms(query)
    if not terms:
        return 0.0
    filename = (asset.filename or "").lower()
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem_norm = re.sub(r"[_\-]+", " ", stem).strip()
    query_norm = re.sub(r"\s+", " ", (query or "").lower()).strip()
    stem_tokens = set(_TOKEN_RE.findall(stem_norm))

    if query_norm and (query_norm == stem_norm or query_norm == filename):
        return 1.0
    if set(terms) <= stem_tokens:
        return 1.0
    if any(term in stem_tokens for term in terms):
        return 0.9
    if query_norm and query_norm in filename:
        return 0.65
    description_tokens = _field_tokens(asset.description)
    if any(term in description_tokens for term in terms):
        return 0.5
    path_tokens = _field_tokens(asset.relative_path)
    if any(term in path_tokens for term in terms):
        return 0.4
    if any(term in _field_tokens(asset.extracted_text) for term in terms):
        return 0.35
    return 0.0


def _zscore_by_type(
    raws: list[tuple[Asset, float]],
) -> dict[int, float]:
    """Per-asset-type z-score of raw cosine similarities.

    Centers each modality band (text-text PDFs vs. text-image media)
    at zero so a standout visual match outranks band-inflated text
    scores. Groups too small to standardize map to 0.0 (neutral).
    Returns a map from ``id(asset)`` to z-score.
    """
    groups: dict[object, list[int]] = {}
    for index, (asset, _) in enumerate(raws):
        groups.setdefault(asset.file_type, []).append(index)
    zscores: dict[int, float] = {}
    for indices in groups.values():
        values = [raws[i][1] for i in indices]
        n = len(values)
        if n < 2:
            for i in indices:
                zscores[id(raws[i][0])] = 0.0
            continue
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(var)
        if std <= 0.0:
            for i in indices:
                zscores[id(raws[i][0])] = 0.0
            continue
        for i in indices:
            zscores[id(raws[i][0])] = (raws[i][1] - mean) / std
    return zscores


def hybrid_rank(
    query: str, raws: list[tuple[Asset, float]]
) -> list[tuple[Asset, float]]:
    """Fuse lexical and semantic signals into one relevance ordering.

    ``LEXICAL_WEIGHT * lexical + semantic_z``: strong exact matches
    dominate, strong semantic matches decide among the rest, weak
    matches sink. Returns (asset, fused_relevance) sorted best first.
    """
    zscores = _zscore_by_type(raws)
    fused = [
        (asset, LEXICAL_WEIGHT * lexical_score(query, asset) + zscores[id(asset)])
        for asset, _ in raws
    ]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def order_results(
    results: list[tuple[Asset, float | None]],
) -> list[tuple[Asset, float | None]]:
    """Authoritative ranking: valid scores first (descending), unscored last.

    The database already orders by distance, but degenerate stored
    vectors (e.g. zero-norm embeddings, which yield non-finite cosine
    distances) and score-less keyword hits must never sort ahead of
    genuinely scored rows. This stable Python-side pass guarantees
    that invariant no matter what the database returns.
    """
    return sorted(
        results,
        key=lambda item: (0, -item[1]) if _is_rankable(item[1]) else (1, 0.0),
    )


def keyword_search(
    db: Session,
    query: str,
    limit: int = 20,
    file_type: FileType | None = None,
) -> list[tuple[Asset, None]]:
    """Case-insensitive LIKE search over filename/description/text/path."""
    pattern = f"%{query.strip()}%"
    q = db.query(Asset).filter(
        or_(
            Asset.filename.ilike(pattern),
            Asset.description.ilike(pattern),
            Asset.extracted_text.ilike(pattern),
            Asset.relative_path.ilike(pattern),
        )
    )
    if file_type is not None:
        q = q.filter(Asset.file_type == file_type)
    return [(row, None) for row in q.order_by(Asset.filename.asc()).limit(limit).all()]


def vector_search(
    db: Session,
    query: str,
    limit: int = 20,
    file_type: FileType | None = None,
) -> tuple[list[tuple[Asset, float]], list[float] | None]:
    """Hybrid lexical + semantic search. Returns ((asset, relevance), query_vec).

    Empty list when no rows have embeddings yet (caller falls back).
    Rows without a finite similarity (e.g. degenerate stored vectors
    whose cosine distance is NaN) carry no ranking signal and are
    excluded; the remainder is fused (never NaN) and ordered best first.
    """
    query_vec = embed_text(query)
    if query_vec is None:
        return [], None
    dist_expr = Asset.embedding.cosine_distance(query_vec).label("dist")
    q = db.query(Asset, dist_expr).filter(Asset.embedding.is_not(None))
    if file_type is not None:
        q = q.filter(Asset.file_type == file_type)
    rows = q.order_by("dist").limit(limit).all()
    # Rescue exact/lexical matches the raw-distance cutoff may have
    # dropped (raw text-image cosine systematically trails text-text).
    # Embedding-gated like the main pool, so the no-embeddings
    # keyword-fallback contract is unchanged.
    terms = query_terms(query)
    lex_filter = _lexical_filter(query, terms)
    if lex_filter is not None:
        seen = {asset.id for asset, _ in rows}
        extra_q = db.query(Asset, dist_expr).filter(
            Asset.embedding.is_not(None), lex_filter
        )
        if file_type is not None:
            extra_q = extra_q.filter(Asset.file_type == file_type)
        for row in extra_q.limit(LEXICAL_CANDIDATE_CAP).all():
            if row[0].id not in seen:
                seen.add(row[0].id)
                rows.append(row)
    raws: list[tuple[Asset, float]] = []
    for asset, dist in rows:
        try:
            score = float(1.0 - dist)
        except (TypeError, ValueError, ArithmeticError):
            continue
        if not _is_rankable(score):
            logger.info(
                "Skipping %s: stored embedding has no finite similarity.",
                asset.relative_path,
            )
            continue
        raws.append((asset, score))
    fused = hybrid_rank(query.strip(), raws)
    return fused[:limit], query_vec


def search_assets(
    db: Session,
    query: str,
    limit: int = 20,
    file_type: FileType | None = None,
) -> tuple[list[tuple[Asset, float | None]], str]:
    """Semantic search with keyword fallback.

    Returns (results, method) where method is 'vector', 'keyword', or
    'keyword (no embeddings yet)'.
    """
    query = (query or "").strip()
    if not query:
        return [], "keyword"
    vec_results, _ = vector_search(db, query, limit=limit, file_type=file_type)
    if vec_results:
        return order_results(vec_results), "vector"
    # No embeddings indexed yet (or no vector hits): keyword fallback.
    kw = keyword_search(db, query, limit=limit, file_type=file_type)
    method = "keyword (no embeddings yet)" if kw else "keyword"
    # If even keyword found nothing, still report vector-attempted context:
    if not kw:
        # Check whether ANY embeddings exist to pick an honest label.
        has_emb = db.query(Asset.id).filter(Asset.embedding.is_not(None)).first()
        method = "vector (no matches)" if has_emb else "keyword (no embeddings yet)"
    return order_results(kw), method
