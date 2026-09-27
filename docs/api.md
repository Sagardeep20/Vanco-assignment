# API Reference

Base URL: `http://localhost:8000`. Interactive docs: `/docs`. Every endpoint
below exists in `backend/app/api/`; error bodies are JSON `{"detail": ...}`.
Status codes: `400` bad input/state, `404` unknown asset/file, `409` job
already running, `422` validation failure, `500` database outage.

## Health

- `GET /` — service liveness. Response: `{"status":"ok","service":"<name>"}`.
- `GET /api/health` — same liveness payload under the API prefix.

## Indexing

- `POST /api/index` — run one full orchestration job (scan → process pending
  → embeddings → summary). No body. Success `200`:
  `{status, dataset_path, discovered, added, updated, skipped, unsupported,
  failed, duplicates, processed, processing_completed, processing_failed}`.
  Missing dataset → `400`; concurrent job → `409`.
- `GET /api/index/status` — in-memory progress snapshot: `{running, stage,
  dataset_path, total_assets, pending, processing, completed, failed,
  skipped, duplicate, unsupported, started_at, completed_at, last_error}`.
  `stage` is one of `idle|starting|scanning|processing|finalizing|completed`;
  idle zeros before any job.

## Processing

- `POST /api/process` — process `pending` rows of all types (generic
  pipeline: PDF text, descriptions, embeddings). Body (optional JSON):
  `{"limit": 100}` (1–1000). Success `200`:
  `{status, processed, completed, failed, errors[]}`.
- `POST /api/process/videos` — process video rows via sampled frames. Same
  body. Success `200`: `{status, total, processed, failed, skipped,
  errors[]}`; completed/non-pending videos count as `skipped`.
- `POST /api/process/reembed` — one-time re-embedding of `completed`
  image/PDF/video rows with the current provider (pixels / sampled
  frames / text). Same body (default limit 1000). Success `200`:
  `{status, processed, completed, failed, errors[]}`; only the
  `embedding` column is rewritten.

## Search

- `GET /api/search` — hybrid lexical + semantic ranked search. Query
  params: `q` (required, 1–500 chars), `limit` (1–100, default 20),
  `file_type` (`image|video|pdf`, optional). Success `200`:
  `{query, method, count, results: [{asset, score}]}` where `method` is
  `vector` or a `keyword…` fallback label and `score` is the fused
  relevance (higher is better, never NaN; `null` for keyword hits).

## Assets

- `GET /api/assets` — paginated listing. Query params: `file_type`
  (`image|video|pdf|other`, optional), `status` (any `AssetStatus`, optional),
  `limit` (1–500, default 50), `offset` (default 0). Success `200`:
  `{total, limit, offset, items: [AssetRead]}`.
- `GET /api/assets/stats` — counts: `{total, images, videos, documents,
  pending, completed, duplicates, failed, unsupported}`.
- `GET /api/assets/{id}` — full `AssetRead` detail (identity, hashes,
  description, extracted text, statuses/timestamps, media metadata).
  Unknown ID → `404`.
- `GET /api/assets/{id}/file` — serve the indexed bytes for inline preview
  with the correct media type. Unknown asset or deleted file → `404`;
  stored path outside the dataset directory → `400`. No path input exists —
  lookup is by database ID only.
- `POST /api/assets/{id}/open` — ask the server host to open the validated
  file with its OS default application (local-desktop convenience).
  Success `200`: `{opened, message, original_path}`; `opened:false` when the
  host cannot open it. Unknown asset/file → `404`, outside dataset → `400`.

`AssetRead` fields: `id, filename, original_path, relative_path, file_type,
mime_type, file_size, file_hash, created_at, modified_at, description,
extracted_text, status, error_message, processing_started_at,
processing_completed_at, last_indexed_at, width, height, duration_seconds`.
`status` is one of
`pending|processing|completed|failed|skipped|duplicate|unsupported`.
