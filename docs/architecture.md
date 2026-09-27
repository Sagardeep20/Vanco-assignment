# Architecture

Companion to the [README](../README.md). This document describes component
responsibilities and the runtime flows. No background workers, queues, cloud
services, or authentication exist — every flow below is synchronous,
in-process, and local.

## Component architecture

- **Frontend** (`frontend/src/`): `App.tsx` owns search (debounced input +
  explicit Search, file-type filters, ranked results), indexing trigger,
  statistics, and the asset gallery. `PreviewModal.tsx` fetches
  `GET /api/assets/{id}` plus a `HEAD` existence check and renders
  image / `<video>` / PDF `<iframe>` previews with metadata, copy-path, and
  open-original actions. No state management library, no test framework.
- **API layer** (`backend/app/api/`): thin FastAPI routers —
  `health`, `index` (orchestrated job + in-memory status),
  `process` (generic + video batch), `search` (query → ranked results),
  `assets` (list/stats/detail/file/open). Routers validate input, map
  domain outcomes to HTTP codes, and never expose tracebacks.
- **Services** (`backend/app/services/`):
  - `scanner.py` — recursive walk, extension classification, SHA-256,
    Pillow dimensions, best-effort ffprobe metadata, idempotent upsert,
    duplicate detection, per-file failure isolation (including a loop-level
    guard for unexpected per-path errors).
  - `ai.py` — PDF text via pypdf, deterministic description templates,
    optional already-present Ollama override (never pulls), generic
    `process_asset`/`process_pending_assets`.
  - `video.py` — duration, capped frame timestamps (~10 s interval,
    max 6), ffmpeg extraction to a temp dir (always removed), per-frame
    notes via the vision abstraction, combined description + embedding.
  - `embeddings.py` — provider abstraction (`auto|local|deterministic`),
    512-d L2-normalized vectors. Text via `embed_text()` (shared by
    indexing and search queries); images from actual pixels and videos
    from mean-pooled sampled frames via `embed_image()` on the same
    OpenCLIP model; PDFs from extracted/searchable text. Deterministic
    hashed fallback when no local model is present; text fallback when
    no image encoder is available.
  - `search.py` — hybrid ranking over embedded rows: generic lexical
    relevance (filename/path/description/text term matches) fused with
    per-asset-type-normalized semantic similarity
    (`LEXICAL_WEIGHT * lexical + semantic_z`), so exact matches win and
    visual similarity decides the rest; rows without usable similarity
    never outrank scored rows. Keyword fallback when nothing is embedded.
  - `indexing.py` — orchestration: scan → process pending → reconcile →
    summary, with an in-memory running flag + lock and per-asset isolation.
  - `files.py` — ID-based lookup support: resolve-and-confine to the
    dataset root, media-type guess, shell-free OS open.
- **Persistence** (`backend/app/database/`, `models/`, `alembic/`):
  SQLAlchemy engine/session, single `assets` table with `vector(512)`,
  status enum (`pending/processing/completed/failed/skipped/duplicate/
  unsupported`), Alembic migration owning the schema. Indexing progress is
  deliberately **not** a table — it is module-level memory in
  `services/indexing.py`.

```text
Local folder ──scan──▶ Scanner ──upsert──▶ PostgreSQL assets table
                              ┌─ pending rows ─▶ ai.py / video.py
                              │                    ├─ description / PDF text
                              │                    └─ embed_text() ─▶ vector(512)
                              └─ GET /api/search?q= ─▶ embed_text(q)
                                                       ─▶ pgvector cosine rank
                                                       ─▶ results + scores
                                                        ─▶ PreviewModal ─▶ /file
```

## Data flow

1. Files land in `data/images|videos|documents/` (nested OK, dotfiles ignored).
2. `scan_dataset` classifies by extension, hashes (SHA-256), reads Pillow
   dimensions / ffprobe metadata, and upserts keyed by `relative_path`.
3. New rows are `pending` (or `duplicate` for repeated hashes,
   `unsupported` for other extensions, `failed` for unreadable files).
4. Processing fills `description`, `extracted_text` (PDFs), and `embedding`
   (image pixels / sampled video frames / PDF text, text fallback),
   moving rows to `completed` (or `failed` with `error_message`).
5. Search embeds the query with the same provider, fuses lexical relevance
   with per-type-normalized similarity, and ranks embedded rows.
6. Preview serves bytes by asset ID; original access reuses the same
   validated resolution plus an optional OS open.

## Indexing flow

`POST /api/index` → `run_indexing_job` (409 if the in-memory `running` flag
is set) → `scan_dataset` → for each `pending` row (limit 1000): videos via
`process_video_asset`, everything else via `process_asset`, each isolated so
one failure only marks its row → recount statuses from the database →
`running=false`, `stage=completed`, summary returned and retained in memory.
`GET /api/index/status` reports the snapshot at any time (`idle` before the
first job).

## Search flow

`GET /api/search?q=&limit=&file_type=` → blank rejected (422) →
`embed_text(q)` with the configured provider → pgvector
`cosine_distance` candidates over rows with non-null embeddings (optional
`file_type` filter, lexical matches rescued past the raw-distance cutoff)
→ per-type z-score normalization + generic lexical scoring →
`LEXICAL_WEIGHT * lexical + semantic_z` fused relevance, best first
(never NaN) → `(asset, relevance)` pairs labeled `vector`, else
case-insensitive LIKE fallback labeled accordingly. Ranking is entirely
backend-side; the UI displays the order and scores unchanged.

## Failure handling

Failures are row-scoped everywhere: scanner records `failed` rows,
processors trap content/tooling errors per asset, orchestration adds a
defensive per-asset catch with rollback, and the scan loop adds a final
per-path catch. Only missing dataset directories (400), concurrent jobs
(409), unknown IDs (404), bad parameters (422), and database outages (500)
fail whole requests — always as JSON `detail` messages, never tracebacks.

## Incremental indexing flow

Second run, nothing changed → scanner matches `relative_path` + `file_hash`,
refreshes trivial fields, keeps status/AI outputs, counts `skipped`; zero
pending rows → processing phase is a no-op and descriptions/embeddings are
byte-identical. Changed content → hash differs → row reset to `pending` with
`description/extracted_text/embedding` cleared → reprocessed. Deleted files
are simply not visited: their rows persist untouched by design.
