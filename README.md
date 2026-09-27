# AI Digital Asset Manager

Local-only AI-powered Digital Asset Management system. Point it at a folder of
images, videos, and PDFs; it scans and hashes every file, generates
descriptions and 512-dimensional embeddings, stores everything in
PostgreSQL + pgvector, and lets you find assets with natural-language search,
preview them in the browser, and locate the original files on disk.

No authentication, no background job queue, no cloud services. AI models are
never downloaded automatically — everything works offline with deterministic
local processing, and optionally uses ML models only if they are already
present on the machine.

## Project overview

Personal and team media collections grow into thousands of unsorted files
across nested folders. Keyword file names ("IMG_0421.jpg") don't describe
content, so finding "that customer testimonial clip" or "the machine-learning
PDF" means manual browsing.

This system solves that locally:

- It ingests a dataset directory (images, videos, PDFs) with recursive,
  incremental, idempotent scanning.
- It records SHA-256 hashes (duplicate detection), media metadata
  (dimensions, duration), extracted PDF text, and generated descriptions.
- It stores a 512-d embedding per asset in pgvector for cosine-similarity
  natural-language search with ranking and file-type filters.
- It serves in-browser previews plus safe access to the original file on disk.

## Core workflow

```text
Local folder (data/images|videos|documents)
  → scan (recursive, SHA-256 hash, metadata)
  → content processing (descriptions, PDF text extraction, video frames)
  → embeddings (512-d, shared provider for assets and queries)
  → PostgreSQL + pgvector (assets table, cosine similarity)
  → natural-language search (GET /api/search?q=...)
  → ranked results (similarity score, file-type filters)
  → preview (image / video player / PDF embed) + original file access
```

## Supported media

Images (`.jpg`, `.jpeg`, `.png`, `.webp`):
- Metadata: width/height via Pillow, file size, MIME type, filesystem timestamps.
- AI description behavior: deterministic template built from filename,
  dimensions, and size (e.g. "Image 'sunset' (320x240px, 782 bytes). …").
  There is no vision model by default, so descriptions carry no color, object,
  or scene semantics. If `AI_PROVIDER=ollama` and the configured model is
  already present in the local Ollama instance, frame/image descriptions may
  come from it instead — the app only checks `/api/tags` and never pulls.

Videos (`.mp4`, `.mov`, `.mkv`, `.webm`):
- Container metadata via ffprobe when available (width/height/duration);
  missing/unreadable metadata never fails the file.
- Frame sampling: approximately one frame every 10 seconds
  (`FRAME_INTERVAL_SECONDS = 10.0`), capped at **6 frames max**
  (`MAX_FRAMES = 6`) so long videos stay cheap. Frames go to a temporary
  directory that is always deleted; only the combined description and its
  embedding are stored.
- **ffmpeg/ffprobe requirement:** real video processing needs both binaries on
  `PATH`. Without them, videos are marked `failed` with the message
  "ffmpeg/ffprobe not available on this system; …" and the job continues.

PDFs:
- Text extraction via pypdf (pure Python), truncated at 8000 characters;
  unreadable PDFs keep `extracted_text = NULL` and get a "No extractable text
  found." description instead of failing.
- Limitation for scanned/image-only PDFs: there is no OCR, so image-only pages
  yield no text and are searchable only by filename/description.

Any other extension (e.g. `.txt`) is recorded with `file_type=other`,
`status=unsupported`: visible in listings/stats, never processed, never fatal.

## Architecture

- **React + Vite + TypeScript** (`frontend/`): search UI with debounce,
  file-type filters, stats, asset gallery, preview modal. No frontend test
  framework; correctness is verified with `npm run build` (`tsc -b && vite build`).
- **FastAPI** (`backend/app/`): sync REST API, routers per area, Pydantic
  schemas, auto docs at `/docs`.
- **PostgreSQL + pgvector** (`assets` table): one row per dataset file —
  identity, SHA-256, description/text, `vector(512)` embedding, status
  workflow, media metadata. Schema is owned by Alembic migrations.
- **SQLAlchemy 2** (`app/database/`): engine/session/`Base`, plus
  `python -m app.database.init_db` as a fail-fast connectivity/extension check.
- **Pillow**: image dimensions and frame thumbnail notes.
- **pypdf**: pure-Python PDF text extraction with graceful degradation.
- **ffmpeg/ffprobe** (optional system binaries, not Python deps): video
  metadata probing and frame extraction.
- **Embedding provider abstraction** (`app/services/embeddings.py`):
  `EMBEDDING_PROVIDER=auto|local|deterministic` (default `auto`). `local`
  loads 512-d OpenCLIP ViT-B/32 (`clip-ViT-B-32`) from already-installed
  packages and already-cached weights only (Hugging Face offline flags forced
  during load); `deterministic` is the hashed bag-of-words fallback; `auto`
  uses local when available. Images are embedded from actual pixels
  (`encode_image`), videos from mean-pooled sampled-frame embeddings, PDFs
  from extracted/searchable text, and text queries via `embed_text()` — all
  L2-normalized in one shared CLIP joint space. Every embedding path degrades
  gracefully to text (never a failed asset just for a missing encoder).
- **Local/optional vision provider** (`app/services/ai.py`): deterministic
  metadata templates by default; Ollama is consulted only when
  `AI_PROVIDER=ollama` with an already-pulled model, otherwise local fallback.

```text
+-----------------+      HTTP (Vite proxy / VITE_API_URL)      +------------------+
|  React + Vite   |  ----------------------------------------> |     FastAPI      |
|  search / stats /| <---------------------------------------- |  routers: health |
|  gallery / modal |              JSON + file bytes             |  index/process/  |
+-----------------+                                            |  search/assets   |
                                                               +--------+---------+
                                                                        |
                         +----------------------------------------------+-------------------------------+
                         |                                              |                               |
                 +-------v--------+   +----------------+   +-------------v------------+   +--------------v---------+
                 | scanner.py     |   | ai.py /        |   | embeddings.py          |   | files.py             |
                 | walk, SHA-256, |   | video.py /     |   | auto|local|            |   | ID-only lookup,      |
                 | Pillow/ffprobe,|   | indexing.py    |   | deterministic 512-d,   |   | dataset-confined     |
                 | upsert, dupes  |   | describe+embed |   | L2-normalized          |   | serve / OS-open      |
                 +-------+--------+   +--------+-------+   +-------------+----------+   +------------------------+
                         |                     |                         |
                         +---------------------+-------------------------+
                                               |
                                    +----------v-----------+
                                    | PostgreSQL+pgvector  |
                                    | assets (vector(512)) |
                                    +----------------------+
```

## Directory structure

```text
ai-digital-asset-manager/
  README.md  docs/  docker-compose.yml  scripts/  data/  backend/  frontend/
  backend/
    requirements.txt  alembic.ini  .env.example
    alembic/                      # migrations (pgvector ext + assets table)
    app/
      main.py                     # FastAPI app, CORS, router registration
      api/                        # health, index, process, search, assets
      core/config.py              # settings (DATASET_PATH, providers, …)
      database/                   # engine, session, init_db check helper
      models/asset.py             # Asset ORM, FileType/AssetStatus, dim 512
      schemas/                    # asset, index, listing, process, search
      services/                   # scanner, ai, video, embeddings, search,
                                  # indexing (orchestration), files
      workers/__init__.py         # reserved; no background workers exist
    tests/                        # scanner/ai/embeddings(+openclip/image)/
                                  # video/indexing/asset_file/reliability/
                                  # search_ranking/search_hybrid/reembed + conftest
  frontend/
    package.json  vite.config.ts  .env.example
    src/App.tsx                   # search, filters, stats, gallery, indexing
    src/PreviewModal.tsx          # image/video/PDF preview + original path
    src/main.tsx  src/index.css
  data/images|videos|documents/   # your files here (git-ignored)
  docs/search-evaluation.md       # 12-query search quality evaluation
  scripts/start-db.bat            # Windows helper to start Postgres
```

## Setup

Local setup WITHOUT Docker. You need your own PostgreSQL with the pgvector
extension installed (for example the official PostgreSQL installer plus the
pgvector extension for your platform). pgvector installation is **not**
automatic — the app only runs `CREATE EXTENSION IF NOT EXISTS vector` against
a server that already provides the extension module.

Prerequisites: Python 3.13+, Node.js 22+ with npm, PostgreSQL + pgvector
available at `localhost:5432` (if your server listens on a custom port such
as 5434, reflect it in `DATABASE_URL`), Git.

Backend:

```bash
cd ai-digital-asset-manager/backend
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
copy .env.example .env        # then edit DATABASE_URL / DATASET_PATH if needed
```

Optional semantic embeddings (large download, do once): the CLIP provider
is used automatically when `torch`, `open_clip_torch`, and cached ViT-B/32
weights are present; otherwise the app keeps working on the deterministic
fallback with no changes required:

```bash
pip install torch torchvision open_clip_torch
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"
```

Create the database objects (Postgres must be running and provide `vector`):

```bash
alembic upgrade head
python -m app.database.init_db
```

Start the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify `GET http://localhost:8000/api/health` returns
`{"status":"ok","service":"ai-digital-asset-manager"}`; interactive docs at
`http://localhost:8000/docs`.

Environment variables (`backend/.env`, see `.env.example`):
`DATABASE_URL`, `BACKEND_HOST`, `BACKEND_PORT`, `FRONTEND_URL`,
`DATASET_PATH` (default `./data`), `AI_PROVIDER`/`OLLAMA_URL`/`OLLAMA_MODEL`
(optional vision override, never auto-pulls), `EMBEDDING_PROVIDER`
(`auto|local|deterministic`, default `auto`), `LOCAL_EMBEDDING_MODEL`
(default `clip-ViT-B-32`), `LOCAL_EMBEDDING_MODEL_PATH` (optional explicit
weights directory).

Frontend:

```bash
cd ai-digital-asset-manager/frontend
npm install
copy .env.example .env         # VITE_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`. Production check: `npm run build`.

(There is also a `docker-compose.yml` + `scripts/start-db.bat` for running just
PostgreSQL via Docker; the Python/Node steps above stay the same.)

## Dataset

Place files under `data/images/`, `data/videos/`, `data/documents/`.
Nested subfolders work; hidden dotfiles (`.gitkeep`, `.DS_Store`) are ignored.
`DATASET_PATH` in `backend/.env` overrides the location (default `./data`).

Scanning is recursive and incremental: re-running indexing only adds new files,
updates changed files (content hash differs → row reset to `pending` with
stale description/embedding cleared), and skips unchanged files without
regenerating anything.

## Indexing

- `POST /api/index` — runs one full orchestration job: scan → process pending
  images/PDFs/videos (descriptions + 512-d embeddings) → persist statuses →
  return the summary (`discovered/added/updated/skipped/unsupported/failed/
  duplicates` plus `processed/processing_completed/processing_failed`).
  A second call while a job runs returns `409` with an "already running"
  message instead of starting another job.
- `GET /api/index/status` — in-memory progress: `running`, `stage`
  (`idle|starting|scanning|processing|finalizing|completed`), per-status
  counts (`total_assets/pending/processing/completed/failed/skipped/
  duplicate/unsupported`), `dataset_path`, `started_at`, `completed_at`.
  Idle before any job has run. The status lives in application memory only —
  no database table — so it resets on server restart.

Details:
- SHA-256 duplicate detection: same content under a different path keeps both
  rows; extras get `status=duplicate` and are skipped by processing.
- Incremental re-indexing: unchanged `completed` rows are never reprocessed;
  changed files become `pending` again and are reprocessed.
- Status values: `pending → processing → completed`, or `→ failed` with
  `error_message`; plus `duplicate`, `unsupported`, `skipped`.
- Failure isolation: a corrupt image, missing ffprobe data, unreadable PDF,
  failed video, or embedding error is recorded on that row; the job always
  finishes and reports per-asset errors (e.g. 100 files → 97 completed,
  3 failed).

## Processing endpoints

- `POST /api/process` (`{"limit": 100, 1–1000}`) — process `pending` rows of
  all types with the generic pipeline (PDF text, descriptions, embeddings).
- `POST /api/process/videos` (same body) — process video rows via sampled
  frames; returns `{total, processed, failed, skipped}`; completed videos are
  skipped, others untouched.
- `POST /api/process/reembed` (same body, default limit 1000) — one-time
  re-embedding of `completed` image/PDF/video rows with the current provider
  (images from pixels, videos from sampled frames, PDFs from text). Only the
  `embedding` column is rewritten; metadata, status, and failed/unsupported
  rows are preserved. Safe to re-run. Run once after switching embedding
  providers so old vectors match new queries:
  `curl -X POST http://localhost:8000/api/process/reembed`.
- `GET /api/assets`, `GET /api/assets/stats`, `GET /api/assets/{id}` —
  listing (with `file_type`/`status`/`limit`/`offset`), counts, and full
  detail used by the preview panel.

## Search

- Natural-language query → embedded with the **same** provider used at index
  time (`embed_text()`), so queries and stored vectors share one space.
- `GET /api/search?q=...&limit=20&file_type=image|video|pdf` runs **hybrid
  ranking**: a generic lexical score (exact filename > filename term >
  description/path/extracted-text term matches; filler words ignored) fused
  with semantic similarity normalized per asset type (z-score). This matters
  because raw CLIP text–text scores (PDFs, ~0.55–0.85) live in a far higher
  band than correct text–image scores (~0.15–0.30); sorting raw cosine buries
  relevant images under unrelated PDFs. Exact matches (e.g. `dog` →
  `dog.jpg`) dominate; pure visual similarity decides when nothing matches
  lexically (e.g. `forest` → `landscape.jpg`).
- The per-result `score` is the fused relevance (higher is better, never
  NaN); `method` is `vector`, or keyword fallbacks such as
  `keyword (no embeddings yet)` when nothing is embedded. Rows with
  missing/degenerate embeddings never outrank scored rows.
- File-type filters map directly to the `file_type` parameter (the UI offers
  All/Images/Videos/PDFs); `q` is 1–500 chars, `limit` 1–100.
- Deterministic embedding fallback: hashed bag-of-words, 512-d,
  L2-normalized. It matches shared tokens reliably (filenames, "sample",
  "pdf") but has no true semantics — colors, objects, and topics don't match.
- Optional local semantic provider: with `EMBEDDING_PROVIDER=local` (or `auto`
  when available), the 512-d CLIP model is loaded **only** from
  already-installed packages and already-cached weights (offline flags
  forced during load); if unavailable, `auto` falls back to deterministic and
  `local` returns a clear error. Models are never downloaded.

## Preview

Clicking any search result or gallery card opens a preview modal fed by
`GET /api/assets/{id}`: images render inline, videos use a native HTML5 player
with controls, PDFs embed in a browser viewer (plus "open in new tab").
The panel also shows filename, type, size, relative path, status (failed
assets show their error), description, and extracted-text excerpt. A missing
physical file is detected up front and reported instead of a blank viewer.

Safe original-file access: `GET /api/assets/{id}/file` looks assets up by
database ID only (never a user-supplied path), resolves the stored path, and
refuses anything outside the dataset directory (400) or missing (404) before
serving with the correct media type. "Open original" (`POST
/api/assets/{id}/open`) asks the server host to open the validated file with
its OS default app (no shell; fixed command + path argument) and otherwise
returns `opened:false`; the modal always offers the full original path with a
copy button, which covers remote-server use.

## Reliability

- Duplicate handling: identical files under different paths are all kept and
  visible; extras persist as `duplicate` and never crash a job.
- Corrupt files: unreadable images/PDFs become `failed` rows with a useful
  `error_message` and timestamps; the job continues.
- Missing files: nothing is auto-deleted — database rows persist by design and
  stay visible; preview/file endpoints return clean 404s.
- Unsupported files: recorded as `unsupported`, skipped by every processing
  path, counted in stats.
- ffmpeg failures: videos fail with "ffmpeg/ffprobe not available…" when the
  binaries are absent; per-frame extraction errors are skipped, and zero
  extracted frames fail only that video.
- Embedding failures: a provider error fails only that asset
  (`status=failed`, message, processing timestamps); the batch continues.
- Path traversal protection: preview/open resolve and confine every path to
  the dataset root; there is no endpoint that accepts a filesystem path, and
  OS opening never uses a shell.
- Scan safety net: any unexpected per-file error (e.g. a symlink escaping the
  dataset) is counted as failed with an error entry instead of aborting the scan.

## Evaluation

Full report: [`docs/search-evaluation.md`](docs/search-evaluation.md) —
originally 12 realistic queries run live against a small synthetic corpus
with the deterministic fallback (verbatim top results, scores, and
success/partial/failed verdicts).

Honest summary of that baseline run: **6 successful, 1 partial, 5 failed.**
Filename/token queries retrieved correctly, and both no-match controls
correctly returned nothing strong. Failures were all explained: template
descriptions carry no color/object/topic semantics and hashed-bucket
collisions can outrank true weak matches. No accuracy beyond what that
document demonstrates is claimed for the fallback.

Since then, with the optional CLIP provider and hybrid ranking, live runs
against the real `data/` corpus verify: `dog` → `dog.jpg` first (exact +
semantic), `forest` → `landscape.jpg` first (pure visual similarity, no
filename match), `a lush green forest landscape` → `landscape.jpg` first —
see the dated addendum at the end of `docs/search-evaluation.md`. Semantic
quality on other collections still depends on having the local semantic
model available.

## Tests

Backend (isolated `assets_test` database, created automatically):

```bash
cd ai-digital-asset-manager/backend
python -m pytest tests -v
```

Current result: **147 passed, 1 skipped** (the skip is a symlink test that
needs OS symlink privileges; the guarded code path is covered by a mocked
equivalent). Suites: scanner, AI processing, embeddings (deterministic +
OpenCLIP), image embeddings, video, indexing orchestration, asset
files/preview security, reliability, search ranking (null/zero-embedding
ordering), hybrid search ranking, re-embedding, plus API endpoint tests.
No frontend test framework exists by design.

Frontend build:

```bash
cd ai-digital-asset-manager/frontend
npm run build
```

Must complete `tsc -b && vite build` with no errors.

## Limitations

- A local semantic/vision model must already be available; models are never
  downloaded automatically (offline flags are forced during model load).
- ffmpeg/ffprobe are required on `PATH` for real video processing; otherwise
  videos fail with a clear message.
- Scanned/image-only PDFs need OCR for deeper understanding (not included);
  only embedded text is extracted (truncated at 8000 chars).
- Indexing progress/status is in-memory and per-process: restarts lose
  history and concurrent server workers don't share the running-job guard.
- Deleted files remain as database rows by design (no auto-prune); previews
  report them as missing.
- The evaluation corpus is small; semantic quality claims are limited to
  what `docs/search-evaluation.md` demonstrates.

## Demo flow

A 1–2 minute end-to-end demo:

1. Put files into `data/images/`, `data/videos/`, `data/documents/`.
2. Start the backend (`uvicorn … --port 8000`) and frontend (`npm run dev`).
3. Press **Start Indexing** in the Dataset section.
4. Watch `GET /api/index/status` (or the returned summary) for
   added/processed/completed counts.
5. Type a natural query (e.g. `sample png image`) and press **Search** —
   results arrive ranked with similarity scores.
6. Apply the **PDFs/Images/Videos** filter to narrow results.
7. Click a result to open the asset preview (image / video player / PDF).
8. Copy or **Open original** via the displayed original path.
9. Open `docs/search-evaluation.md` to show the honest 12-query quality report.
