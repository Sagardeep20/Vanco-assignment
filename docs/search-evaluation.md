# Semantic Search Evaluation

Date: 2026-09-27. Evaluator: reproducible manual run against the live local stack
(PostgreSQL + pgvector, `EMBEDDING_PROVIDER=auto` → deterministic hashed
512-d fallback, no model downloads). No production code was modified for this
evaluation.

Method: started the backend (`uvicorn app.main:app`), ran
`POST /api/index` on the real `data/` directory, then issued
`GET /api/search?q=<query>&limit=5` for each query below and recorded the
verbatim top results, scores, and method. Verdicts use three labels:

- **SUCCESS** — the expected asset is top-ranked with a clearly
  discriminating score, or (for no-match queries) no result exceeds a weak
  score.
- **PARTIAL** — the expected asset appears but with a weak score and/or for
  the wrong reason (e.g. filename token only, no content evidence).
- **FAILED** — the expected asset is missing, a wrong asset leads by a
  misleading margin, or all scores are ~0 for a query that should match.

## Dataset (actual contents of `data/`)

| Asset | Facts (inspected, not assumed) |
|---|---|
| `images/sample.png` | 320×240 solid steel-blue RGB(70,130,180); synthetic placeholder. Indexed `completed`; description is a metadata template, no color/object words. |
| `images/nested/nested.jpg` | 160×120 solid green RGB(46,138,87); synthetic placeholder. Indexed `completed`; template description. |
| `documents/sample.pdf` | 34-byte minimal PDF (`%PDF-1.4`, one empty object). pypdf extracts nothing. Indexed `completed` with "No extractable text found." |
| `videos/sample.mp4` | 2060 bytes: `ftypmp42` header + zeros, no real streams. ffmpeg/ffprobe absent on this host → status `failed`, **no description, no embedding**. |
| `readme.txt` | 31 bytes ("sample notes file (unsupported)"). Status `unsupported`, no description/embedding. |

Indexing state at eval time: total 5, completed 3, failed 1 (video), unsupported 1,
pending 0. Only the 3 completed rows have embeddings, so vector search can only
ever return those 3. Every query below returned `method: "vector"`.

**Dataset-size limitation (stated up front):** 4 real files, all synthetic
placeholders with template descriptions, is too small for a meaningful semantic
evaluation. Concept queries (colors, people, topics) have no true positive in
the corpus by construction. Results below are reported honestly against that
corpus; nothing was fabricated.

## Query results

### Q1 — "sample png image" (image concept; expect `images/sample.png`)
- Top results: `sample.png` (0.5455), `sample.pdf` (0.3873), `nested.jpg` (0.1054).
- Expected asset appeared: yes, ranked #1 with a clear margin.
- Verdict: **SUCCESS**. Exact filename/type token overlap retrieves correctly.

### Q2 — "nested photo" (image concept; expect `images/nested/nested.jpg`)
- Top results: `nested.jpg` (0.3873), `sample.pdf` (0.0), `sample.png` (0.0).
- Expected asset appeared: yes, ranked #1.
- Verdict: **SUCCESS**. Filename token "nested" discriminates; unrelated rows score exactly 0.

### Q3 — "blue picture" (object/color concept; `sample.png` is steel-blue)
- Top results: `sample.pdf` (0.0), `nested.jpg` (0.0), `sample.png` (0.0).
- Expected asset appeared: no ranking signal at all (three-way 0.0 tie, arbitrary order).
- Verdict: **FAILED**. Limitation: template descriptions contain no color or
  visual-content words, and hashed token embeddings have no notion of color, so
  a visually correct query cannot match.

### Q4 — "green landscape photo" (scene concept; `nested.jpg` is solid green)
- Top results: all three rows at 0.0.
- Expected asset appeared: no.
- Verdict: **FAILED**. Same limitation as Q3: no visual semantics in text or vectors.

### Q5 — "woman standing with a cat" (people/scene; no such content exists)
- Top results: `sample.png` (0.0845), `nested.jpg` (0.0816), `sample.pdf` (0.0).
- Expected: no strong match. Max score 0.085 (incidental stop-word overlap).
- Verdict: **SUCCESS** (correct rejection). The system correctly returns nothing
  convincing, though note even the weak scores are noise, not understanding.

### Q6 — "customer testimonial video" (video concept; expect `videos/sample.mp4`)
- Top results: `sample.png` (0.4364), `sample.pdf` (0.3873), `nested.jpg` (0.1054).
- Expected asset appeared: **no** — the only video has no embedding (failed
  processing without ffmpeg) so it cannot be returned by vector search at all.
- Verdict: **FAILED**. Two compounding causes: (a) unprocessed assets are
  invisible to semantic search; (b) worse, an unrelated image leads at 0.44 —
  hashed-bucket collisions produce confident-looking scores for unrelated
  queries, so the score cannot be trusted as a relevance signal.

### Q7 — "PDF about machine learning" (document topic; corpus PDF has no text)
- Top results: `sample.pdf` (0.1118), others 0.0.
- Expected asset appeared: yes, ranked #1, but only via the filename token
  "pdf"; score is weak and there is zero topical evidence (no extractable text).
- Verdict: **PARTIAL**. Right file, wrong reason, weak confidence.

### Q8 — "sample document pdf" (document topic; expect `documents/sample.pdf`)
- Top results: `sample.pdf` (0.6455), `sample.png` (0.3273), `nested.jpg` (0.0).
- Expected asset appeared: yes, #1 with a strong margin.
- Verdict: **SUCCESS**. Multi-token filename/description overlap works well.

### Q9 — "video clip mp4" (video concept; expect `videos/sample.mp4`)
- Top results: `sample.pdf` (0.5164), `sample.png` (0.3273), `nested.jpg` (0.0).
- Expected asset appeared: no (no video embedding exists, as in Q6).
- Verdict: **FAILED**. A PDF leads at 0.52 for a video query — again
  hash-collision noise outranking any meaningful signal, with the true video
  unretrievable.

### Q10 — "sample" (cross-media; expect `sample.png` + `sample.pdf`)
- Top results: `sample.pdf` (0.6708), `sample.png` (0.5669), `nested.jpg` (0.0).
- Expected assets appeared: yes, both, top-2 with high scores.
- Verdict: **SUCCESS**. Cross-media keyword-style retrieval is the system's
  strongest case.

### Q11 — "notes readme" (expect `readme.txt`)
- Top results: `sample.png` (0.1336), `nested.jpg` (0.1291), `sample.pdf` (0.0).
- Expected asset appeared: no — unsupported files get no description/embedding
  and are invisible to vector search (keyword fallback never triggers because
  vector rows always exist).
- Verdict: **FAILED**. Unsupported content is unsearchable; the returned weak
  hits are noise.

### Q12 — "quantum database replication" (control; expect no match)
- Top results: all three rows at 0.0.
- Expected: no match. Observed: exact-zero sweep.
- Verdict: **SUCCESS** (correct rejection; clean control).

Additional check (not counted): `q=sample&file_type=pdf` returned only
`sample.pdf` (0.6708) — file-type filtering composes correctly with ranking.

## Summary

- Queries evaluated: **12** (image ×3, people/scene ×1, object/color ×1, video ×2,
  PDF/topic ×2, cross-media ×1, no-match controls ×2).
- **Successful: 6** (Q1, Q2, Q5, Q8, Q10, Q12).
- **Partial: 1** (Q7).
- **Failed: 5** (Q3, Q4, Q6, Q9, Q11).

Major failure patterns:
1. **No true visual/topic semantics.** Descriptions are filename/size templates;
   colors, objects, scenes, and document topics are unretrievable (Q3, Q4, Q7).
   A real vision/caption model would be needed — explicitly out of scope.
2. **Uncalibrated hash-collision scores.** Unrelated queries score up to ~0.52
   (Q6, Q9) while true weak matches score ~0.11 (Q7). There is no score
   threshold that separates hits from noise.
3. **Unprocessed content is invisible.** The failed video and the unsupported
   text file have no embeddings, so video queries (Q6, Q9) and notes queries
   (Q11) can never succeed; the API still returns other rows instead of an
   honest empty set.

Current limitations: synthetic 4-file corpus (see table); deterministic hash
embeddings match shared tokens only; `method` was `vector` for every query so
keyword-fallback behavior is unevaluated here (covered by backend tests);
scores are cosine similarities over incidental token overlap, not relevance.

## Addendum (2026-09-28): CLIP provider + hybrid ranking on the real corpus

The section above describes the deterministic-fallback era and is kept as
a historical baseline. After enabling the local OpenCLIP ViT-B/32 provider
(images from pixels, videos from sampled frames, PDFs from text) with
hybrid lexical + per-type-normalized semantic ranking, the same live
procedure (`POST /api/index`, then `GET /api/search`) was re-run against
the real `data/` corpus. Verbatim top-5 results (fused relevance scores):

- "dog" → `images/dog.jpg` (+6.21), `documents/sample.pdf` (+1.24),
  `videos/nature.mp4` (+1.00), `images/nested/nested.jpg` (+0.29),
  `images/sample.png` (+0.17). Exact filename + semantic agreement wins.
- "forest" → `images/landscape.jpg` (+1.65), `documents/sample.pdf`
  (+1.18), `videos/nature.mp4` (+1.00), `images/nested/nested.jpg`
  (+0.90), `images/person.jpg` (+0.48). No filename/path contains
  "forest": pure visual semantic retrieval.
- "a lush green forest landscape" → `images/landscape.jpg` (+4.77),
  `images/nested/nested.jpg` (+1.60), `documents/sample.pdf` (+1.26),
  `videos/nature.mp4` (+1.00), `images/sample.png` (+0.22).

Raw CLIP cosine on this corpus explains why fusion is required:
unrelated PDF text–text similarities sit at 0.55–0.81 while correct
text–image matches score 0.15–0.30, so sorting raw cosine buries relevant
images under unrelated PDFs. No keywords were added to any description;
no query-specific rules exist in the ranking code.
