# Custom knowledge RAG (technical PDF guides)

This project builds a **retrieval-augmented generation (RAG)** pipeline for long, technical product guides (for example custom blinds and shades): ingest digital PDFs, preserve structure including tables, **chunk** with overlap and metadata, **embed** into a vector store with a **keyword** index, run **hybrid retrieval** with **reranking**, and answer questions with citations.

**Maintaining this document:** Whenever you add or change setup steps, CLI commands, dependencies, deployment targets, environment variables, or pipeline stages, **update this README in the same change** so it stays the single source of truth for how to run and operate the app.

---

## Roadmap (target architecture)

Use this as a checklist. Items marked **Done** reflect the repository today; others are planned.

| Stage | What it does | Production notes |
|--------|----------------|------------------|
| **Phase 1 — Ingest & extract** | PDF → structured IR + **registry** (SQLite), **content-addressed** storage under `documents/<sha256>/`, idempotent ingest | Long-running work should run in a **worker** (queue/cron); production swaps local `storage/` for **object storage** + managed DB |
| **Phase 2 — Chunk** | **Done (entry):** read `extraction.json`, emit **JSONL** chunk records with sliding windows (default **20% overlap**), heuristic **section_path**, `text_embed` prefix; optional registry update | Bump `CHUNKER_VERSION` in `rag_chunker/__init__.py` when logic changes; same version + existing `chunks.jsonl` skips unless `--force` |
| **Phase 3 — Embed & index** | **Done (entry):** embed `text_embed`, store **float32** vectors + **FTS5** keyword index in per-document **SQLite** (`storage/index/<sha256>.sqlite`); **`search`** runs **cosine + BM25 → RRF** | Production: swap SQLite for **pgvector + Postgres FTS** or a hosted hybrid service; keep `embedding_model` + dims in registry |
| **Phase 4 — Retrieve** | **Done (entry):** wide hybrid pool + **rerank** (`none` \| **cohere** \| **cross-encoder**); **`rag_eval`** JSONL runner (**MRR**, **recall@k**); **`POST /v1/retrieve`** on **`rag_api`** | Tune thresholds on a fixed eval set |
| **Phase 5 — Generate** | **Done (entry):** **`rag_generate ask`** — OpenAI Chat with **`rag_retrieve`** context + source list; **`POST /v1/ask`** on **`rag_api`** | Add logging, moderation, streaming, non-OpenAI providers |
| **Platform** | **Done (entry):** optional **API keys**, **CORS**, **rate limits**, **request IDs** + logs; **`POST /v1/ingest`** runs **ingest → chunk → index** with job rows in **`api_jobs.sqlite`**; **`REDIS_URL`** + **RQ** worker (**`rag`**) or **`BackgroundTasks`** if Redis unset; **Docker Compose**: **Redis** + **api** + **worker** + **`/data` volume** | Tenants/quotas; **S3** + **managed DB** at scale |

---

## What is implemented today

- **Phase 1 (ingest + extract):**
  - **PyMuPDF** extraction: text blocks and **tables** (`find_tables()`), **reading order**, deduplication of text inside table regions → JSON artifact (`ExtractionArtifact`, `schema_version: extraction.v1`).
  - **Registry:** SQLite at `storage/registry.db` tracks each document by **SHA-256** of file bytes, status (`pending` / `ready` / `failed`), paths, extraction versions, and errors.
  - **Content-addressed layout:** `storage/documents/<sha256>/source.pdf` and `extraction.json` (gitignored with the rest of `storage/`).
  - **Idempotent ingest:** same file hash and same `extraction_version` skips re-processing unless `--force`.
- **Phase 2 (chunk — entry):**
  - **`rag_chunker`** reads an `ExtractionArtifact` and writes **`chunks.jsonl`** (one JSON object per line) plus an optional **`chunks_manifest.json`**.
  - Long **text** blocks use a **sliding window** with configurable `max_chars` and **overlap ratio** (default **0.2**). **Tables** split by row groups with the header repeated; oversized non-table markdown falls back to character windows.
  - **Section hints:** leading lines in a text block that match simple heading heuristics update a rolling `section_path` (bumped when you change rules — document in code).
  - **Registry:** optional `chunks_relpath` + `chunker_version` on the document row after `--write-registry`.
- **Phase 3 (embed + index — entry):**
  - **`rag_index`** reads **`chunks.jsonl`**, calls the **OpenAI Embeddings API** (default model `text-embedding-3-small`, override with `OPENAI_EMBEDDING_MODEL` or `--embedding-model`), and writes a **SQLite** database with vector blobs and an **FTS5** table on chunk text + section path.
  - **`python -m rag_index search`** runs **hybrid retrieval**: dense cosine similarity + **BM25**-style FTS scores, merged with **reciprocal rank fusion (RRF)**.
  - **Registry** fields: `index_db_relpath`, `embedding_model`, `embedding_dimensions` (via `--write-registry`).
  - **Tests / CI:** set **`RAG_INDEX_FAKE_EMBEDDINGS=1`** for deterministic pseudo-embeddings (no API key).
- **Phase 4 (retrieve + rerank — entry):**
  - **`rag_retrieve`** — wide RRF pool (`--candidates`), then **`--rerank`**: **`none`**, **`cohere`** (`COHERE_API_KEY`), or **`cross-encoder`** (local **sentence-transformers** / **PyTorch**, `CROSS_ENCODER_MODEL` defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - **`rag_eval`** — eval cases as **JSONL** (`EvalCase`: `id`, `question`, optional `gold_chunk_ids`, `gold_pages`, `gold_substrings`); reports **MRR** and **recall@k**.
- **Phase 5 (generate — entry):**
  - **`rag_generate ask`** — runs **`retrieve`** (same `--rerank` / pool flags), builds a context block from hits, calls **OpenAI Chat Completions** (`OPENAI_API_KEY`, optional `OPENAI_CHAT_MODEL`, default `gpt-4o-mini`).
- **HTTP API (`rag_api`) + workers (`rag_worker`):** **FastAPI** — **`GET /health`**, **`POST /v1/retrieve`**, **`POST /v1/ask`**, **`POST /v1/ingest`** (full pipeline **ingest → chunk → embed/index**), **`GET /v1/jobs/{job_id}`**; uploads land under **`storage/incoming/`** then **`REDIS_URL`** + **RQ worker** (Docker) or **in-process `BackgroundTasks`** when **`REDIS_URL`** is unset; optional **`RAG_API_KEYS`**, CORS, rate limits, **`X-Request-ID`**.
- **CLI:** `rag_extractor` also exposes **`eval`**, **`ask`** (forwarding to `rag_eval` / `rag_generate`).
- **Tests:** smoke PDF; fast tests for ingest, chunk, index, retrieve, **eval** (fake embeddings), API **health** + validation.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `rag_extractor/` | Extraction + ingest + registry + CLI |
| `rag_chunker/` | Phase 2 chunking (`chunker`, `models`, CLI) |
| `rag_index/` | Phase 3 embeddings, SQLite+FTS store, hybrid `search`, CLI |
| `rag_retrieve/` | Phase 4 candidate pool + rerank (`pipeline`, `rerankers`, CLI) |
| `rag_eval/` | Eval JSONL schema + MRR / recall@k runner |
| `rag_generate/` | LLM answers with retrieved context (`answer`, CLI) |
| `rag_api/` | FastAPI HTTP API, middleware, job store |
| `rag_worker/` | Full document pipeline + RQ task entrypoints + **`python -m rag_worker.worker`** |
| `rag_storage/` | Postgres schema/init, S3/local **blob** helpers, config |
| `fixtures/eval/` | Example eval JSONL (lines starting with `#` are ignored) |
| `fixtures/pdfs/` | Example PDFs for local testing (optional; add your own) |
| `storage/` | `registry.db`, `api_jobs.sqlite` (job rows), `incoming/` (uploaded PDFs awaiting processing), `documents/<sha256>/`, `index/<sha256>.sqlite`, optional `artifacts/` (**gitignored**) |
| `tests/` | Unit/smoke tests |
| `requirements.txt` | Python dependencies |

---

## Prerequisites

- **Python 3.10+** (3.11+ recommended). The repo has been run on newer CPython versions; pin your production version explicitly when you deploy.
- **Git** (optional, for version control).

---

## Local setup

### 1. Clone or copy the project

```powershell
cd c:\path\to\custom_knowledge_program
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. (Optional) Add a test PDF

Place a digital PDF under `fixtures/pdfs/` or pass any file path to the CLI.

---

## Usage (current)

### Environment

Copy [`.env.example`](.env.example) to **`.env`** in the project root and set secrets there. With **`python-dotenv`** (from `requirements.txt`), importing **`rag_storage`** loads **`.env`** into the process so you do not need to export variables in the shell for local runs. Variables already set in the environment (including in Docker) take precedence. **`.env`** is gitignored; keep **`.env.example`** committed as the template.

| Variable | Meaning |
|----------|---------|
| `RAG_STORAGE_ROOT` | Root for local files: `documents/`, `index/*.sqlite` (SQLite mode), `incoming/`, etc. (default: `<project>/storage`) |
| `DATABASE_URL` | If set (PostgreSQL + **pgvector**), **registry**, **vector index**, and **API job rows** use this database instead of local `registry.db` / `api_jobs.sqlite`. Index rows are keyed by document `content_sha256`; `index_db_relpath` in the registry becomes **`postgres`**. Requires **`CREATE EXTENSION vector`** (see `rag_storage/pg.py` init). |
| `RAG_PG_VECTOR_DIM` | Column dimension for `vector(...)` (default **`1536`**, matching `text-embedding-3-small`). Change only if your embedding size differs and you recreate the DB schema. |
| `RAG_S3_BUCKET` | If set (with AWS credentials / endpoint), **ingest** mirrors **`source.pdf`** and **`extraction.json`** to this bucket after a successful extract (keys mirror paths under `RAG_STORAGE_ROOT`). |
| `RAG_S3_PREFIX` | Optional key prefix inside the bucket (no leading/trailing slash). |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Standard AWS SDK credentials for S3. |
| `AWS_ENDPOINT_URL` or `S3_ENDPOINT_URL` | Use for **MinIO**, **LocalStack**, or **Cloudflare R2**-compatible endpoints. |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | Region for the S3 client (default `us-east-1` in code if unset). |
| `OPENAI_API_KEY` | Required for real embeddings (Phase 3) unless using fake embeddings |
| `OPENAI_EMBEDDING_MODEL` | Default `text-embedding-3-small` |
| `RAG_INDEX_FAKE_EMBEDDINGS` | Set to `1` to use deterministic fake vectors (tests / offline) |
| `RAG_INDEX_FAKE_DIMS` | Dimension for fake vectors (default `256`) |
| `COHERE_API_KEY` | For Phase 4 `--rerank cohere` |
| `COHERE_RERANK_MODEL` | Default `rerank-english-v3.0` |
| `CROSS_ENCODER_MODEL` | Hugging Face id for **cross-encoder** rerank (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `OPENAI_CHAT_MODEL` | Chat model for **`rag_generate ask`** (default `gpt-4o-mini`) |
| `RAG_API_HOST` | HTTP API bind host (default `127.0.0.1`; **`python -m rag_api`**) |
| `RAG_API_PORT` | HTTP API port (default `8000`) |
| `RAG_API_KEYS` | Optional comma-separated API keys. If set, **`/v1/*`** requires **`X-API-Key: <key>`** or **`Authorization: Bearer <key>`** ( **`/health`** stays open). |
| `RAG_API_CORS_ORIGINS` | Optional comma-separated allowed browser origins (enables CORS when non-empty). |
| `RAG_API_RATE_LIMIT` | SlowAPI limit for **`/v1/retrieve`**, **`/v1/ask`**, **`GET /v1/jobs/...`** (default `120/minute` per client IP). |
| `RAG_API_RATE_LIMIT_INGEST` | Limit for **`POST /v1/ingest`** (default `30/minute`). |
| `RAG_API_LOG_LEVEL` | Root/uvicorn log level (default `INFO`). |
| `REDIS_URL` | If set (e.g. `redis://localhost:6379/0`), **`POST /v1/ingest`** enqueues work on queue **`rag`**; run **`python -m rag_worker.worker`** (or the **worker** service in Docker). If unset, the pipeline runs in **`BackgroundTasks`** in the API process (dev-friendly; not for multi-replica APIs). |

### HTTP API

Requires the same **`RAG_STORAGE_ROOT`** layout as the CLI. For **`POST /v1/ask`** and for **embedding during ingest**, set **`OPENAI_API_KEY`** (or **`RAG_INDEX_FAKE_EMBEDDINGS=1`** for tests). Responses include **`X-Request-ID`** (pass the same header to correlate logs).

**Start server** (default `http://127.0.0.1:8000`; use `--host 0.0.0.0` to listen on all interfaces):

```powershell
python -m rag_api --host 127.0.0.1 --port 8000
```

**Endpoints**

| Method | Path | Body / notes |
|--------|------|----------------|
| `GET` | `/health` | Returns `{"status":"ok"}`. No API key required. |
| `POST` | `/v1/retrieve` | JSON: **`query`** (required); **`sha256`** (registry id or prefix) *or* **`index_db`** (path to index SQLite); optional **`top`**, **`candidates`**, **`rerank`**. Response: **`hits`**. |
| `POST` | `/v1/ask` | JSON: **`question`** (required); **`sha256`** or **`index_db`**; optional **`model`**, **`top`**, **`candidates`**, **`rerank`**. Response: **`answer`**, **`chunk_ids`**, **`pages`**. |
| `POST` | `/v1/ingest` | **Multipart** **`file`**: PDF only (must begin with **`%PDF`**); optional query **`force=true`**. Returns **`202`** with **`job_id`** and **`status`**: **`queued`** (Redis) or **`pending`** (in-process). Runs **ingest → chunk → index**; poll until **`ready`** before **`/v1/retrieve`** / **`/v1/ask`**. |
| `GET` | `/v1/jobs/{job_id}` | Poll job status: **`pending`**, **`queued`**, **`ingesting`**, **`chunking`**, **`indexing`**, **`ready`**, **`failed`**; includes **`content_sha256`** and **`error_message`** when known. |

OpenAPI docs: **`GET /docs`** (Swagger UI) when the server is running.

### Docker

**`docker-compose.yml`** runs **Postgres (pgvector)**, **Redis**, the **API**, and a **worker** (shared **`/data`** volume for PDFs, chunks JSONL, and SQLite files when not using RDS/S3-only layouts).

```powershell
docker compose build
docker compose up
```

- **`DATABASE_URL`** is set for **api** and **worker** to `postgresql://rag:rag@db:5432/rag` (override for managed Postgres).
- **`OPENAI_API_KEY`**: required for **`/v1/ask`** and for real embeddings on **`/v1/ingest`** (set on **api** and **worker**).
- **`RAG_API_KEYS`**: recommended on the **api** in any exposed deployment.
- **`RAG_INDEX_FAKE_EMBEDDINGS=1`**: optional on **worker** for offline/tests (no OpenAI for embeddings).

**Local dev without Postgres:** unset **`DATABASE_URL`** so the app uses **SQLite** registry + per-document index files under **`storage/`** (default).

**Local dev without Redis:** unset **`REDIS_URL`** so the pipeline runs in **`BackgroundTasks`** in the API process, or run Redis and **`python -m rag_worker.worker`** with the same **`RAG_STORAGE_ROOT`** / **`DATABASE_URL`**.

### Ingest (recommended for Phase 1)

Copies the PDF into content-addressed storage, runs extraction, updates SQLite. Safe to run twice on the same file (skips if already ingested at the same extraction version).

```powershell
python -m rag_extractor ingest path\to\your\guide.pdf
```

- **Force re-extract** (same bytes, same version string in DB — e.g. after you change extractor code and bump `EXTRACTION_VERSION` in `rag_extractor/__init__.py`):

```powershell
python -m rag_extractor ingest path\to\your\guide.pdf --force
```

- **Custom storage root** (for CI or isolated runs):

```powershell
python -m rag_extractor ingest path\to\your\guide.pdf --storage-root D:\ragdata
```

Output is JSON on stdout (paths, `content_sha256`, `skipped`, `status`). Exit code `1` if extraction failed (row in registry will be `failed` with `error_message`).

### List and show registry records

```powershell
python -m rag_extractor list
python -m rag_extractor list --json
python -m rag_extractor show <64-char-sha256>
python -m rag_extractor show <unique-prefix>
python -m rag_extractor show <prefix> --json
```

### Extract only (no registry, no copy)

Useful for one-off JSON without touching the registry:

```powershell
python -m rag_extractor extract path\to\your\guide.pdf
```

- **Default output:** `<storage>/artifacts/<stem>_extraction.json` (under `RAG_STORAGE_ROOT` if set)
- **Custom output:**

```powershell
python -m rag_extractor extract path\to\your\guide.pdf -o path\to\artifact.json
```

- **Override `document_id` in the JSON** (default when not using `ingest`: first 16 hex chars of SHA-256; **`ingest` uses the full SHA-256**):

```powershell
python -m rag_extractor extract path\to\your\guide.pdf --document-id my-doc-001
```

### Chunk (Phase 2)

From an **ingested** document (registry row `ready` with `artifact_relpath`):

```powershell
python -m rag_chunker run --sha256 <64-char-or-unique-prefix> --write-registry
```

Default outputs: `storage/documents/<sha256>/chunks.jsonl` and `chunks_manifest.json` next to it.

**From an extraction file directly** (no registry update unless the document is ingested and `--write-registry` is set):

```powershell
python -m rag_chunker run --artifact path\to\extraction.json
```

**Options:** `--max-chars` (default 2500), `--overlap` (default `0.2`), `-o` / `--out` for JSONL path, `--manifest` for manifest path, `--force` to re-chunk when `chunker_version` matches, `--storage-root`.

**Shortcut:**

```powershell
python -m rag_extractor chunk --sha256 <prefix> --write-registry
```

(`chunk` forwards to `rag_chunker run …`.)

### Embed and index (Phase 3)

**Build** (after `chunks.jsonl` exists — typically from chunk step):

```powershell
$env:OPENAI_API_KEY="sk-..."
python -m rag_index build --sha256 <prefix> --write-registry --force
```

- Default index path: `storage/index/<sha256>.sqlite`.
- **Idempotency:** if the registry already points at an index whose `indexer_version` matches the code and you did not pass **`--force`**, the build is **skipped**.
- **Without OpenAI** (tests): `$env:RAG_INDEX_FAKE_EMBEDDINGS="1"` before `build` and `search`.

**Search** (hybrid RRF):

```powershell
python -m rag_index search --sha256 <prefix> "battery wand clearance"
```

Or pass **`--index path\to\index.sqlite`** and a query string.

**Shortcut:**

```powershell
python -m rag_extractor index build --sha256 <prefix> --write-registry
python -m rag_extractor index search --sha256 <prefix> "your question"
```

### Retrieve + rerank (Phase 4)

Use when you want a **larger hybrid pool** before cutting to final results (recommended for production RAG).

**RRF only** (no Cohere; same ordering as Phase 3 search but with `--candidates` > `--top`):

```powershell
python -m rag_retrieve --sha256 <prefix> "battery wand" --candidates 40 --top 8 --rerank none
```

**Cohere rerank** (set API key):

```powershell
$env:COHERE_API_KEY="..."
python -m rag_retrieve --sha256 <prefix> "battery wand" --candidates 40 --top 8 --rerank cohere --json
```

**Shortcut:**

```powershell
python -m rag_extractor retrieve --sha256 <prefix> "your question" --rerank cohere --top 8
```

**Cross-encoder** (downloads model on first run; CPU OK for small pools):

```powershell
python -m rag_retrieve --sha256 <prefix> "question" --rerank cross-encoder --candidates 30 --top 8
```

### Eval runner (`rag_eval`)

Create a **JSONL** file: one JSON object per line (`EvalCase`). Optional gold signals (any match = relevant):

- `gold_chunk_ids`: list of exact `chunk_id` strings from `chunks.jsonl`
- `gold_pages`: list of page numbers that should fall inside a hit’s `page_start`–`page_end`
- `gold_substrings`: each must appear in `text_full` (case-insensitive)

Lines whose first non-whitespace character is `#` are skipped (comments).

```powershell
python -m rag_eval run --cases fixtures/eval/example_eval.jsonl --sha256 <prefix> --rerank none
```

Optional: `--per-case out.jsonl` to dump per-question ranks and retrieved ids.

**Shortcut:** `python -m rag_extractor eval run --cases ... --sha256 ...`

### Grounded answers (`rag_generate`)

Requires **`OPENAI_API_KEY`**. Uses the same retrieval path as **`rag_retrieve`** (including **`--rerank`**).

```powershell
$env:OPENAI_API_KEY="sk-..."
python -m rag_generate ask --sha256 <prefix> -q "What changed in the March 2026 revision?" --rerank none --top 8
```

**Shortcut:** `python -m rag_extractor ask --sha256 <prefix> -q "..." --json`

### Run tests

```powershell
python -m unittest discover -s tests -v
```

The large-PDF smoke test is skipped if `fixtures/pdfs/MPM_PS_US_MAR2026_02252026.pdf` is missing.

---

## Dependencies (current)

Listed in `requirements.txt`:

- **pymupdf** — PDF text, layout blocks, table detection
- **pydantic** — Validated schemas (extraction + chunks)
- **numpy** — Vector math for search
- **openai** — Embedding API client
- **cohere** — Rerank API client (Phase 4)
- **sentence-transformers** — Cross-encoder rerank (Phase 4); pulls **PyTorch** transitively
- **fastapi** — HTTP API framework
- **uvicorn** — ASGI server for **`python -m rag_api`**
- **httpx** — Used by FastAPI’s test client (and compatible HTTP calls)
- **slowapi** — Rate limiting for **`/v1/*`**
- **python-multipart** — Multipart uploads for **`POST /v1/ingest`**
- **redis** — Broker for **RQ** when **`REDIS_URL`** is set
- **rq** — Job queue; worker runs **`rag_worker.tasks.process_document_job`**
- **psycopg** — PostgreSQL driver when **`DATABASE_URL`** is set
- **pgvector** — Vector type registration for **psycopg**
- **boto3** — S3 uploads when **`RAG_S3_BUCKET`** is set

---

## Artifact schema (Phase 1)

The JSON artifact includes `schema_version`, `extraction_version`, `document_id`, `source_sha256`, PDF metadata, and a `pages[]` array with **blocks** of type `text`, `table`, or `image`. Tables include `rows` and `markdown` derived from extracted cells.

## Chunk records (Phase 2)

Each line in `chunks.jsonl` is a **`ChunkRecord`** (`schema_version: chunk.v1`): `chunk_id`, `chunker_version`, `document_id`, `source_sha256`, `source_filename`, `extraction_version`, `chunk_index`, `page_start` / `page_end`, `section_path`, `block_ids`, `content_type` (`prose` | `table`), `text_full`, `text_embed` (breadcrumb prefix + body), and character offsets for debugging.

Indexing and retrieval env vars are listed under **Environment** above.

---

## Future setup (placeholders — fill in as you build)

When the following exist, document them here in the same PR that adds them:

- **Web app** (e.g. Next.js on Vercel): install (`npm`/`pnpm`), `env` vars, `dev` / `build` / `start`; call **`POST /v1/ask`** or **`/v1/retrieve`** on this repo’s API (or a gateway in front of it)
- **Database / vector store** (e.g. Postgres + pgvector, Pinecone, etc.): connection strings, migrations, indexes
- **Keyword search** (same DB `tsvector`, Meilisearch, etc.): URLs, API keys, index names
- **Workers** (Inngest, Trigger.dev, Celery, etc.): how jobs are triggered and secrets configured
- **LLM / embedding APIs**: provider keys, model names, rate limits
- **Object storage** (S3, R2, Vercel Blob): buckets, CORS, upload limits

---

## Troubleshooting

- **Extraction is slow on large PDFs:** Table detection runs per page; use a **background job** in production and show progress in the UI.
- **`storage/artifacts/` missing:** It is created automatically on first successful write.
- **Import errors:** Ensure the project root is the current working directory and the venv has `requirements.txt` installed.

---

## License

Add a license file when you choose one; until then, assume all rights reserved unless you state otherwise.
