from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware

from rag_api.auth import require_api_key
from rag_api import job_store
from rag_api.logging_config import configure_logging
from rag_api.registry_resolve import resolve_index_db
from rag_api.request_context import RequestIdMiddleware
from rag_extractor.paths import storage_root
from rag_generate.answer import answer_with_retrieval
from rag_index.search import SearchHit
from rag_retrieve.pipeline import retrieve
from rag_worker.pipeline import run_document_pipeline
from rag_worker.queue import enqueue_document_job, use_redis_queue

log = logging.getLogger(__name__)

_rate_limit = os.environ.get("RAG_API_RATE_LIMIT", "120/minute")
_rate_limit_ingest = os.environ.get("RAG_API_RATE_LIMIT_INGEST", "30/minute")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("RAG API starting (rate_limit=%s, ingest_limit=%s)", _rate_limit, _rate_limit_ingest)
    yield
    log.info("RAG API shutdown")


app = FastAPI(
    title="Custom knowledge RAG API",
    description="HTTP wrapper around hybrid retrieve and grounded ask (same behavior as CLI).",
    version="0.3.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_origins = (os.environ.get("RAG_API_CORS_ORIGINS") or "").strip()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(RequestIdMiddleware)


def _hits_to_rows(hits: list[SearchHit]) -> list[dict]:
    rows: list[dict] = []
    for h in hits:
        row = {"chunk_id": h.chunk_id, "rrf_or_rerank_score": h.rrf_score, **h.payload}
        rows.append(row)
    return rows


class RetrieveBody(BaseModel):
    """Hybrid retrieval + optional rerank (see ``rag_retrieve`` CLI)."""

    sha256: str | None = Field(
        default=None,
        description="Document id: full 64-char SHA-256 or unique registry prefix.",
    )
    index_db: str | None = Field(
        default=None,
        description="Explicit path to index SQLite (alternative to sha256).",
    )
    query: str = Field(..., min_length=1)
    top: int = Field(default=10, ge=1, le=200)
    candidates: int = Field(default=40, ge=1, le=500)
    rerank: str = Field(default="none", description="none | cohere | cross-encoder")


class RetrieveResponse(BaseModel):
    hits: list[dict]


class AskBody(BaseModel):
    """Retrieve context, then OpenAI chat completion (requires OPENAI_API_KEY)."""

    sha256: str | None = None
    index_db: str | None = None
    question: str = Field(..., min_length=1)
    model: str | None = Field(default=None, description="Chat model override (else OPENAI_CHAT_MODEL / gpt-4o-mini).")
    top: int = Field(default=8, ge=1, le=200)
    candidates: int = Field(default=40, ge=1, le=500)
    rerank: str = Field(default="none")


class AskResponse(BaseModel):
    answer: str
    chunk_ids: list[str]
    pages: list[list[int | None]]


class IngestAcceptedResponse(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    original_filename: str | None = None
    content_sha256: str | None = None
    skipped: bool | None = None
    reason: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/retrieve", response_model=RetrieveResponse)
@limiter.limit(_rate_limit)
def post_retrieve(
    request: Request,
    body: RetrieveBody,
    _: None = Depends(require_api_key),
) -> RetrieveResponse:
    try:
        idx = resolve_index_db(sha256=body.sha256, index_db=body.index_db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    pool = max(body.candidates, body.top)
    try:
        hits = retrieve(
            idx,
            body.query,
            final_k=body.top,
            candidate_pool=pool,
            reranker=body.rerank,
        )
    except Exception as e:
        log.exception("retrieve failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return RetrieveResponse(hits=_hits_to_rows(hits))


@app.post("/v1/ask", response_model=AskResponse)
@limiter.limit(_rate_limit)
def post_ask(
    request: Request,
    body: AskBody,
    _: None = Depends(require_api_key),
) -> AskResponse:
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set; generation is unavailable.",
        )
    try:
        idx = resolve_index_db(sha256=body.sha256, index_db=body.index_db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    pool = max(body.candidates, body.top)
    try:
        answer, hits = answer_with_retrieval(
            idx,
            body.question,
            chat_model=body.model,
            final_k=body.top,
            candidate_pool=pool,
            reranker=body.rerank,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    pages: list[list[int | None]] = [
        [h.payload.get("page_start"), h.payload.get("page_end")] for h in hits
    ]
    return AskResponse(
        answer=answer,
        chunk_ids=[h.chunk_id for h in hits],
        pages=pages,
    )


@app.post("/v1/ingest", response_model=IngestAcceptedResponse, status_code=202)
@limiter.limit(_rate_limit_ingest)
async def post_ingest(
    request: Request,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    file: UploadFile = File(..., description="PDF bytes (must start with %PDF)"),
    force: bool = False,
) -> IngestAcceptedResponse:
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    if suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Filename must end with .pdf")

    job_id = str(uuid.uuid4())
    incoming = storage_root() / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    dest = incoming / f"{job_id}.pdf"
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if not dest.read_bytes()[:5].startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="Not a PDF: file must start with %PDF header.",
            )
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    job_store.create_job(job_id, file.filename or "upload.pdf")
    try:
        if use_redis_queue():
            enqueue_document_job(job_id, dest, force=force)
            out_status = "queued"
        else:
            background_tasks.add_task(run_document_pipeline, job_id, dest, force=force)
            out_status = "pending"
    except Exception as e:
        log.exception("failed to queue pipeline for job %s", job_id)
        dest.unlink(missing_ok=True)
        job_store.set_status(job_id, "failed", error_message=f"queue_error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Failed to start pipeline (Redis down or misconfigured?).",
        ) from e

    log.info("accepted pipeline job %s file=%s redis=%s", job_id, file.filename, use_redis_queue())
    return IngestAcceptedResponse(job_id=job_id, status=out_status)


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
@limiter.limit(_rate_limit)
def get_job_status(
    request: Request,
    job_id: str,
    _: None = Depends(require_api_key),
) -> JobStatusResponse:
    rec = job_store.get_job(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return JobStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        original_filename=rec.original_filename,
        content_sha256=rec.content_sha256,
        skipped=rec.skipped,
        reason=rec.reason,
        error_message=rec.error_message,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )
