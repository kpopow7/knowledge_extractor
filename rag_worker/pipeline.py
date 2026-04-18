from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from rag_api import job_store
from rag_chunker import CHUNKER_VERSION, DEFAULT_MAX_CHUNK_CHARS, DEFAULT_OVERLAP_RATIO
from rag_chunker.chunker import chunk_from_path, write_jsonl, write_manifest
from rag_extractor.ingest import ingest_pdf
from rag_extractor.paths import index_dir, storage_root
from rag_extractor.registry import DocumentRegistry
from rag_index import INDEXER_VERSION
from rag_index.build import build_index
from rag_index.store import ChunkIndex
from rag_index.store_pg import ChunkIndexPostgres
from rag_storage.config import use_postgres

log = logging.getLogger(__name__)


def run_chunk_for_sha256(sha: str, *, force: bool) -> dict[str, Any]:
    """Emit ``chunks.jsonl`` from the registry extraction artifact; update registry."""
    reg = DocumentRegistry()
    rec = reg.get(sha)
    if rec is None:
        raise RuntimeError("Document not in registry.")
    if rec.status != "ready" or not rec.artifact_relpath:
        raise RuntimeError("Document has no ready extraction artifact.")

    if (
        rec.chunks_relpath
        and rec.chunker_version == CHUNKER_VERSION
        and not force
    ):
        return {
            "skipped": True,
            "reason": "already_chunked_same_version",
            "chunks_relpath": rec.chunks_relpath,
        }

    artifact_path = storage_root() / rec.artifact_relpath
    chunks, art = chunk_from_path(
        artifact_path,
        max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS,
        overlap_ratio=DEFAULT_OVERLAP_RATIO,
    )
    jsonl_path = storage_root() / "documents" / sha / "chunks.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(chunks, jsonl_path)
    man_path = jsonl_path.with_name(jsonl_path.stem + "_manifest.json")
    write_manifest(
        chunks,
        art,
        man_path,
        max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS,
        overlap_ratio=DEFAULT_OVERLAP_RATIO,
    )
    rel = jsonl_path.resolve().relative_to(storage_root().resolve()).as_posix()
    reg.mark_chunks(content_sha256=sha, chunks_relpath=rel, chunker_version=CHUNKER_VERSION)
    return {
        "skipped": False,
        "chunk_count": len(chunks),
        "chunks_relpath": rel,
    }


def run_index_for_sha256(
    sha: str,
    *,
    embedding_model: str | None,
    force: bool,
) -> dict[str, Any]:
    """Embed chunks into per-document SQLite + FTS or Postgres + pgvector; update registry."""
    reg = DocumentRegistry()
    rec = reg.get(sha)
    if rec is None:
        raise RuntimeError("Document not in registry.")
    if not rec.chunks_relpath:
        raise RuntimeError("No chunks.jsonl — chunk step must succeed first.")

    chunks_path = storage_root() / rec.chunks_relpath
    index_dir().mkdir(parents=True, exist_ok=True)
    out_db = index_dir() / f"{sha}.sqlite"

    if not force and rec.index_db_relpath:
        if use_postgres() and rec.index_db_relpath == "postgres":
            iv = ChunkIndexPostgres(sha).get_meta("indexer_version")
            if iv == INDEXER_VERSION:
                return {
                    "skipped": True,
                    "reason": "index_present_same_indexer_version",
                    "index_path": f"postgres:{sha}",
                }
        else:
            idxp = storage_root() / rec.index_db_relpath
            if idxp.is_file():
                iv = ChunkIndex(idxp).get_meta("indexer_version")
                if iv == INDEXER_VERSION:
                    return {
                        "skipped": True,
                        "reason": "index_present_same_indexer_version",
                        "index_path": str(idxp),
                    }

    if use_postgres():
        n, model, dims = build_index(
            chunks_path,
            None,
            embedding_model=embedding_model,
            clear=True,
            postgres_source_sha256=sha,
        )
        rel = "postgres"
    else:
        n, model, dims = build_index(
            chunks_path,
            out_db,
            embedding_model=embedding_model,
            clear=True,
        )
        rel = out_db.resolve().relative_to(storage_root().resolve()).as_posix()

    reg.mark_index(
        content_sha256=sha,
        index_db_relpath=rel,
        embedding_model=model,
        embedding_dimensions=dims,
    )
    return {
        "skipped": False,
        "chunk_count": n,
        "embedding_model": model,
        "dimensions": dims,
        "index_relpath": rel,
    }


def run_document_pipeline(job_id: str, temp_pdf: Path, *, force: bool) -> None:
    """
    Ingest PDF (copy + extract), then chunk, then embed/index. Updates ``job_store`` status.
    Deletes ``temp_pdf`` after a successful ingest copy (or on ingest failure).
    """
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL")

    try:
        job_store.set_status(job_id, "ingesting")
        result = ingest_pdf(temp_pdf, force=force)
        try:
            temp_pdf.unlink(missing_ok=True)
        except OSError:
            pass

        if result.status == "failed":
            from rag_extractor.registry import DocumentRegistry

            rec = DocumentRegistry().get(result.content_sha256)
            err = (rec.error_message if rec else None) or "ingest failed"
            job_store.set_status(
                job_id,
                "failed",
                content_sha256=result.content_sha256,
                skipped=result.skipped,
                reason=result.reason,
                error_message=err,
            )
            return

        sha = result.content_sha256

        job_store.set_status(
            job_id,
            "chunking",
            content_sha256=sha,
            skipped=result.skipped,
            reason=result.reason,
        )
        run_chunk_for_sha256(sha, force=force)

        job_store.set_status(job_id, "indexing", content_sha256=sha)
        run_index_for_sha256(sha, embedding_model=embedding_model, force=force)

        job_store.set_status(job_id, "ready", content_sha256=sha, clear_error=True)
        log.info("pipeline job %s ready sha256=%s", job_id, sha)
    except Exception as e:  # noqa: BLE001
        log.exception("pipeline job %s failed", job_id)
        job_store.set_status(job_id, "failed", error_message=f"{type(e).__name__}: {e}")


def run_reprocess_for_sha256(content_sha256: str, *, force: bool) -> dict[str, Any]:
    """
    Re-run extraction from the stored ``source.pdf``, then chunk and index (admin maintenance).
    Requires the document registry row and on-disk source at ``source_relpath``.
    """
    reg = DocumentRegistry()
    rec = reg.get(content_sha256)
    if rec is None:
        raise RuntimeError("Document not in registry.")
    src = storage_root() / rec.source_relpath
    if not src.is_file():
        raise RuntimeError(f"Source PDF missing at {rec.source_relpath}")

    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL")
    result = ingest_pdf(src, force=force)
    if result.status == "failed":
        raise RuntimeError("Re-ingest (extract) failed; check registry error_message.")

    sha = result.content_sha256
    chunk_out = run_chunk_for_sha256(sha, force=force)
    index_out = run_index_for_sha256(sha, embedding_model=embedding_model, force=force)
    return {
        "content_sha256": sha,
        "ingest": {"skipped": result.skipped, "reason": result.reason},
        "chunk": chunk_out,
        "index": index_out,
    }
