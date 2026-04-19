from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg

from rag_storage.config import database_url

log = logging.getLogger(__name__)

# text-embedding-3-small default dimensions; override if you use another fixed size
PG_VECTOR_DIM = int(os.environ.get("RAG_PG_VECTOR_DIM", "1536"))


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg.connect(url)
    try:
        from pgvector.psycopg import register_vector

        register_vector(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    """Create extensions and tables (idempotent)."""
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
              content_sha256 VARCHAR(64) PRIMARY KEY,
              original_filename TEXT NOT NULL,
              byte_size BIGINT NOT NULL,
              status TEXT NOT NULL,
              extraction_version TEXT,
              extractor_package_version TEXT,
              schema_version TEXT,
              page_count INTEGER,
              source_relpath TEXT NOT NULL,
              artifact_relpath TEXT,
              chunks_relpath TEXT,
              chunker_version TEXT,
              index_db_relpath TEXT,
              embedding_model TEXT,
              embedding_dimensions INTEGER,
              error_message TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_meta_kv (
              source_sha256 VARCHAR(64) NOT NULL,
              k TEXT NOT NULL,
              v TEXT NOT NULL,
              PRIMARY KEY (source_sha256, k)
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS index_chunks (
              source_sha256 VARCHAR(64) NOT NULL,
              chunk_id TEXT NOT NULL,
              document_id TEXT NOT NULL,
              payload_json JSONB NOT NULL,
              dims INTEGER NOT NULL,
              embedding vector({PG_VECTOR_DIM}),
              fts_body TEXT NOT NULL,
              PRIMARY KEY (source_sha256, chunk_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_index_chunks_sha ON index_chunks(source_sha256)"
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_index_chunks_fts
            ON index_chunks USING GIN (to_tsvector('english', fts_body))
            """
        )
        _maybe_create_hnsw_index(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_usage (
              tenant_id TEXT NOT NULL,
              day_utc TEXT NOT NULL,
              asks INTEGER NOT NULL DEFAULT 0,
              retrieves INTEGER NOT NULL DEFAULT 0,
              ingests INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (tenant_id, day_utc)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
              cache_key VARCHAR(64) PRIMARY KEY,
              model TEXT NOT NULL,
              dims INTEGER NOT NULL,
              vec_json JSONB NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_jobs (
              job_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              original_filename TEXT,
              content_sha256 TEXT,
              skipped INTEGER,
              reason TEXT,
              error_message TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS tenant_id TEXT")


def _maybe_create_hnsw_index(conn: psycopg.Connection) -> None:
    """ANN index for scale; disable with RAG_PG_DISABLE_HNSW=1 if unsupported or for tests."""
    v = (os.environ.get("RAG_PG_DISABLE_HNSW") or "").strip().lower()
    if v in ("1", "true", "yes"):
        return
    try:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_index_chunks_embedding_hnsw
            ON index_chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
    except psycopg.Error as e:
        log.warning(
            "Could not create HNSW index on index_chunks.embedding (set RAG_PG_DISABLE_HNSW=1 to skip): %s",
            e,
        )
