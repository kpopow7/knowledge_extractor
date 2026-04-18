from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from rag_extractor.paths import registry_db_path
from rag_extractor.registry_models import DocumentRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    content_sha256 TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
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
);

CREATE INDEX IF NOT EXISTS documents_status ON documents(status);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "chunks_relpath" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN chunks_relpath TEXT")
    if "chunker_version" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN chunker_version TEXT")
    if "index_db_relpath" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN index_db_relpath TEXT")
    if "embedding_model" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT")
    if "embedding_dimensions" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN embedding_dimensions INTEGER")


class DocumentRegistrySqlite:
    """SQLite-backed registry for idempotent ingest (Phase 1)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or registry_db_path()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(SCHEMA)
            _migrate(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, content_sha256: str) -> DocumentRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_sha256 = ?",
                (content_sha256,),
            ).fetchone()
        return None if row is None else self._row_to_record(row)

    def upsert_pending(
        self,
        *,
        content_sha256: str,
        original_filename: str,
        byte_size: int,
        source_relpath: str,
    ) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT content_sha256 FROM documents WHERE content_sha256 = ?",
                (content_sha256,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO documents (
                        content_sha256, original_filename, byte_size, status,
                        extraction_version, extractor_package_version, schema_version,
                        page_count, source_relpath, artifact_relpath, chunks_relpath,
                        chunker_version, index_db_relpath, embedding_model, embedding_dimensions,
                        error_message,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?)
                    """,
                    (content_sha256, original_filename, byte_size, source_relpath, now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE documents SET
                        original_filename = ?, byte_size = ?, status = 'pending',
                        source_relpath = ?, artifact_relpath = NULL,
                        chunks_relpath = NULL, chunker_version = NULL,
                        index_db_relpath = NULL, embedding_model = NULL, embedding_dimensions = NULL,
                        error_message = NULL,
                        extraction_version = NULL, extractor_package_version = NULL,
                        schema_version = NULL, page_count = NULL,
                        updated_at = ?
                    WHERE content_sha256 = ?
                    """,
                    (original_filename, byte_size, source_relpath, now, content_sha256),
                )

    def mark_ready(
        self,
        *,
        content_sha256: str,
        extraction_version: str,
        extractor_package_version: str,
        schema_version: str,
        page_count: int,
        artifact_relpath: str,
    ) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE documents SET
                    status = 'ready',
                    extraction_version = ?,
                    extractor_package_version = ?,
                    schema_version = ?,
                    page_count = ?,
                    artifact_relpath = ?,
                    error_message = NULL,
                    updated_at = ?
                WHERE content_sha256 = ?
                """,
                (
                    extraction_version,
                    extractor_package_version,
                    schema_version,
                    page_count,
                    artifact_relpath,
                    now,
                    content_sha256,
                ),
            )

    def mark_failed(self, *, content_sha256: str, error_message: str) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE documents SET
                    status = 'failed',
                    error_message = ?,
                    artifact_relpath = NULL,
                    updated_at = ?
                WHERE content_sha256 = ?
                """,
                (error_message[:8000], now, content_sha256),
            )

    def mark_chunks(
        self,
        *,
        content_sha256: str,
        chunks_relpath: str,
        chunker_version: str,
    ) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE documents SET
                    chunks_relpath = ?,
                    chunker_version = ?,
                    updated_at = ?
                WHERE content_sha256 = ?
                """,
                (chunks_relpath, chunker_version, now, content_sha256),
            )

    def mark_index(
        self,
        *,
        content_sha256: str,
        index_db_relpath: str,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE documents SET
                    index_db_relpath = ?,
                    embedding_model = ?,
                    embedding_dimensions = ?,
                    updated_at = ?
                WHERE content_sha256 = ?
                """,
                (index_db_relpath, embedding_model, embedding_dimensions, now, content_sha256),
            )

    def list_recent(self, limit: int = 50) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        d = dict(row)
        return DocumentRecord(
            content_sha256=d["content_sha256"],
            original_filename=d["original_filename"],
            byte_size=d["byte_size"],
            status=d["status"],
            extraction_version=d.get("extraction_version"),
            extractor_package_version=d.get("extractor_package_version"),
            schema_version=d.get("schema_version"),
            page_count=d.get("page_count"),
            source_relpath=d["source_relpath"],
            artifact_relpath=d.get("artifact_relpath"),
            chunks_relpath=d.get("chunks_relpath"),
            chunker_version=d.get("chunker_version"),
            index_db_relpath=d.get("index_db_relpath"),
            embedding_model=d.get("embedding_model"),
            embedding_dimensions=d.get("embedding_dimensions"),
            error_message=d.get("error_message"),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )
