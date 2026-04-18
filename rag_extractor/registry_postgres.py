from __future__ import annotations

from datetime import datetime, timezone

from psycopg.rows import dict_row

from rag_extractor.registry_models import DocumentRecord
from rag_storage.pg import connect, init_schema


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _row_to_record(d: dict) -> DocumentRecord:
    return DocumentRecord(
        content_sha256=d["content_sha256"],
        original_filename=d["original_filename"],
        byte_size=int(d["byte_size"]),
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
        created_at=str(d["created_at"]),
        updated_at=str(d["updated_at"]),
    )


class DocumentRegistryPostgres:
    """Postgres-backed document registry (same fields as SQLite)."""

    def __init__(self) -> None:
        init_schema()

    def get(self, content_sha256: str) -> DocumentRecord | None:
        with connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM documents WHERE content_sha256 = %s",
                    (content_sha256,),
                )
                row = cur.fetchone()
        return None if row is None else _row_to_record(dict(row))

    def upsert_pending(
        self,
        *,
        content_sha256: str,
        original_filename: str,
        byte_size: int,
        source_relpath: str,
    ) -> None:
        now = _utc_now_iso()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content_sha256 FROM documents WHERE content_sha256 = %s",
                    (content_sha256,),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        """
                        INSERT INTO documents (
                            content_sha256, original_filename, byte_size, status,
                            extraction_version, extractor_package_version, schema_version,
                            page_count, source_relpath, artifact_relpath, chunks_relpath,
                            chunker_version, index_db_relpath, embedding_model, embedding_dimensions,
                            error_message,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, 'pending', NULL, NULL, NULL, NULL, %s, NULL, NULL, NULL, NULL, NULL, NULL, NULL, %s, %s)
                        """,
                        (content_sha256, original_filename, byte_size, source_relpath, now, now),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE documents SET
                            original_filename = %s, byte_size = %s, status = 'pending',
                            source_relpath = %s, artifact_relpath = NULL,
                            chunks_relpath = NULL, chunker_version = NULL,
                            index_db_relpath = NULL, embedding_model = NULL, embedding_dimensions = NULL,
                            error_message = NULL,
                            extraction_version = NULL, extractor_package_version = NULL,
                            schema_version = NULL, page_count = NULL,
                            updated_at = %s
                        WHERE content_sha256 = %s
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
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents SET
                        status = 'ready',
                        extraction_version = %s,
                        extractor_package_version = %s,
                        schema_version = %s,
                        page_count = %s,
                        artifact_relpath = %s,
                        error_message = NULL,
                        updated_at = %s
                    WHERE content_sha256 = %s
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
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents SET
                        status = 'failed',
                        error_message = %s,
                        artifact_relpath = NULL,
                        updated_at = %s
                    WHERE content_sha256 = %s
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
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents SET
                        chunks_relpath = %s,
                        chunker_version = %s,
                        updated_at = %s
                    WHERE content_sha256 = %s
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
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents SET
                        index_db_relpath = %s,
                        embedding_model = %s,
                        embedding_dimensions = %s,
                        updated_at = %s
                    WHERE content_sha256 = %s
                    """,
                    (index_db_relpath, embedding_model, embedding_dimensions, now, content_sha256),
                )

    def list_recent(self, limit: int = 50) -> list[DocumentRecord]:
        with connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM documents ORDER BY updated_at DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        return [_row_to_record(dict(r)) for r in rows]
