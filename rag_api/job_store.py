from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag_storage.config import use_postgres


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def jobs_db_path() -> Path:
    from rag_extractor.paths import storage_root

    return storage_root() / "api_jobs.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  original_filename TEXT,
  content_sha256 TEXT,
  skipped INTEGER DEFAULT 0,
  reason TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


@dataclass
class IngestJobRecord:
    job_id: str
    status: str
    original_filename: str | None
    content_sha256: str | None
    skipped: bool | None
    reason: str | None
    error_message: str | None
    created_at: str
    updated_at: str


@contextmanager
def _connect() -> sqlite3.Connection:
    jobs_db_path().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(jobs_db_path()), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(ingest_jobs)").fetchall()
    cols = {r[1] for r in rows}
    if not cols:
        return
    conn.execute(
        "UPDATE ingest_jobs SET status = 'ingesting' WHERE status = 'running'",
    )


def create_job(job_id: str, original_filename: str) -> None:
    if use_postgres():
        from rag_api import job_store_pg

        job_store_pg.create_job(job_id, original_filename)
        return
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ingest_jobs (
              job_id, status, original_filename, content_sha256, skipped, reason, error_message, created_at, updated_at
            ) VALUES (?, 'pending', ?, NULL, NULL, NULL, NULL, ?, ?)
            """,
            (job_id, original_filename, now, now),
        )


def set_status(
    job_id: str,
    status: str,
    *,
    content_sha256: str | None = None,
    skipped: bool | None = None,
    reason: str | None = None,
    error_message: str | None = None,
    clear_error: bool = False,
) -> None:
    """
    Update job row. Pass ``clear_error=True`` with ``ready`` to null out ``error_message``.
    Use ``content_sha256=None`` to leave the column unchanged (omit update) — implemented below.
    """
    if use_postgres():
        from rag_api import job_store_pg

        job_store_pg.set_status(
            job_id,
            status,
            content_sha256=content_sha256,
            skipped=skipped,
            reason=reason,
            error_message=error_message,
            clear_error=clear_error,
        )
        return
    now = _utc_now()
    fields: list[str] = ["status = ?", "updated_at = ?"]
    params: list = [status, now]

    if content_sha256 is not None:
        fields.append("content_sha256 = ?")
        params.append(content_sha256)
    if skipped is not None:
        fields.append("skipped = ?")
        params.append(1 if skipped else 0)
    if reason is not None:
        fields.append("reason = ?")
        params.append(reason)
    if error_message is not None:
        fields.append("error_message = ?")
        params.append(error_message[:8000])
    elif clear_error:
        fields.append("error_message = NULL")

    params.append(job_id)
    sql = f"UPDATE ingest_jobs SET {', '.join(fields)} WHERE job_id = ?"
    with _connect() as conn:
        conn.execute(sql, params)


def get_job(job_id: str) -> IngestJobRecord | None:
    if use_postgres():
        from rag_api import job_store_pg

        return job_store_pg.get_job(job_id)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    skipped = row["skipped"]
    return IngestJobRecord(
        job_id=row["job_id"],
        status=row["status"],
        original_filename=row["original_filename"],
        content_sha256=row["content_sha256"],
        skipped=None if skipped is None else bool(skipped),
        reason=row["reason"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_jobs(limit: int = 100, offset: int = 0) -> list[IngestJobRecord]:
    if use_postgres():
        from rag_api import job_store_pg

        return job_store_pg.list_jobs(limit=limit, offset=offset)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ingest_jobs ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    out: list[IngestJobRecord] = []
    for row in rows:
        skipped = row["skipped"]
        out.append(
            IngestJobRecord(
                job_id=row["job_id"],
                status=row["status"],
                original_filename=row["original_filename"],
                content_sha256=row["content_sha256"],
                skipped=None if skipped is None else bool(skipped),
                reason=row["reason"],
                error_message=row["error_message"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )
    return out
