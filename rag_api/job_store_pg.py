from __future__ import annotations

from datetime import datetime, timezone

from psycopg.rows import dict_row

from rag_storage.pg import connect, init_schema


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def create_job(job_id: str, original_filename: str) -> None:
    init_schema()
    now = _utc_now()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingest_jobs (
                  job_id, status, original_filename, content_sha256, skipped, reason, error_message, created_at, updated_at
                ) VALUES (%s, 'pending', %s, NULL, NULL, NULL, NULL, %s, %s)
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
    now = _utc_now()
    fields: list[str] = ["status = %s", "updated_at = %s"]
    params: list = [status, now]

    if content_sha256 is not None:
        fields.append("content_sha256 = %s")
        params.append(content_sha256)
    if skipped is not None:
        fields.append("skipped = %s")
        params.append(1 if skipped else 0)
    if reason is not None:
        fields.append("reason = %s")
        params.append(reason)
    if error_message is not None:
        fields.append("error_message = %s")
        params.append(error_message[:8000])
    elif clear_error:
        fields.append("error_message = NULL")

    params.append(job_id)
    sql = f"UPDATE ingest_jobs SET {', '.join(fields)} WHERE job_id = %s"
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def get_job(job_id: str):
    from rag_api.job_store import IngestJobRecord

    init_schema()
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM ingest_jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    skipped = d.get("skipped")
    return IngestJobRecord(
        job_id=d["job_id"],
        status=d["status"],
        original_filename=d.get("original_filename"),
        content_sha256=d.get("content_sha256"),
        skipped=None if skipped is None else bool(skipped),
        reason=d.get("reason"),
        error_message=d.get("error_message"),
        created_at=str(d["created_at"]),
        updated_at=str(d["updated_at"]),
    )


def list_jobs(limit: int = 100, offset: int = 0):
    from rag_api.job_store import IngestJobRecord

    init_schema()
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM ingest_jobs ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            rows = cur.fetchall()
    out: list[IngestJobRecord] = []
    for r in rows:
        d = dict(r)
        skipped = d.get("skipped")
        out.append(
            IngestJobRecord(
                job_id=d["job_id"],
                status=d["status"],
                original_filename=d.get("original_filename"),
                content_sha256=d.get("content_sha256"),
                skipped=None if skipped is None else bool(skipped),
                reason=d.get("reason"),
                error_message=d.get("error_message"),
                created_at=str(d["created_at"]),
                updated_at=str(d["updated_at"]),
            )
        )
    return out
