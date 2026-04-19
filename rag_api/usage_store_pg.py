from __future__ import annotations

from datetime import datetime, timezone

from psycopg.rows import dict_row

from rag_storage.pg import connect, init_schema


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def get_counts(tenant_id: str, day: str | None = None) -> tuple[int, int, int]:
    d = day or _today()
    init_schema()
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT asks, retrieves, ingests FROM tenant_usage
                WHERE tenant_id = %s AND day_utc = %s
                """,
                (tenant_id, d),
            )
            row = cur.fetchone()
    if row is None:
        return (0, 0, 0)
    return int(row["asks"]), int(row["retrieves"]), int(row["ingests"])


def increment(tenant_id: str, field: str) -> None:
    if field not in ("asks", "retrieves", "ingests"):
        raise ValueError("field must be asks, retrieves, or ingests")
    d = _today()
    init_schema()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tenant_usage (tenant_id, day_utc, asks, retrieves, ingests)
                VALUES (%s, %s, 0, 0, 0)
                ON CONFLICT (tenant_id, day_utc) DO NOTHING
                """,
                (tenant_id, d),
            )
            cur.execute(
                f"UPDATE tenant_usage SET {field} = {field} + 1 WHERE tenant_id = %s AND day_utc = %s",
                (tenant_id, d),
            )
