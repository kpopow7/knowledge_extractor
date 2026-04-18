from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from rag_extractor.paths import storage_root

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenant_usage (
  tenant_id TEXT NOT NULL,
  day_utc TEXT NOT NULL,
  asks INTEGER NOT NULL DEFAULT 0,
  retrieves INTEGER NOT NULL DEFAULT 0,
  ingests INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (tenant_id, day_utc)
);
"""


def _db_path() -> Path:
    return storage_root() / "tenant_usage.sqlite"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@contextmanager
def _connect() -> sqlite3.Connection:
    _db_path().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_counts(tenant_id: str, day: str | None = None) -> tuple[int, int, int]:
    d = day or _today()
    with _lock:
        with _connect() as conn:
            row = conn.execute(
                "SELECT asks, retrieves, ingests FROM tenant_usage WHERE tenant_id = ? AND day_utc = ?",
                (tenant_id, d),
            ).fetchone()
            if row is None:
                return (0, 0, 0)
            return int(row[0]), int(row[1]), int(row[2])


def increment(tenant_id: str, field: str) -> None:
    if field not in ("asks", "retrieves", "ingests"):
        raise ValueError("field must be asks, retrieves, or ingests")
    d = _today()
    col = field
    with _lock:
        with _connect() as conn:
            conn.execute(
                f"""
                INSERT INTO tenant_usage (tenant_id, day_utc, asks, retrieves, ingests)
                VALUES (?, ?, 0, 0, 0)
                ON CONFLICT(tenant_id, day_utc) DO NOTHING
                """,
                (tenant_id, d),
            )
            conn.execute(
                f"UPDATE tenant_usage SET {col} = {col} + 1 WHERE tenant_id = ? AND day_utc = ?",
                (tenant_id, d),
            )
