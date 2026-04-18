from __future__ import annotations

import json
import re
import sqlite3
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


def _pack_vec(v: list[float]) -> bytes:
    a = array("f", v)
    return a.tobytes()


def _unpack_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


SCHEMA = """
CREATE TABLE IF NOT EXISTS index_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vec BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    body,
    tokenize = 'unicode61'
);

CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(source_sha256);
"""


class ChunkIndex:
    """Per-corpus SQLite: dense vectors + FTS5 keyword index."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM index_meta")

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO index_meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT v FROM index_meta WHERE k = ?", (key,)).fetchone()
        return None if row is None else str(row["v"])

    def insert_chunk(
        self,
        *,
        chunk_id: str,
        document_id: str,
        source_sha256: str,
        payload: dict,
        vector: list[float],
        fts_body: str,
    ) -> None:
        blob = _pack_vec(vector)
        dims = len(vector)
        pj = json.dumps(payload, ensure_ascii=False)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, source_sha256, payload_json, dims, vec)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, source_sha256, pj, dims, blob),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, body) VALUES (?, ?)",
                (chunk_id, fts_body),
            )

    def load_matrix(self) -> tuple[list[str], np.ndarray]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, vec, dims FROM chunks ORDER BY chunk_id"
            ).fetchall()
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        ids = [r["chunk_id"] for r in rows]
        first = _unpack_vec(rows[0]["vec"])
        d = int(rows[0]["dims"])
        mat = np.zeros((len(rows), d), dtype=np.float32)
        mat[0] = first
        for i, r in enumerate(rows[1:], start=1):
            mat[i] = _unpack_vec(r["vec"])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat = mat / norms
        return ids, mat

    def get_payload(self, chunk_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def search_fts(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25) lower is better for bm25."""
        if not query.strip():
            return []
        fts_q = _fts_match_query(query)
        if not fts_q:
            return []
        with self._conn() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT chunk_id, bm25(chunks_fts) AS score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (fts_q, k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(r["chunk_id"], float(r["score"])) for r in rows]


def _fts_match_query(raw: str) -> str:
    """Token AND-query for FTS5; avoids bare punctuation breaking MATCH."""
    parts = [w for w in re.split(r"\W+", raw) if len(w) > 1][:20]
    if not parts:
        return ""
    return " AND ".join(parts)
