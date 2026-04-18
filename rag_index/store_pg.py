from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from rag_storage.pg import PG_VECTOR_DIM, connect, init_schema


def _pad_vec(v: list[float]) -> list[float]:
    if len(v) == PG_VECTOR_DIM:
        return v
    if len(v) > PG_VECTOR_DIM:
        return v[:PG_VECTOR_DIM]
    return v + [0.0] * (PG_VECTOR_DIM - len(v))


def _fts_query(raw: str) -> str:
    parts = [w for w in re.split(r"\W+", raw) if len(w) > 1][:20]
    if not parts:
        return ""
    return " ".join(parts)


class ChunkIndexPostgres:
    """Dense vectors + full-text (Postgres tsvector) for one document (``source_sha256``)."""

    def __init__(self, source_sha256: str) -> None:
        self.source_sha256 = source_sha256
        init_schema()

    def clear(self) -> None:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM index_chunks WHERE source_sha256 = %s",
                    (self.source_sha256,),
                )
                cur.execute(
                    "DELETE FROM index_meta_kv WHERE source_sha256 = %s",
                    (self.source_sha256,),
                )

    def set_meta(self, key: str, value: str) -> None:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO index_meta_kv (source_sha256, k, v) VALUES (%s, %s, %s)
                    ON CONFLICT (source_sha256, k) DO UPDATE SET v = EXCLUDED.v
                    """,
                    (self.source_sha256, key, value),
                )

    def get_meta(self, key: str) -> str | None:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT v FROM index_meta_kv WHERE source_sha256 = %s AND k = %s",
                    (self.source_sha256, key),
                )
                row = cur.fetchone()
        return None if row is None else str(row[0])

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
        pj = json.dumps(payload, ensure_ascii=False)
        dims = len(vector)
        vec = _pad_vec(vector)
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO index_chunks (
                      source_sha256, chunk_id, document_id, payload_json, dims, embedding, fts_body
                    ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                    """,
                    (
                        self.source_sha256,
                        chunk_id,
                        document_id,
                        pj,
                        dims,
                        vec,
                        fts_body,
                    ),
                )

    def load_matrix(self) -> tuple[list[str], np.ndarray]:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, embedding, dims FROM index_chunks
                    WHERE source_sha256 = %s
                    ORDER BY chunk_id
                    """,
                    (self.source_sha256,),
                )
                rows = cur.fetchall()
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        ids = [str(r[0]) for r in rows]
        d = int(rows[0][2])
        mat = np.zeros((len(rows), d), dtype=np.float32)
        for i, r in enumerate(rows):
            emb = r[1]
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            take = min(d, arr.size)
            mat[i, :take] = arr[:take]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat = mat / norms
        return ids, mat

    def get_payload(self, chunk_id: str) -> dict | None:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT payload_json FROM index_chunks
                    WHERE source_sha256 = %s AND chunk_id = %s
                    """,
                    (self.source_sha256, chunk_id),
                )
                row = cur.fetchone()
        if row is None:
            return None
        raw = row[0]
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)

    def search_fts(self, query: str, k: int) -> list[tuple[str, float]]:
        q = _fts_query(query)
        if not q.strip():
            return []
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, ts_rank_cd(
                        to_tsvector('english', fts_body),
                        plainto_tsquery('english', %s)
                    ) AS rank
                    FROM index_chunks
                    WHERE source_sha256 = %s
                      AND to_tsvector('english', fts_body) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (q, self.source_sha256, q, k),
                )
                rows = cur.fetchall()
        # Return (chunk_id, score); higher rank first — reciprocal_rank_fusion treats list order as best-first
        return [(str(r[0]), float(r[1])) for r in rows]
