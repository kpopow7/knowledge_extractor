from __future__ import annotations

import json
from datetime import datetime, timezone

from psycopg.rows import dict_row

from rag_index.embedding_cache import embedding_cache_key
from rag_storage.pg import connect, init_schema


def lookup(model: str, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
    n = len(texts)
    slots: list[list[float] | None] = [None] * n
    if n == 0:
        return slots, []

    keys = [embedding_cache_key(model, t) for t in texts]
    init_schema()
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT cache_key, vec_json FROM embedding_cache
                WHERE model = %s AND cache_key = ANY(%s)
                """,
                (model, keys),
            )
            rows = cur.fetchall()
    by_key = {str(r["cache_key"]): r["vec_json"] for r in rows}
    for i, k in enumerate(keys):
        raw = by_key.get(k)
        if raw is None:
            continue
        if isinstance(raw, list):
            slots[i] = [float(x) for x in raw]
        else:
            slots[i] = json.loads(raw) if isinstance(raw, str) else raw

    missing = [i for i in range(n) if slots[i] is None]
    return slots, missing


def store(model: str, texts: list[str], vectors: list[list[float]]) -> None:
    if not texts:
        return
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    init_schema()
    with connect() as conn:
        with conn.cursor() as cur:
            for t, vec in zip(texts, vectors, strict=True):
                k = embedding_cache_key(model, t)
                payload = json.dumps(vec, ensure_ascii=False)
                cur.execute(
                    """
                    INSERT INTO embedding_cache (cache_key, model, dims, vec_json, created_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                      vec_json = EXCLUDED.vec_json,
                      dims = EXCLUDED.dims,
                      model = EXCLUDED.model,
                      created_at = EXCLUDED.created_at
                    """,
                    (k, model, len(vec), payload, now),
                )
