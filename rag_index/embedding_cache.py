from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from rag_extractor.paths import storage_root

log = logging.getLogger(__name__)

_lock = threading.Lock()


def cache_enabled() -> bool:
    """Opt-in disk cache for OpenAI embeddings (never used for fake embeddings)."""
    return (os.environ.get("RAG_EMBEDDING_CACHE") or "").strip().lower() in ("1", "true", "yes")


def _db_path() -> Path:
    return storage_root() / "cache" / "embedding_cache.sqlite"


def _connect() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
          cache_key TEXT PRIMARY KEY,
          model TEXT NOT NULL,
          dims INTEGER NOT NULL,
          vec_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _cache_key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()


def lookup(model: str, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
    """
    Return parallel list of vectors (None if missing) and indices that need an API call.
    """
    n = len(texts)
    slots: list[list[float] | None] = [None] * n
    if not cache_enabled() or n == 0:
        return slots, list(range(n))

    keys = [_cache_key(model, t) for t in texts]
    with _lock:
        conn = _connect()
        try:
            for i, k in enumerate(keys):
                row = conn.execute(
                    "SELECT vec_json FROM embeddings WHERE cache_key = ? AND model = ?",
                    (k, model),
                ).fetchone()
                if row is not None:
                    slots[i] = json.loads(row[0])
        finally:
            conn.close()

    missing = [i for i in range(n) if slots[i] is None]
    return slots, missing


def store(model: str, texts: list[str], vectors: list[list[float]]) -> None:
    if not cache_enabled() or not texts:
        return
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _lock:
        conn = _connect()
        try:
            for t, vec in zip(texts, vectors, strict=True):
                k = _cache_key(model, t)
                conn.execute(
                    """
                    INSERT INTO embeddings (cache_key, model, dims, vec_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                      vec_json = excluded.vec_json,
                      dims = excluded.dims,
                      model = excluded.model,
                      created_at = excluded.created_at
                    """,
                    (k, model, len(vec), json.dumps(vec), now),
                )
            conn.commit()
        finally:
            conn.close()
