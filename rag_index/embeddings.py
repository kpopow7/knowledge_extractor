from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np

from rag_index import DEFAULT_EMBEDDING_MODEL, DEFAULT_FAKE_DIMS


def fake_embedding(texts: Sequence[str], dims: int = DEFAULT_FAKE_DIMS) -> list[list[float]]:
    """Deterministic pseudo-vectors for tests / offline runs (set RAG_INDEX_FAKE_EMBEDDINGS=1)."""
    out: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8", errors="replace")).digest()
        rng = int.from_bytes(h[:8], "big")
        vec = np.zeros(dims, dtype=np.float64)
        for i in range(dims):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            vec[i] = (rng / 0x7FFFFFFF) - 0.5
        n = np.linalg.norm(vec) or 1.0
        vec = vec / n
        out.append(vec.tolist())
    return out


def embed_texts(
    texts: Sequence[str],
    *,
    model: str | None = None,
) -> tuple[list[list[float]], str, int]:
    """
    Returns (vectors, model_name, dimensions).
    Uses OpenAI unless RAG_INDEX_FAKE_EMBEDDINGS=1.
    """
    model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if os.environ.get("RAG_INDEX_FAKE_EMBEDDINGS") == "1":
        dims = int(os.environ.get("RAG_INDEX_FAKE_DIMS", str(DEFAULT_FAKE_DIMS)))
        return fake_embedding(list(texts), dims=dims), f"fake:{dims}", dims
    if model.startswith("fake:"):
        dims = int(model.split(":", 1)[1])
        return fake_embedding(list(texts), dims=dims), model, dims

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "Install openai (pip install openai) or set RAG_INDEX_FAKE_EMBEDDINGS=1 for tests."
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (or use RAG_INDEX_FAKE_EMBEDDINGS=1).")

    from rag_index.embedding_cache import lookup, store

    texts_list = list(texts)
    slots, missing_idx = lookup(model, texts_list)
    if missing_idx:
        client = OpenAI(api_key=api_key)
        batch_size = 100
        new_texts = [texts_list[i] for i in missing_idx]
        new_vecs: list[list[float]] = []
        for i in range(0, len(new_texts), batch_size):
            batch = new_texts[i : i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            ordered = sorted(resp.data, key=lambda x: x.index)
            for item in ordered:
                new_vecs.append(item.embedding)
        for j, idx in enumerate(missing_idx):
            slots[idx] = new_vecs[j]
        store(model, new_texts, new_vecs)

    all_vecs = [v for v in slots if v is not None]
    if len(all_vecs) != len(texts_list):
        raise RuntimeError("embedding cache / API merge produced wrong count")
    dims = len(all_vecs[0]) if all_vecs else 0
    return all_vecs, model, dims


def cosine_topk(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """query (d,), matrix (n,d) row-normalized — returns (row_index, score) descending."""
    q = query / (np.linalg.norm(query) or 1.0)
    sims = matrix @ q
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]
