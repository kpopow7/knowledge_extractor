from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_index.embeddings import embed_texts
from rag_index.hybrid import reciprocal_rank_fusion
from rag_index.store import ChunkIndex


@dataclass
class SearchHit:
    chunk_id: str
    rrf_score: float
    payload: dict


def search_hybrid(
    index_db: os.PathLike[str] | str,
    query: str,
    *,
    top_k: int = 15,
    candidate_pool: int | None = None,
    rrf_k: int = 60,
    vec_candidates: int = 50,
    kw_candidates: int = 50,
    embedding_model: str | None = None,
) -> list[SearchHit]:
    """
    Dense cosine on normalized vectors + FTS5 BM25, merged with RRF.
    Uses the embedding model recorded in the index unless overridden.
    If ``candidate_pool`` is set and greater than ``top_k``, returns that many hits
    (for downstream reranking); otherwise returns ``top_k`` results.
    """
    idx = ChunkIndex(Path(index_db))
    model = embedding_model or idx.get_meta("embedding_model")
    ids, mat = idx.load_matrix()
    if not ids or mat.size == 0:
        return []

    q_vecs, _, _ = embed_texts([query], model=model)
    q = np.array(q_vecs[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)

    sims = mat @ q
    vec_order = list(np.argsort(-sims)[:vec_candidates])
    semantic_ids = [ids[i] for i in vec_order]

    fts_rows = idx.search_fts(query, kw_candidates)
    fts_ids = [r[0] for r in fts_rows]

    merged = reciprocal_rank_fusion([semantic_ids, fts_ids], k=rrf_k)
    take = max(top_k, candidate_pool) if candidate_pool is not None else top_k
    out: list[SearchHit] = []
    for cid, score in merged[:take]:
        pl = idx.get_payload(cid)
        if pl is not None:
            out.append(SearchHit(chunk_id=cid, rrf_score=score, payload=pl))
    return out
