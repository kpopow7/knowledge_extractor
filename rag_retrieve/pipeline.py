from __future__ import annotations

import os

from rag_index.search import SearchHit, search_hybrid
from rag_retrieve.rerankers import Reranker, get_reranker


def retrieve(
    index_db: os.PathLike[str] | str,
    query: str,
    *,
    final_k: int = 10,
    candidate_pool: int = 40,
    reranker: str | Reranker | None = None,
    embedding_model: str | None = None,
    vec_candidates: int = 60,
    kw_candidates: int = 60,
) -> list[SearchHit]:
    """
    Hybrid retrieval over a larger pool, then rerank to ``final_k``.
    """
    if isinstance(reranker, Reranker):
        rr = reranker
    else:
        rr = get_reranker((reranker or "none").strip().lower())

    pool = max(candidate_pool, final_k)
    hits = search_hybrid(
        index_db,
        query,
        top_k=final_k,
        candidate_pool=pool,
        vec_candidates=vec_candidates,
        kw_candidates=kw_candidates,
        embedding_model=embedding_model,
    )
    return rr.rerank(query, hits, top_n=final_k)
