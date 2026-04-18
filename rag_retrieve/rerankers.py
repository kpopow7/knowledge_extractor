from __future__ import annotations

import os
from abc import ABC, abstractmethod

from rag_index.search import SearchHit


class Reranker(ABC):
    """Reorder a list of hits; may truncate to ``top_n``."""

    @abstractmethod
    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        pass


class PassthroughReranker(Reranker):
    """Keep RRF order; only slice to ``top_n``."""

    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        return hits[:top_n]


class CohereReranker(Reranker):
    """https://docs.cohere.com/docs/reranking"""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("COHERE_RERANK_MODEL", "rerank-english-v3.0")

    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        try:
            import cohere
        except ImportError as e:
            raise RuntimeError("Install cohere (pip install cohere) for Cohere reranking.") from e

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is not set.")

        client = cohere.Client(api_key=api_key)
        texts = [(h.payload.get("text_full") or "")[:48000] for h in hits]
        resp = client.rerank(
            query=query,
            documents=texts,
            top_n=min(top_n, len(texts)),
            model=self._model,
        )
        out: list[SearchHit] = []
        for r in resp.results:
            h = hits[r.index]
            out.append(
                SearchHit(
                    chunk_id=h.chunk_id,
                    rrf_score=float(r.relevance_score),
                    payload={**h.payload, "rerank_score": float(r.relevance_score)},
                )
            )
        return out


class CrossEncoderReranker(Reranker):
    """Local cross-encoder (sentence-transformers). Heavy dependency (PyTorch)."""

    _models: dict[str, object] = {}

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.environ.get(
            "CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        try:
            import numpy as np
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "Install sentence-transformers for cross-encoder reranking: pip install sentence-transformers"
            ) from e

        if self._model_name not in CrossEncoderReranker._models:
            CrossEncoderReranker._models[self._model_name] = CrossEncoder(self._model_name)
        model = CrossEncoderReranker._models[self._model_name]

        texts = [(h.payload.get("text_full") or "")[:8000] for h in hits]
        pairs = [[query, t] for t in texts]
        scores = model.predict(pairs, show_progress_bar=False)
        scores = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-scores)[:top_n]
        out: list[SearchHit] = []
        for i in order:
            ii = int(i)
            h = hits[ii]
            sc = float(scores[ii])
            out.append(
                SearchHit(
                    chunk_id=h.chunk_id,
                    rrf_score=sc,
                    payload={
                        **h.payload,
                        "rerank_score": sc,
                        "reranker": "cross-encoder",
                    },
                )
            )
        return out


def get_reranker(name: str) -> Reranker:
    n = name.strip().lower()
    if n in ("", "none", "passthrough", "rrf"):
        return PassthroughReranker()
    if n in ("cohere", "cohere-rerank"):
        return CohereReranker()
    if n in ("cross-encoder", "cross_encoder", "ce", "st"):
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker: {name!r} (use: none, cohere, cross-encoder)")
