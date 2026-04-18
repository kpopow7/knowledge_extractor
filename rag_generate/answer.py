from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

from rag_index.search import SearchHit
from rag_index.targets import SearchIndexTarget
from rag_retrieve.pipeline import retrieve


def _env_float(name: str, default: float | None) -> float | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def retrieve_timeout_sec() -> float | None:
    """Max seconds for hybrid retrieval + rerank (None = unlimited)."""
    return _env_float("RAG_RETRIEVE_TIMEOUT_SEC", 90.0)


def llm_timeout_sec() -> float | None:
    """Max seconds for each OpenAI chat completion (HTTP / generation)."""
    return _env_float("RAG_LLM_TIMEOUT_SEC", 120.0)


def ask_total_budget_sec() -> float | None:
    """Optional ceiling for the whole non-streaming ask (retrieve + LLM)."""
    return _env_float("RAG_ASK_TOTAL_BUDGET_SEC", 180.0)


def stream_total_budget_sec() -> float | None:
    """Optional max wall time for streaming token generation (after retrieval)."""
    return _env_float("RAG_ASK_STREAM_BUDGET_SEC", 180.0)


def _format_context(hits: list[SearchHit], max_chars_per_chunk: int = 6000) -> str:
    blocks: list[str] = []
    for h in hits:
        pl = h.payload
        fn = pl.get("source_filename") or "document"
        p0 = pl.get("page_start")
        p1 = pl.get("page_end")
        head = f"[{fn} | pages {p0}–{p1}]"
        body = (pl.get("text_full") or "")[:max_chars_per_chunk]
        blocks.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _openai_client(*, llm_timeout: float | None = None):
    try:
        import httpx
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai (pip install openai) for generation.") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    to = llm_timeout if llm_timeout is not None else llm_timeout_sec()
    timeout = httpx.Timeout(to, connect=min(30.0, to or 30.0)) if to else httpx.Timeout(120.0, connect=30.0)
    return OpenAI(api_key=api_key, timeout=timeout)


def _retrieve_step(
    index_ref: SearchIndexTarget | os.PathLike[str] | str,
    question: str,
    *,
    final_k: int,
    candidate_pool: int,
    reranker: str,
    embedding_model: str | None,
) -> list[SearchHit]:
    return retrieve(
        index_ref,
        question,
        final_k=final_k,
        candidate_pool=candidate_pool,
        reranker=reranker,
        embedding_model=embedding_model,
    )


def _run_retrieve_with_optional_timeout(
    index_ref: SearchIndexTarget | os.PathLike[str] | str,
    question: str,
    *,
    final_k: int,
    candidate_pool: int,
    reranker: str,
    embedding_model: str | None,
    retrieve_timeout: float | None,
) -> list[SearchHit]:
    from .budgets import run_with_timeout

    def _do() -> list[SearchHit]:
        return _retrieve_step(
            index_ref,
            question,
            final_k=final_k,
            candidate_pool=candidate_pool,
            reranker=reranker,
            embedding_model=embedding_model,
        )

    rt = retrieve_timeout if retrieve_timeout is not None else retrieve_timeout_sec()
    return run_with_timeout(_do, rt, label="retrieval")


def answer_with_retrieval(
    index_ref: SearchIndexTarget | os.PathLike[str] | str,
    question: str,
    *,
    chat_model: str | None = None,
    final_k: int = 8,
    candidate_pool: int = 40,
    reranker: str = "none",
    embedding_model: str | None = None,
    retrieve_timeout: float | None = None,
    llm_timeout: float | None = None,
    total_budget: float | None = None,
) -> tuple[str, list[SearchHit]]:
    """
    Retrieve chunks, then call the OpenAI Chat Completions API with a grounded prompt.
    Requires OPENAI_API_KEY (same client as embeddings).

    Timeouts: ``retrieve_timeout`` / ``llm_timeout`` override env defaults.
    ``total_budget`` wraps the whole operation (retrieve + completion).
    """
    from .budgets import run_with_timeout

    def _whole() -> tuple[str, list[SearchHit]]:
        hits = _run_retrieve_with_optional_timeout(
            index_ref,
            question,
            final_k=final_k,
            candidate_pool=candidate_pool,
            reranker=reranker,
            embedding_model=embedding_model,
            retrieve_timeout=retrieve_timeout,
        )
        if not hits:
            return (
                "No retrieved context matched this question. Try a different query or rebuild the index.",
                [],
            )

        client = _openai_client(llm_timeout=llm_timeout)

        model = chat_model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        context = _format_context(hits)
        system = (
            "You are a technical assistant for product specification guides. "
            "Answer using ONLY the provided context. If the answer is not in the context, say you do not know. "
            "Cite sources inline using [filename, page X] when possible."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, hits

    tb = total_budget if total_budget is not None else ask_total_budget_sec()
    return run_with_timeout(_whole, tb, label="ask")


def iter_ask_stream_events(
    index_ref: SearchIndexTarget | os.PathLike[str] | str,
    question: str,
    *,
    chat_model: str | None = None,
    final_k: int = 8,
    candidate_pool: int = 40,
    reranker: str = "none",
    embedding_model: str | None = None,
    retrieve_timeout: float | None = None,
    llm_timeout: float | None = None,
    stream_budget: float | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Retrieve (blocking), then stream LLM tokens as event dicts:
    ``retrieval`` (chunk metadata), ``token`` (text delta), ``done``, or ``error``.
    """
    try:
        hits = _run_retrieve_with_optional_timeout(
            index_ref,
            question,
            final_k=final_k,
            candidate_pool=candidate_pool,
            reranker=reranker,
            embedding_model=embedding_model,
            retrieve_timeout=retrieve_timeout,
        )
    except TimeoutError as e:
        yield {"type": "error", "detail": str(e)}
        return

    if not hits:
        yield {
            "type": "error",
            "detail": "No retrieved context matched this question. Try a different query or rebuild the index.",
        }
        return

    pages: list[list[int | None]] = [
        [h.payload.get("page_start"), h.payload.get("page_end")] for h in hits
    ]
    yield {
        "type": "retrieval",
        "chunk_ids": [h.chunk_id for h in hits],
        "pages": pages,
    }

    client = _openai_client(llm_timeout=llm_timeout)

    model = chat_model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    context = _format_context(hits)
    system = (
        "You are a technical assistant for product specification guides. "
        "Answer using ONLY the provided context. If the answer is not in the context, say you do not know. "
        "Cite sources inline using [filename, page X] when possible."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        stream=True,
    )

    budget = stream_budget if stream_budget is not None else stream_total_budget_sec()
    start = time.monotonic()
    try:
        for chunk in stream:
            if budget is not None and time.monotonic() - start > budget:
                yield {"type": "error", "detail": f"Stream exceeded budget ({budget}s)"}
                return
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta.content if choice.delta else None
            if delta:
                yield {"type": "token", "text": delta}
    except Exception as e:
        yield {"type": "error", "detail": str(e)}
        return

    yield {"type": "done"}
