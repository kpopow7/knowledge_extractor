from __future__ import annotations

import os

from rag_index.search import SearchHit
from rag_index.targets import SearchIndexTarget
from rag_retrieve.pipeline import retrieve


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


def answer_with_retrieval(
    index_ref: SearchIndexTarget | os.PathLike[str] | str,
    question: str,
    *,
    chat_model: str | None = None,
    final_k: int = 8,
    candidate_pool: int = 40,
    reranker: str = "none",
    embedding_model: str | None = None,
) -> tuple[str, list[SearchHit]]:
    """
    Retrieve chunks, then call the OpenAI Chat Completions API with a grounded prompt.
    Requires OPENAI_API_KEY (same client as embeddings).
    """
    hits = retrieve(
        index_ref,
        question,
        final_k=final_k,
        candidate_pool=candidate_pool,
        reranker=reranker,
        embedding_model=embedding_model,
    )
    if not hits:
        return (
            "No retrieved context matched this question. Try a different query or rebuild the index.",
            [],
        )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai (pip install openai) for generation.") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = chat_model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
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
