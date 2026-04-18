from __future__ import annotations

import json
from pathlib import Path

from rag_chunker.models import ChunkRecord
from rag_index import INDEXER_VERSION
from rag_index.embeddings import embed_texts
from rag_index.store import ChunkIndex
from rag_index.store_pg import ChunkIndexPostgres
from rag_storage.config import use_postgres


def load_chunks_jsonl(path: Path) -> list[ChunkRecord]:
    rows: list[ChunkRecord] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(ChunkRecord.model_validate_json(line))
    return rows


def build_index(
    chunks_jsonl: Path,
    index_db: Path | None,
    *,
    embedding_model: str | None = None,
    clear: bool = True,
    postgres_source_sha256: str | None = None,
) -> tuple[int, str, int]:
    """
    Embed ``text_embed`` for each chunk, write SQLite index + FTS, or Postgres + pgvector when configured.
    Returns (count, model, dimensions).
    """
    chunks = load_chunks_jsonl(chunks_jsonl)
    if not chunks:
        return 0, "", 0

    texts = [c.text_embed or c.text_full for c in chunks]
    vectors, model, dims = embed_texts(texts, model=embedding_model)

    if postgres_source_sha256 and use_postgres():
        idx: ChunkIndex | ChunkIndexPostgres = ChunkIndexPostgres(postgres_source_sha256)
    else:
        if index_db is None:
            raise ValueError("index_db path required when not building Postgres index")
        idx = ChunkIndex(index_db)
    if clear:
        idx.clear()

    for c, vec in zip(chunks, vectors, strict=True):
        payload = c.model_dump()
        fts_body = f"{c.text_full}\n{' '.join(c.section_path)}"
        idx.insert_chunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            source_sha256=c.source_sha256,
            payload=payload,
            vector=vec,
            fts_body=fts_body,
        )

    idx.set_meta("embedding_model", model)
    idx.set_meta("embedding_dimensions", str(dims))
    idx.set_meta("indexer_version", INDEXER_VERSION)

    return len(chunks), model, dims
