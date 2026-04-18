from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from rag_chunker.models import ChunkRecord
from rag_index.build import build_index
from rag_retrieve.pipeline import retrieve


def _chunk(suffix: str, idx: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        schema_version="chunk.v1",
        chunker_version="t",
        chunk_id=f"cid{idx}",
        document_id="doc",
        source_sha256="d" * 64,
        source_filename="f.pdf",
        extraction_version="e",
        chunk_index=idx,
        page_start=1,
        page_end=1,
        section_path=[],
        block_ids=[f"b{idx}"],
        content_type="prose",
        text_full=text,
        text_embed=f"[Source: f.pdf]\n{text}",
        char_start_in_doc=0,
        char_end_in_doc=len(text),
    )


class TestRetrievePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._root = Path(self._tmp.name)
        os.environ["RAG_INDEX_FAKE_EMBEDDINGS"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("RAG_INDEX_FAKE_EMBEDDINGS", None)

    def test_passthrough_rerank_returns_k(self) -> None:
        chunks = [
            _chunk("a", 0, "alpha battery specification"),
            _chunk("b", 1, "unrelated zebra fabric"),
        ]
        jl = self._root / "c.jsonl"
        jl.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8")
        db = self._root / "i.sqlite"
        build_index(jl, db, clear=True)

        hits = retrieve(db, "battery", final_k=1, candidate_pool=4, reranker="none")
        self.assertEqual(len(hits), 1)
        self.assertIn("battery", hits[0].payload.get("text_full", "").lower())


if __name__ == "__main__":
    unittest.main()
