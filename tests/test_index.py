from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from rag_chunker.models import ChunkRecord
from rag_index.build import build_index, load_chunks_jsonl
from rag_index.search import search_hybrid
def _sample_chunk(suffix: str = "") -> ChunkRecord:
    return ChunkRecord(
        schema_version="chunk.v1",
        chunker_version="t",
        chunk_id=f"cid{suffix}",
        document_id="doc",
        source_sha256="c" * 64,
        source_filename="f.pdf",
        extraction_version="e",
        chunk_index=0,
        page_start=1,
        page_end=1,
        section_path=["Intro"],
        block_ids=["b1"],
        content_type="prose",
        text_full=f"battery wand specifications {suffix}",
        text_embed=f"[Source: f.pdf | Section: Intro]\nbattery wand specifications {suffix}",
        char_start_in_doc=0,
        char_end_in_doc=50,
    )


class TestIndexBuildSearch(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._root = Path(self._tmp.name)
        os.environ["RAG_STORAGE_ROOT"] = str(self._root / "storage")
        os.environ["RAG_INDEX_FAKE_EMBEDDINGS"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("RAG_INDEX_FAKE_EMBEDDINGS", None)
        os.environ.pop("RAG_STORAGE_ROOT", None)

    def test_build_and_hybrid_search(self) -> None:
        c1 = _sample_chunk("a")
        c2 = _sample_chunk("b")
        c2.chunk_id = "cid2"
        c2.chunk_index = 1
        c2.text_full = "unrelated cellular shade fabric"
        c2.text_embed = f"[Source: f.pdf | Section: X]\n{c2.text_full}"

        jl = self._root / "chunks.jsonl"
        jl.write_text(c1.model_dump_json() + "\n" + c2.model_dump_json() + "\n", encoding="utf-8")

        db = self._root / "t.sqlite"
        n, model, dims = build_index(jl, db, clear=True)
        self.assertEqual(n, 2)
        self.assertTrue(model.startswith("fake:"))
        self.assertGreater(dims, 0)

        hits = search_hybrid(db, "battery wand", top_k=5)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("battery", hits[0].payload.get("text_full", "").lower())


class TestLoadJsonl(unittest.TestCase):
    def test_roundtrip(self) -> None:
        c = _sample_chunk()
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d) / "x.jsonl"
            fp.write_text(c.model_dump_json() + "\n", encoding="utf-8")
            rows = load_chunks_jsonl(fp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].chunk_id, c.chunk_id)


if __name__ == "__main__":
    unittest.main()
