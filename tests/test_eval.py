from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from rag_chunker.models import ChunkRecord
from rag_eval.runner import load_cases, run_evaluation
from rag_index.build import build_index


def _chunk(i: int, text: str, cid: str) -> ChunkRecord:
    return ChunkRecord(
        schema_version="chunk.v1",
        chunker_version="t",
        chunk_id=cid,
        document_id="doc",
        source_sha256="e" * 64,
        source_filename="f.pdf",
        extraction_version="e",
        chunk_index=i,
        page_start=1,
        page_end=1,
        section_path=[],
        block_ids=[f"b{i}"],
        content_type="prose",
        text_full=text,
        text_embed=f"[Source: f.pdf]\n{text}",
        char_start_in_doc=0,
        char_end_in_doc=len(text),
    )


class TestEvalRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._root = Path(self._tmp.name)
        os.environ["RAG_INDEX_FAKE_EMBEDDINGS"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("RAG_INDEX_FAKE_EMBEDDINGS", None)

    def test_recall_and_mrr(self) -> None:
        chunks = [
            _chunk(0, "alpha uniqueterm one", "c0"),
            _chunk(1, "beta other", "c1"),
        ]
        jl = self._root / "c.jsonl"
        jl.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8")
        db = self._root / "idx.sqlite"
        build_index(jl, db, clear=True)

        ev = self._root / "eval.jsonl"
        ev.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "question": "uniqueterm",
                    "gold_substrings": ["uniqueterm"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cases = load_cases(ev)
        _, metrics = run_evaluation(db, cases, ks=[1, 5], final_k=10, reranker="none")
        self.assertEqual(metrics.n_labeled, 1)
        self.assertGreaterEqual(metrics.recall_at[1], 1.0)
        self.assertGreater(metrics.mrr, 0.0)


if __name__ == "__main__":
    unittest.main()
