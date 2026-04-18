from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import fitz

from rag_extractor.ingest import ingest_pdf
from rag_extractor.paths import storage_root
from rag_extractor.registry import DocumentRegistry


class TestIngest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self._storage = root / "storage"
        os.environ["RAG_STORAGE_ROOT"] = str(self._storage)

    def tearDown(self) -> None:
        os.environ.pop("RAG_STORAGE_ROOT", None)

    def _minimal_pdf(self) -> Path:
        p = Path(self._tmp.name) / "minimal.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Phase 1 ingest test")
        doc.save(str(p))
        doc.close()
        return p

    def test_ingest_idempotent(self) -> None:
        pdf = self._minimal_pdf()
        r1 = ingest_pdf(pdf)
        self.assertEqual(r1.status, "ready")
        self.assertFalse(r1.skipped)
        self.assertIsNotNone(r1.artifact_path)
        self.assertTrue(r1.artifact_path and r1.artifact_path.is_file())

        r2 = ingest_pdf(pdf)
        self.assertEqual(r2.status, "ready")
        self.assertTrue(r2.skipped)
        self.assertEqual(r2.reason, "already_ingested_same_version")

    def test_ingest_force(self) -> None:
        pdf = self._minimal_pdf()
        ingest_pdf(pdf)
        r2 = ingest_pdf(pdf, force=True)
        self.assertFalse(r2.skipped)
        self.assertEqual(r2.status, "ready")

    def test_registry_list(self) -> None:
        pdf = self._minimal_pdf()
        ingest_pdf(pdf)
        reg = DocumentRegistry()
        rows = reg.list_recent(10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "ready")
        self.assertEqual(rows[0].page_count, 1)

    def test_storage_root_env(self) -> None:
        self.assertEqual(storage_root().resolve(), self._storage.resolve())


if __name__ == "__main__":
    unittest.main()
