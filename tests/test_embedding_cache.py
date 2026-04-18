from __future__ import annotations

import os
import tempfile
import unittest

from rag_index.embedding_cache import cache_enabled, lookup, store


class TestEmbeddingCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["RAG_STORAGE_ROOT"] = self._tmp.name
        os.environ["RAG_EMBEDDING_CACHE"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("RAG_EMBEDDING_CACHE", None)
        os.environ.pop("RAG_STORAGE_ROOT", None)

    def test_roundtrip_and_miss(self) -> None:
        self.assertTrue(cache_enabled())
        store("text-embedding-test", ["hello"], [[0.1, 0.2, 0.3]])
        slots, missing = lookup("text-embedding-test", ["hello", "world"])
        self.assertEqual(slots[0], [0.1, 0.2, 0.3])
        self.assertIsNone(slots[1])
        self.assertEqual(missing, [1])
