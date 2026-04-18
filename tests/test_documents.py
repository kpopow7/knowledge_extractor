from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_api.app import app
from rag_api import job_store
from rag_api.tenants import reload_registry


class TestDocumentsApi(unittest.TestCase):
    def test_list_documents_open_mode(self) -> None:
        with TestClient(app) as client:
            r = client.get("/v1/documents")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("documents", data)
        self.assertIsInstance(data["documents"], list)

    @patch.dict(os.environ, {"RAG_API_KEYS": "k1"}, clear=False)
    def test_list_documents_requires_auth_when_keys_set(self) -> None:
        reload_registry()
        try:
            with TestClient(app) as client:
                r = client.get("/v1/documents")
            self.assertEqual(r.status_code, 401)
            with TestClient(app) as client:
                r2 = client.get("/v1/documents", headers={"X-API-Key": "k1"})
            self.assertEqual(r2.status_code, 200)
            self.assertIn("documents", r2.json())
        finally:
            reload_registry()

    def test_tenant_scoped_lists_from_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {"RAG_STORAGE_ROOT": tmp, "DATABASE_URL": ""},
                clear=False,
            ):
                job_store.create_job("job-a", "a.pdf", tenant_id="t1")
                job_store.set_status("job-a", "ready", content_sha256="a" * 64)
                job_store.create_job("job-b", "b.pdf", tenant_id="t2")
                job_store.set_status("job-b", "ready", content_sha256="b" * 64)

                shas = job_store.list_tenant_document_shas("t1", limit=10, offset=0)
                self.assertEqual(shas, ["a" * 64])
                shas2 = job_store.list_tenant_document_shas("t2", limit=10, offset=0)
                self.assertEqual(shas2, ["b" * 64])
