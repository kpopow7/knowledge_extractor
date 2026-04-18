from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_api.app import app


class TestAdminApi(unittest.TestCase):
    def test_admin_disabled(self) -> None:
        with patch.dict(os.environ, {"RAG_ADMIN_API_KEYS": ""}, clear=False):
            with TestClient(app) as client:
                r = client.get("/v1/admin/jobs")
        self.assertEqual(r.status_code, 403)

    @patch.dict(os.environ, {"RAG_ADMIN_API_KEYS": "adm-secret"}, clear=False)
    def test_admin_list_jobs(self) -> None:
        with TestClient(app) as client:
            r = client.get("/v1/admin/jobs", headers={"X-Admin-API-Key": "adm-secret"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("jobs", data)
        self.assertIsInstance(data["jobs"], list)
