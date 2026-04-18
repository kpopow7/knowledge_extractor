from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_api.app import app


class TestRagApi(unittest.TestCase):
    def test_health(self) -> None:
        with TestClient(app) as client:
            r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_web_ui_redirect_and_static(self) -> None:
        with TestClient(app) as client:
            r = client.get("/", follow_redirects=False)
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r.headers.get("location"), "/ui/")
        with TestClient(app) as client:
            r2 = client.get("/ui/")
        self.assertEqual(r2.status_code, 200)
        self.assertIn(b"Knowledge RAG", r2.content)
        self.assertIn(b"/v1/ask/stream", r2.content)

    def test_prometheus_metrics_endpoint(self) -> None:
        with TestClient(app) as client:
            r = client.get("/metrics")
        if r.status_code == 404:
            self.skipTest("Prometheus disabled (RAG_PROMETHEUS_METRICS=0)")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/plain", r.headers.get("content-type", ""))
        self.assertIn(b"http_requests", r.content)

    def test_request_id_header(self) -> None:
        with TestClient(app) as client:
            r = client.get("/health", headers={"X-Request-ID": "custom-id"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("X-Request-ID"), "custom-id")

    def test_retrieve_requires_target(self) -> None:
        with TestClient(app) as client:
            r = client.post("/v1/retrieve", json={"query": "test"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("detail", r.json())

    @patch.dict(os.environ, {"RAG_API_KEYS": "secret-key"}, clear=False)
    def test_v1_requires_api_key_when_configured(self) -> None:
        with TestClient(app) as client:
            r = client.post("/v1/retrieve", json={"query": "test", "sha256": "abc"})
        self.assertEqual(r.status_code, 401)
        with TestClient(app) as client:
            r2 = client.post(
                "/v1/retrieve",
                json={"query": "test", "sha256": "abc"},
                headers={"X-API-Key": "secret-key"},
            )
        self.assertNotEqual(r2.status_code, 401)

    def test_ingest_rejects_non_pdf(self) -> None:
        with TestClient(app) as client:
            r = client.post(
                "/v1/ingest",
                files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
            )
        self.assertEqual(r.status_code, 400)
        detail = r.json().get("detail", "")
        self.assertIn("%PDF", str(detail))

    def test_ingest_rejects_non_pdf_extension(self) -> None:
        with TestClient(app) as client:
            r = client.post(
                "/v1/ingest",
                files={"file": ("x.txt", b"%PDF-1.4 fake", "text/plain")},
            )
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
