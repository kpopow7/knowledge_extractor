from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_api.app import app
from rag_api.tenants import reload_registry


class TestPlatform(unittest.TestCase):
    def test_quota_blocks_ask(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "tenants": [
                        {
                            "tenant_id": "t1",
                            "api_keys": ["q-key"],
                            "quotas": {"asks_per_day": 0},
                        }
                    ]
                },
                f,
            )
            path = f.name
        try:
            with patch.dict(
                os.environ,
                {"RAG_TENANTS_FILE": path},
                clear=False,
            ):
                reload_registry()
                with TestClient(app) as client:
                    r = client.post(
                        "/v1/ask",
                        headers={"X-API-Key": "q-key"},
                        json={
                            "question": "hi",
                            "sha256": "a" * 64,
                        },
                    )
            self.assertEqual(r.status_code, 429)
        finally:
            os.unlink(path)
            os.environ.pop("RAG_TENANTS_FILE", None)
            reload_registry()

    def test_stream_endpoint_exists(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with TestClient(app) as client:
                r = client.post(
                    "/v1/ask/stream",
                    json={"question": "x", "sha256": "a" * 64},
                )
        self.assertEqual(r.status_code, 503)
