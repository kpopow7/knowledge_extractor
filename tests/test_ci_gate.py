from __future__ import annotations

import unittest

from rag_eval.gate import run_gate


class TestCiEvalGate(unittest.TestCase):
    def test_ci_workspace_passes(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "ci"
        rc = run_gate(root)
        self.assertEqual(rc, 0)
