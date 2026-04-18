from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from rag_eval.runner import load_cases, metrics_to_json, run_evaluation
from rag_index.build import build_index


def _read_thresholds(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_thresholds(metrics_json: dict, thresholds: dict) -> list[str]:
    """Return list of human-readable failures (empty if pass)."""
    failures: list[str] = []
    nl = metrics_json.get("n_labeled", 0)
    min_nl = thresholds.get("min_n_labeled")
    if min_nl is not None and nl < int(min_nl):
        failures.append(f"n_labeled={nl} < min_n_labeled={min_nl}")

    mrr = metrics_json.get("mrr", 0.0)
    min_mrr = thresholds.get("min_mrr")
    if min_mrr is not None and mrr + 1e-9 < float(min_mrr):
        failures.append(f"mrr={mrr:.6f} < min_mrr={min_mrr}")

    recall = metrics_json.get("recall") or {}
    min_recall = thresholds.get("min_recall_at") or {}
    for k_str, min_val in min_recall.items():
        key = f"@{k_str}" if not str(k_str).startswith("@") else str(k_str)
        if key not in recall:
            key_alt = f"@{k_str}".replace("@@", "@")
            if key_alt in recall:
                key = key_alt
        got = float(recall.get(key, 0.0))
        need = float(min_val)
        if got + 1e-9 < need:
            failures.append(f"recall[{key}]={got:.6f} < {need}")
    return failures


def run_gate(
    workspace: Path,
    *,
    ks: list[int] | None = None,
    final_k: int = 20,
    candidate_pool: int = 40,
    reranker: str = "none",
) -> int:
    """
    Build a temporary index from ``workspace/chunks.jsonl``, run ``cases.jsonl``,
    compare to ``thresholds.json``. Exits 0 on pass, 1 on failure.
    Forces ``RAG_INDEX_FAKE_EMBEDDINGS=1`` unless already set.
    """
    if ks is None:
        ks = [1, 5, 10]
    workspace = workspace.resolve()
    chunks = workspace / "chunks.jsonl"
    cases_path = workspace / "cases.jsonl"
    thr_path = workspace / "thresholds.json"
    for p, label in ((chunks, "chunks.jsonl"), (cases_path, "cases.jsonl"), (thr_path, "thresholds.json")):
        if not p.is_file():
            print(f"Missing {label} under {workspace}", file=sys.stderr)
            return 1

    if os.environ.get("RAG_INDEX_FAKE_EMBEDDINGS") != "1":
        os.environ["RAG_INDEX_FAKE_EMBEDDINGS"] = "1"

    thresholds = _read_thresholds(thr_path)
    cases = load_cases(cases_path)

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "gate_index.sqlite"
        build_index(chunks, db, clear=True)
        _per, metrics = run_evaluation(
            db,
            cases,
            ks=ks,
            final_k=final_k,
            candidate_pool=candidate_pool,
            reranker=reranker,
        )
        summary = metrics_to_json(metrics, ks)
        print(json.dumps(summary, indent=2))

    failures = check_thresholds(summary, thresholds)
    if failures:
        print("Eval gate FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("Eval gate OK.")
    return 0
