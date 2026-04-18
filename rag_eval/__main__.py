from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_eval.runner import load_cases, metrics_to_json, run_evaluation
from rag_extractor.paths import storage_root
from rag_extractor.registry import DocumentRegistry


def _apply_storage_root(ns: argparse.Namespace) -> None:
    root = getattr(ns, "storage_root", None)
    if root is not None:
        os.environ["RAG_STORAGE_ROOT"] = str(Path(root).resolve())


def _resolve_sha256(reg: DocumentRegistry, key: str) -> str:
    key = key.strip().lower()
    if len(key) == 64:
        return key
    rows = reg.list_recent(500)
    matches = [r.content_sha256 for r in rows if r.content_sha256.startswith(key)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"No document matches prefix {key!r}.")
    raise SystemExit(f"Ambiguous prefix {key!r}: {len(matches)} matches.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrieval eval: MRR and recall@k.")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run eval cases against an index")
    run.add_argument("--cases", type=Path, required=True, help="JSONL of EvalCase rows")
    run.add_argument("--index", type=Path, help="SQLite index path")
    run.add_argument("--sha256", metavar="HASH", help="Resolve index from registry")
    run.add_argument("--ks", default="1,5,10,20", help="Comma-separated k for recall@k")
    run.add_argument("--final-k", type=int, default=20, dest="final_k")
    run.add_argument("--candidates", type=int, default=40, dest="candidate_pool")
    run.add_argument("--rerank", default="none")
    run.add_argument("--per-case", type=Path, help="Write per-case JSON lines here")
    run.add_argument("--storage-root", type=Path, default=None)

    args = p.parse_args(argv)
    _apply_storage_root(args)

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    if args.cmd != "run":
        return 1

    idx_path: Path | None = args.index
    if args.sha256:
        reg = DocumentRegistry()
        sha = _resolve_sha256(reg, args.sha256)
        rec = reg.get(sha)
        if not rec or not rec.index_db_relpath:
            raise SystemExit("No index in registry.")
        idx_path = storage_root() / rec.index_db_relpath
    if idx_path is None:
        raise SystemExit("Provide --index or --sha256.")

    cases = load_cases(args.cases)
    per_case, metrics = run_evaluation(
        idx_path,
        cases,
        ks=ks,
        final_k=args.final_k,
        candidate_pool=args.candidate_pool,
        reranker=args.rerank,
    )

    if args.per_case:
        args.per_case.parent.mkdir(parents=True, exist_ok=True)
        with args.per_case.open("w", encoding="utf-8") as f:
            for row in per_case:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = metrics_to_json(metrics, ks)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
