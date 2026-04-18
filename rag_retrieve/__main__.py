from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_retrieve.pipeline import retrieve
from rag_api.registry_resolve import resolve_index_db


def _apply_storage_root(ns: argparse.Namespace) -> None:
    root = getattr(ns, "storage_root", None)
    if root is not None:
        os.environ["RAG_STORAGE_ROOT"] = str(Path(root).resolve())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4: hybrid retrieval + rerank.")
    p.add_argument("--index", type=Path, help="SQLite index path")
    p.add_argument("--sha256", metavar="HASH", help="Resolve index from registry")
    p.add_argument("query", nargs="?", help="Search query")
    p.add_argument("--top", type=int, default=10, dest="final_k", help="Final results after rerank")
    p.add_argument(
        "--candidates",
        type=int,
        default=40,
        dest="candidate_pool",
        help="RRF pool size before rerank (should be >= --top)",
    )
    p.add_argument(
        "--rerank",
        default="none",
        help="none | cohere (COHERE_API_KEY) | cross-encoder (sentence-transformers)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--storage-root", type=Path, default=None)

    args = p.parse_args(argv)
    _apply_storage_root(args)

    if not args.query:
        raise SystemExit("Query text required.")

    try:
        idx_ref = resolve_index_db(
            sha256=args.sha256,
            index_db=str(args.index) if args.index else None,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    hits = retrieve(
        idx_ref,
        args.query,
        final_k=args.final_k,
        candidate_pool=max(args.candidate_pool, args.final_k),
        reranker=args.rerank,
    )

    if args.json:
        rows = []
        for h in hits:
            row = {"chunk_id": h.chunk_id, "rrf_or_rerank_score": h.rrf_score, **h.payload}
            rows.append(row)
        print(json.dumps(rows, indent=2))
    else:
        for h in hits:
            title = " / ".join(h.payload.get("section_path") or []) or "(section)"
            sc = h.payload.get("rerank_score", h.rrf_score)
            print(f"{sc:.4f}  p{h.payload.get('page_start')}-{h.payload.get('page_end')}  {title}")
            print((h.payload.get("text_full") or "")[:400].replace("\n", " "))
            print("---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
