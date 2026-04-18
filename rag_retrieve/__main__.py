from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_retrieve.pipeline import retrieve
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

    idx_path: Path | None = args.index
    if args.sha256:
        reg = DocumentRegistry()
        sha = _resolve_sha256(reg, args.sha256)
        rec = reg.get(sha)
        if not rec or not rec.index_db_relpath:
            raise SystemExit("No index in registry for this document.")
        idx_path = storage_root() / rec.index_db_relpath
    if idx_path is None:
        raise SystemExit("Provide --index or --sha256.")

    hits = retrieve(
        idx_path,
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
