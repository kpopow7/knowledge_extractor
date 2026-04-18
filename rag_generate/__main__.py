from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_generate.answer import answer_with_retrieval
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
    p = argparse.ArgumentParser(description="Grounded LLM answers using rag_retrieve context.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="Retrieve + chat completion")
    ask.add_argument("--index", type=Path, help="SQLite index path")
    ask.add_argument("--sha256", metavar="HASH", help="Resolve index from registry")
    ask.add_argument("-q", "--question", required=True, help="User question")
    ask.add_argument("--model", default=None, help="Chat model (default OPENAI_CHAT_MODEL or gpt-4o-mini)")
    ask.add_argument("--top", type=int, default=8, dest="final_k")
    ask.add_argument("--candidates", type=int, default=40, dest="candidate_pool")
    ask.add_argument("--rerank", default="none", help="none | cohere | cross-encoder")
    ask.add_argument("--json", action="store_true", help="Print answer + chunk_ids JSON")
    ask.add_argument("--storage-root", type=Path, default=None)

    args = p.parse_args(argv)
    _apply_storage_root(args)

    if args.cmd != "ask":
        return 1

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

    answer, hits = answer_with_retrieval(
        idx_path,
        args.question,
        chat_model=args.model,
        final_k=args.final_k,
        candidate_pool=max(args.candidate_pool, args.final_k),
        reranker=args.rerank,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "answer": answer,
                    "chunk_ids": [h.chunk_id for h in hits],
                    "pages": [
                        (h.payload.get("page_start"), h.payload.get("page_end")) for h in hits
                    ],
                },
                indent=2,
            )
        )
    else:
        print(answer)
        if hits:
            print("\n--- sources ---")
            for h in hits:
                pl = h.payload
                print(
                    f"- {pl.get('source_filename')} p.{pl.get('page_start')}-{pl.get('page_end')} ({h.chunk_id})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
