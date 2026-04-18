from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_generate.answer import answer_with_retrieval
from rag_api.registry_resolve import resolve_index_db


def _apply_storage_root(ns: argparse.Namespace) -> None:
    root = getattr(ns, "storage_root", None)
    if root is not None:
        os.environ["RAG_STORAGE_ROOT"] = str(Path(root).resolve())


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

    try:
        idx_ref = resolve_index_db(
            sha256=args.sha256,
            index_db=str(args.index) if args.index else None,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    answer, hits = answer_with_retrieval(
        idx_ref,
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
