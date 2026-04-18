from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from rag_extractor.extract import extract_pdf, write_artifact
from rag_extractor.ingest import ingest_pdf
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
    p = argparse.ArgumentParser(description="Extract structured content from digital PDFs (Phase 1 IR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="Run extraction and write JSON artifact (no registry)")
    ex.add_argument("pdf", type=Path, help="Path to PDF file")
    ex.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Output JSON path (default: storage/artifacts/<stem>_extraction.json)",
    )
    ex.add_argument("--document-id", dest="document_id", default=None, help="Override document_id in JSON")
    ex.add_argument("--storage-root", type=Path, default=None, help="Override storage root (env RAG_STORAGE_ROOT)")

    ing = sub.add_parser("ingest", help="Register PDF: copy to storage, extract, SQLite registry (idempotent)")
    ing.add_argument("pdf", type=Path, help="Path to PDF file")
    ing.add_argument("--force", action="store_true", help="Re-run extraction even if hash+version already ingested")
    ing.add_argument("--storage-root", type=Path, default=None, help="Override storage root (env RAG_STORAGE_ROOT)")

    lst = sub.add_parser("list", help="List documents in the registry")
    lst.add_argument("--limit", type=int, default=50)
    lst.add_argument("--json", action="store_true", help="Print as JSON array")
    lst.add_argument("--storage-root", type=Path, default=None, help="Override storage root")

    sh = sub.add_parser("show", help="Show one document record by content SHA-256 (or unique prefix)")
    sh.add_argument("content_sha256", help="Full 64-char hex hash or unique prefix")
    sh.add_argument("--json", action="store_true")
    sh.add_argument("--storage-root", type=Path, default=None, help="Override storage root")

    ch = sub.add_parser(
        "chunk",
        help="Phase 2 chunking (forwards to rag_chunker run); e.g. chunk --sha256 <hash> --write-registry",
    )
    ch.add_argument("chunk_args", nargs=argparse.REMAINDER, help="Extra args for rag_chunker run")

    ix = sub.add_parser(
        "index",
        help="Phase 3 embed + index (forwards to rag_index: build | search …)",
    )
    ix.add_argument("index_args", nargs=argparse.REMAINDER, help="Subcommand and flags for rag_index")

    rt = sub.add_parser(
        "retrieve",
        help="Phase 4 hybrid pool + rerank (forwards to rag_retrieve)",
    )
    rt.add_argument("retrieve_args", nargs=argparse.REMAINDER, help="Flags for rag_retrieve")

    ev = sub.add_parser("eval", help="Retrieval eval JSONL (forwards to rag_eval)")
    ev.add_argument("eval_args", nargs=argparse.REMAINDER, help="Flags for rag_eval")

    ask = sub.add_parser("ask", help="LLM answer with retrieved context (forwards to rag_generate)")
    ask.add_argument("ask_args", nargs=argparse.REMAINDER, help="Flags for rag_generate")

    args = p.parse_args(argv)
    _apply_storage_root(args)

    if args.cmd == "chunk":
        from rag_chunker.__main__ import main as chunk_main

        extra = list(getattr(args, "chunk_args", []) or [])
        if not extra or extra[0] != "run":
            extra = ["run"] + extra
        return chunk_main(extra)

    if args.cmd == "index":
        from rag_index.__main__ import main as index_main

        extra = list(getattr(args, "index_args", []) or [])
        if not extra:
            extra = ["-h"]
        return index_main(extra)

    if args.cmd == "retrieve":
        from rag_retrieve.__main__ import main as retrieve_main

        extra = list(getattr(args, "retrieve_args", []) or [])
        if not extra:
            extra = ["-h"]
        return retrieve_main(extra)

    if args.cmd == "eval":
        from rag_eval.__main__ import main as eval_main

        extra = list(getattr(args, "eval_args", []) or [])
        if not extra or extra[0] != "run":
            extra = ["run"] + extra
        if extra == ["run"]:
            extra = ["run", "-h"]
        return eval_main(extra)

    if args.cmd == "ask":
        from rag_generate.__main__ import main as ask_main

        extra = list(getattr(args, "ask_args", []) or [])
        if not extra or extra[0] != "ask":
            extra = ["ask"] + extra
        if extra == ["ask"]:
            extra = ["ask", "-h"]
        return ask_main(extra)

    if args.cmd == "extract":
        pdf = args.pdf.resolve()
        out = args.out
        if out is None:
            out = storage_root() / "artifacts" / f"{pdf.stem}_extraction.json"
        artifact = extract_pdf(pdf, document_id=args.document_id)
        written = write_artifact(artifact, out)
        print(written)
        return 0

    if args.cmd == "ingest":
        result = ingest_pdf(args.pdf, force=args.force)
        root = storage_root()
        payload = {
            "content_sha256": result.content_sha256,
            "skipped": result.skipped,
            "reason": result.reason,
            "status": result.status,
            "source_path": result.source_path.resolve().as_posix(),
            "artifact_path": result.artifact_path.resolve().as_posix() if result.artifact_path else None,
            "storage_root": root.resolve().as_posix(),
        }
        print(json.dumps(payload, indent=2))
        return 0 if result.status == "ready" else 1

    reg = DocumentRegistry()

    if args.cmd == "list":
        rows = reg.list_recent(limit=args.limit)
        if args.json:
            print(json.dumps([asdict(r) for r in rows], indent=2))
            return 0
        print(f"{'STATUS':<8} {'PAGES':>5} {'SHA256':<18} {'FILENAME'}")
        for r in rows:
            short = r.content_sha256[:16] + "…"
            pages = str(r.page_count) if r.page_count is not None else "-"
            print(f"{r.status:<8} {pages:>5} {short:<18} {r.original_filename}")
        return 0

    if args.cmd == "show":
        sha = _resolve_sha256(reg, args.content_sha256)
        rec = reg.get(sha)
        if rec is None:
            raise SystemExit("Document not found.")
        root = storage_root()
        data = asdict(rec)
        data["source_abspath"] = str(root / rec.source_relpath) if rec.source_relpath else None
        data["artifact_abspath"] = str(root / rec.artifact_relpath) if rec.artifact_relpath else None
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            for k, v in data.items():
                print(f"{k}: {v}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
