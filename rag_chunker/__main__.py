from __future__ import annotations

import argparse

import rag_storage  # noqa: F401 — loads .env for RAG_STORAGE_ROOT / DATABASE_URL
import json
import os
import sys
from pathlib import Path

from rag_chunker import CHUNKER_VERSION
from rag_chunker.chunker import chunk_from_path, write_jsonl, write_manifest
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
    p = argparse.ArgumentParser(description="Chunk Phase 1 extraction artifacts (Phase 2).")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Emit chunks.jsonl (+ manifest) from extraction.json")
    run.add_argument("--artifact", type=Path, help="Path to extraction.json")
    run.add_argument(
        "--sha256",
        dest="content_sha256",
        metavar="HASH",
        help="Use registry to resolve extraction.json under storage",
    )
    run.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Output chunks.jsonl path (default: next to artifact or documents/<sha>/chunks.jsonl)",
    )
    run.add_argument("--manifest", type=Path, help="Write manifest JSON (default: beside chunks with _manifest suffix)")
    run.add_argument("--max-chars", type=int, default=None, help="Max characters per chunk window")
    run.add_argument("--overlap", type=float, default=None, help="Overlap ratio (e.g. 0.2 for 20%%)")
    run.add_argument(
        "--write-registry",
        action="store_true",
        help="Update SQLite registry with chunks_relpath (requires --sha256 or resolvable path)",
    )
    run.add_argument("--force", action="store_true", help="Re-emit even if chunker_version already matches")
    run.add_argument("--storage-root", type=Path, default=None, help="Override RAG_STORAGE_ROOT")

    args = p.parse_args(argv)
    _apply_storage_root(args)

    if args.cmd != "run":
        return 1

    from rag_chunker import DEFAULT_MAX_CHUNK_CHARS, DEFAULT_OVERLAP_RATIO

    max_c = args.max_chars if args.max_chars is not None else DEFAULT_MAX_CHUNK_CHARS
    ov = args.overlap if args.overlap is not None else DEFAULT_OVERLAP_RATIO

    artifact_path: Path | None = args.artifact
    sha: str | None = args.content_sha256
    reg = DocumentRegistry()

    if artifact_path and sha:
        raise SystemExit("Use only one of --artifact or --sha256.")

    if sha:
        sha = _resolve_sha256(reg, sha)
        rec = reg.get(sha)
        if rec is None:
            raise SystemExit("Document not in registry.")
        if rec.status != "ready" or not rec.artifact_relpath:
            raise SystemExit("Document has no ready extraction artifact.")
        if (
            rec.chunks_relpath
            and rec.chunker_version == CHUNKER_VERSION
            and not args.force
        ):
            root = storage_root()
            out = {
                "skipped": True,
                "reason": "already_chunked_same_version",
                "content_sha256": sha,
                "chunks_relpath": rec.chunks_relpath,
                "chunker_version": rec.chunker_version,
                "chunks_path": str(root / rec.chunks_relpath) if rec.chunks_relpath else None,
            }
            print(json.dumps(out, indent=2))
            return 0
        artifact_path = storage_root() / rec.artifact_relpath

    if artifact_path is None:
        raise SystemExit("Provide --artifact path/to/extraction.json or --sha256 <hash>.")

    artifact_path = artifact_path.resolve()
    chunks, art = chunk_from_path(artifact_path, max_chunk_chars=max_c, overlap_ratio=ov)

    if args.out:
        jsonl_path = args.out.resolve()
    elif sha:
        jsonl_path = storage_root() / "documents" / sha / "chunks.jsonl"
    else:
        jsonl_path = artifact_path.parent / "chunks.jsonl"

    write_jsonl(chunks, jsonl_path)
    man_path = args.manifest
    if man_path is None:
        man_path = jsonl_path.with_name(jsonl_path.stem + "_manifest.json")
    else:
        man_path = Path(args.manifest).resolve()
    write_manifest(chunks, art, man_path, max_chunk_chars=max_c, overlap_ratio=ov)

    if args.write_registry:
        doc_key = sha if sha is not None else art.source_sha256
        if reg.get(doc_key) is None:
            raise SystemExit("--write-registry requires a registry row for this document (run ingest first).")
        try:
            rel = jsonl_path.resolve().relative_to(storage_root().resolve()).as_posix()
        except ValueError as e:
            raise SystemExit(
                "--write-registry requires --out (or default paths) under RAG_STORAGE_ROOT."
            ) from e
        reg.mark_chunks(content_sha256=doc_key, chunks_relpath=rel, chunker_version=CHUNKER_VERSION)

    out = {
        "chunk_count": len(chunks),
        "chunks_path": str(jsonl_path),
        "manifest_path": str(man_path),
        "chunker_version": CHUNKER_VERSION,
        "content_sha256": art.source_sha256,
        "skipped": False,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
