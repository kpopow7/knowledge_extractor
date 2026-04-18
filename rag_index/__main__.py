from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_index import INDEXER_VERSION
from rag_index.build import build_index, load_chunks_jsonl
from rag_index.search import search_hybrid
from rag_index.store import ChunkIndex
from rag_index.store_pg import ChunkIndexPostgres
from rag_index.targets import SearchIndexTarget
from rag_extractor.paths import index_dir, storage_root
from rag_extractor.registry import DocumentRegistry
from rag_storage.config import use_postgres


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
    p = argparse.ArgumentParser(description="Embed chunks and search (Phase 3).")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Embed chunks.jsonl into SQLite + FTS")
    b.add_argument("--chunks", type=Path, help="Path to chunks.jsonl")
    b.add_argument("--sha256", metavar="HASH", help="Use registry chunks_relpath")
    b.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Output SQLite path (default: storage/index/<sha>.sqlite from --sha256)",
    )
    b.add_argument("--embedding-model", default=None, help="Override OPENAI_EMBEDDING_MODEL")
    b.add_argument("--write-registry", action="store_true", help="Update registry index fields")
    b.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if indexer_version + model match registry",
    )
    b.add_argument("--storage-root", type=Path, default=None)

    s = sub.add_parser("search", help="Hybrid vector + FTS (RRF)")
    s.add_argument("--index", type=Path, help="SQLite index path")
    s.add_argument("--sha256", metavar="HASH", help="Resolve index from registry index_db_relpath")
    s.add_argument("query", nargs="?", help="Search query")
    s.add_argument("--top", type=int, default=10, dest="top_k")
    s.add_argument("--json", action="store_true", help="Print JSON array of hits")
    s.add_argument("--storage-root", type=Path, default=None)

    args = p.parse_args(argv)
    _apply_storage_root(args)

    if args.cmd == "build":
        sha: str | None = None
        chunks_path: Path | None = args.chunks
        reg = DocumentRegistry()

        if args.sha256:
            sha = _resolve_sha256(reg, args.sha256)
            rec = reg.get(sha)
            if rec is None:
                raise SystemExit("Document not in registry.")
            if not rec.chunks_relpath:
                raise SystemExit("No chunks.jsonl — run rag_chunker first.")
            if not args.force and rec.index_db_relpath:
                if use_postgres() and rec.index_db_relpath == "postgres":
                    iv = ChunkIndexPostgres(sha).get_meta("indexer_version")
                    if iv == INDEXER_VERSION:
                        print(
                            json.dumps(
                                {
                                    "skipped": True,
                                    "reason": "index_present_same_indexer_version",
                                    "index_path": f"postgres:{sha}",
                                },
                                indent=2,
                            )
                        )
                        return 0
                else:
                    idxp = storage_root() / rec.index_db_relpath
                    if idxp.is_file():
                        iv = ChunkIndex(idxp).get_meta("indexer_version")
                        if iv == INDEXER_VERSION:
                            print(
                                json.dumps(
                                    {
                                        "skipped": True,
                                        "reason": "index_present_same_indexer_version",
                                        "index_path": str(idxp),
                                    },
                                    indent=2,
                                )
                            )
                            return 0

            chunks_path = storage_root() / rec.chunks_relpath

        if chunks_path is None:
            raise SystemExit("Provide --chunks or --sha256.")

        chunks_path = chunks_path.resolve()
        if args.out:
            out_db = args.out.resolve()
        elif sha:
            index_dir().mkdir(parents=True, exist_ok=True)
            out_db = index_dir() / f"{sha}.sqlite"
        else:
            out_db = chunks_path.parent / "index.sqlite"

        if use_postgres() and sha:
            n, model, dims = build_index(
                chunks_path,
                None,
                embedding_model=args.embedding_model,
                clear=True,
                postgres_source_sha256=sha,
            )
            rel = "postgres"
            index_path_display = f"postgres:{sha}"
        else:
            n, model, dims = build_index(
                chunks_path,
                out_db,
                embedding_model=args.embedding_model,
                clear=True,
            )
            rel = out_db.resolve().relative_to(storage_root().resolve()).as_posix()
            index_path_display = str(out_db)

        if args.write_registry:
            if sha is None:
                loaded = load_chunks_jsonl(chunks_path)
                if not loaded:
                    raise SystemExit("--write-registry needs chunks or --sha256.")
                sha = loaded[0].source_sha256
                if reg.get(sha) is None:
                    raise SystemExit("--write-registry: registry has no row for this source_sha256; use --sha256.")
            reg.mark_index(
                content_sha256=sha,
                index_db_relpath=rel,
                embedding_model=model,
                embedding_dimensions=dims,
            )

        print(
            json.dumps(
                {
                    "chunk_count": n,
                    "embedding_model": model,
                    "dimensions": dims,
                    "index_path": index_path_display,
                    "index_relpath": rel,
                    "skipped": False,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "search":
        if not args.query:
            raise SystemExit("Query text required.")
        idx_ref: SearchIndexTarget | Path | None = None
        if args.sha256:
            reg = DocumentRegistry()
            sha = _resolve_sha256(reg, args.sha256)
            rec = reg.get(sha)
            if not rec or not rec.index_db_relpath:
                raise SystemExit("No index in registry for this document.")
            if use_postgres() and rec.index_db_relpath == "postgres":
                idx_ref = SearchIndexTarget.from_postgres_document(sha)
            else:
                idx_ref = SearchIndexTarget.from_sqlite_file(storage_root() / rec.index_db_relpath)
        elif args.index:
            idx_ref = SearchIndexTarget.from_sqlite_file(Path(args.index).resolve())
        if idx_ref is None:
            raise SystemExit("Provide --index or --sha256.")

        hits = search_hybrid(idx_ref, args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps([{"chunk_id": h.chunk_id, "rrf": h.rrf_score, **h.payload} for h in hits], indent=2))
        else:
            for h in hits:
                title = " / ".join(h.payload.get("section_path") or []) or "(section)"
                print(f"{h.rrf_score:.4f}  p{h.payload.get('page_start')}-{h.payload.get('page_end')}  {title}")
                print((h.payload.get("text_full") or "")[:320].replace("\n", " "))
                print("---")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
