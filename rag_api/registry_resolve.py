from __future__ import annotations

from pathlib import Path

from rag_extractor.paths import storage_root
from rag_extractor.registry import DocumentRegistry
from rag_index.targets import SearchIndexTarget
from rag_storage.config import use_postgres


def resolve_sha256(reg: DocumentRegistry, key: str) -> str:
    key = key.strip().lower()
    if len(key) == 64:
        return key
    rows = reg.list_recent(500)
    matches = [r.content_sha256 for r in rows if r.content_sha256.startswith(key)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No document matches prefix {key!r}.")
    raise ValueError(f"Ambiguous prefix {key!r}: {len(matches)} matches.")


def resolve_search_index(sha256_prefix: str) -> SearchIndexTarget:
    reg = DocumentRegistry()
    sha = resolve_sha256(reg, sha256_prefix)
    rec = reg.get(sha)
    if not rec or not rec.index_db_relpath:
        raise ValueError("No index in registry for this document.")
    if use_postgres() and rec.index_db_relpath == "postgres":
        return SearchIndexTarget.from_postgres_document(sha)
    return SearchIndexTarget.from_sqlite_file(storage_root() / rec.index_db_relpath)


def resolve_index_db(*, sha256: str | None, index_db: str | None) -> SearchIndexTarget:
    """
    Resolve search index from registry (``sha256`` prefix/full) or an explicit SQLite file.
    With ``DATABASE_URL`` and a Postgres-backed index, returns a Postgres target.
    """
    if sha256 and index_db:
        raise ValueError("Provide only one of sha256 or index_db.")
    if index_db:
        p = Path(index_db).expanduser().resolve()
        if not p.is_file():
            raise ValueError(f"Index file not found: {p}")
        return SearchIndexTarget.from_sqlite_file(p)
    if sha256:
        return resolve_search_index(sha256)
    raise ValueError("Provide sha256 or index_db.")
