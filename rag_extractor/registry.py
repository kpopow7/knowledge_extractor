from __future__ import annotations

import os
from pathlib import Path

from rag_extractor.registry_models import DocumentRecord
from rag_extractor.registry_sqlite import DocumentRegistrySqlite


class DocumentRegistry:
    """Document registry: SQLite on disk by default, Postgres when ``DATABASE_URL`` is set."""

    def __new__(cls, db_path: Path | None = None):
        if (os.environ.get("DATABASE_URL") or "").strip():
            from rag_extractor.registry_postgres import DocumentRegistryPostgres

            return DocumentRegistryPostgres()
        return DocumentRegistrySqlite(db_path)

__all__ = ["DocumentRecord", "DocumentRegistry"]
