from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SearchIndexTarget:
    """Where hybrid search loads vectors + text index from."""

    kind: Literal["sqlite", "postgres"]
    sqlite_path: Path | None = None
    source_sha256: str | None = None

    @staticmethod
    def from_sqlite_file(path: Path) -> SearchIndexTarget:
        return SearchIndexTarget(kind="sqlite", sqlite_path=path.resolve())

    @staticmethod
    def from_postgres_document(source_sha256: str) -> SearchIndexTarget:
        return SearchIndexTarget(kind="postgres", source_sha256=source_sha256)
