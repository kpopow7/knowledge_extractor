from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def storage_root() -> Path:
    override = os.environ.get("RAG_STORAGE_ROOT")
    if override:
        return Path(override).resolve()
    return project_root() / "storage"


def documents_dir() -> Path:
    return storage_root() / "documents"


def registry_db_path() -> Path:
    return storage_root() / "registry.db"


def index_dir() -> Path:
    return storage_root() / "index"
