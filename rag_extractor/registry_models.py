from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentRecord:
    content_sha256: str
    original_filename: str
    byte_size: int
    status: str
    extraction_version: str | None
    extractor_package_version: str | None
    schema_version: str | None
    page_count: int | None
    source_relpath: str
    artifact_relpath: str | None
    chunks_relpath: str | None
    chunker_version: str | None
    index_db_relpath: str | None
    embedding_model: str | None
    embedding_dimensions: int | None
    error_message: str | None
    created_at: str
    updated_at: str
