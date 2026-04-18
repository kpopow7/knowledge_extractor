from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """One retrievable unit produced from Phase 1 IR."""

    schema_version: str = Field(default="chunk.v1")
    chunker_version: str
    chunk_id: str
    document_id: str
    source_sha256: str
    source_filename: str
    extraction_version: str
    chunk_index: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    content_type: Literal["prose", "table", "mixed"] = "prose"
    text_full: str
    text_embed: str = ""
    char_start_in_doc: int = Field(
        ge=0,
        description="For prose: half-open offsets within the parent text block after heading strip; for tables: local piece offsets.",
    )
    char_end_in_doc: int = Field(ge=0)
