from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """PDF coordinate space: origin top-left; units in points (1/72 inch)."""

    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    block_id: str
    type: Literal["text"] = "text"
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    text: str


class TableBlock(BaseModel):
    block_id: str
    type: Literal["table"] = "table"
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    rows: list[list[str]]
    markdown: str


class ImageBlock(BaseModel):
    block_id: str
    type: Literal["image"] = "image"
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    xref: int | None = None


ContentBlock = TextBlock | TableBlock | ImageBlock


class PageExtraction(BaseModel):
    page_number: int = Field(ge=1)
    width_pt: float
    height_pt: float
    blocks: list[ContentBlock]


class ExtractionArtifact(BaseModel):
    """Canonical Phase-1 artifact: full structured IR + provenance for reprocessing."""

    schema_version: str
    extraction_version: str
    extractor_package_version: str
    document_id: str
    source_filename: str
    source_sha256: str
    page_count: int
    pdf_metadata: dict[str, str | None]
    pages: list[PageExtraction]
    warnings: list[str] = Field(default_factory=list)
