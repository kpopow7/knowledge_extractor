from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz

fitz.no_recommend_layout()

from rag_extractor import EXTRACTION_VERSION, SCHEMA_VERSION, __version__
from rag_extractor.models import (
    BoundingBox,
    ExtractionArtifact,
    ImageBlock,
    PageExtraction,
    TableBlock,
    TextBlock,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rect_from_tuple(t: tuple[float, float, float, float]) -> fitz.Rect:
    return fitz.Rect(t[0], t[1], t[2], t[3])


def _bbox_from_rect(r: fitz.Rect) -> BoundingBox:
    return BoundingBox(x0=float(r.x0), y0=float(r.y0), x1=float(r.x1), y1=float(r.y1))


def _block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        parts: list[str] = []
        for span in line.get("spans", []):
            parts.append(span.get("text") or "")
        line_text = "".join(parts).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def _normalize_table_rows(raw: list[list[str | None]] | None) -> list[list[str]]:
    if not raw:
        return []
    rows: list[list[str]] = []
    for row in raw:
        cells = [(c or "").strip() if c is not None else "" for c in row]
        rows.append(cells)
    while rows and not any(c for c in rows[-1]):
        rows.pop()
    return rows


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    out: list[str] = []
    for i, row in enumerate(padded):
        out.append("|" + "|".join(row) + "|")
        if i == 0:
            out.append("|" + "|".join(["---"] * width) + "|")
    return "\n".join(out) + "\n"


def _overlap_ratio(inner: fitz.Rect, outer: fitz.Rect) -> float:
    inter = inner & outer
    if inter.is_empty:
        return 0.0
    area_i = inter.get_area()
    area_in = inner.get_area()
    if area_in <= 0:
        return 0.0
    return area_i / area_in


def _text_block_overlaps_tables(bbox: fitz.Rect, table_rects: list[fitz.Rect], threshold: float = 0.45) -> bool:
    for tr in table_rects:
        if _overlap_ratio(bbox, tr) >= threshold:
            return True
    return False


def extract_pdf(
    pdf_path: str | Path,
    *,
    document_id: str | None = None,
) -> ExtractionArtifact:
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))

    source_sha256 = _sha256_file(path)
    doc_id = document_id or source_sha256[:16]

    doc = fitz.open(path)
    warnings: list[str] = []

    meta: dict[str, str | None] = {}
    try:
        raw = doc.metadata or {}
        for k in ("title", "author", "subject", "keywords", "creator", "producer", "creationDate", "modDate"):
            v = raw.get(k)
            meta[k] = v if isinstance(v, str) or v is None else str(v)
    except Exception as e:  # noqa: BLE001 — surface as warning, continue
        warnings.append(f"metadata_read:{e!s}")

    pages_out: list[PageExtraction] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        w, h = page.rect.width, page.rect.height

        table_finder = page.find_tables()
        table_rects = [fitz.Rect(t.bbox) for t in table_finder.tables]

        merged: list[tuple[str, int, fitz.Rect, object]] = []

        for ti, table in enumerate(table_finder.tables):
            bbox = fitz.Rect(table.bbox)
            rows = _normalize_table_rows(table.extract())
            md = _rows_to_markdown(rows)
            merged.append(("table", ti, bbox, (rows, md)))

        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        text_i = 0
        for block in text_dict.get("blocks", []):
            btype = block.get("type", 0)
            if btype == 1:
                bbox = _rect_from_tuple(block["bbox"])
                xref = block.get("xref")
                merged.append(("image", text_i, bbox, int(xref) if xref is not None else None))
                text_i += 1
                continue
            if btype != 0:
                continue

            bbox = _rect_from_tuple(block["bbox"])
            if _text_block_overlaps_tables(bbox, table_rects):
                continue
            text = _block_text(block)
            if not text:
                continue
            merged.append(("text", text_i, bbox, text))
            text_i += 1

        merged.sort(key=lambda x: (x[2].y0, x[2].x0))

        blocks: list = []
        seq = 0
        for kind, _idx, rect, payload in merged:
            bid = f"p{page_number:04d}_b{seq:04d}"
            seq += 1
            if kind == "text":
                blocks.append(
                    TextBlock(
                        block_id=bid,
                        page_number=page_number,
                        bbox=_bbox_from_rect(rect),
                        text=payload,  # type: ignore[arg-type]
                    )
                )
            elif kind == "table":
                rows, md = payload  # type: ignore[misc]
                blocks.append(
                    TableBlock(
                        block_id=bid,
                        page_number=page_number,
                        bbox=_bbox_from_rect(rect),
                        rows=rows,
                        markdown=md or "",
                    )
                )
            elif kind == "image":
                blocks.append(
                    ImageBlock(
                        block_id=bid,
                        page_number=page_number,
                        bbox=_bbox_from_rect(rect),
                        xref=payload,  # type: ignore[arg-type]
                    )
                )

        pages_out.append(
            PageExtraction(
                page_number=page_number,
                width_pt=float(w),
                height_pt=float(h),
                blocks=blocks,
            )
        )

    doc.close()

    return ExtractionArtifact(
        schema_version=SCHEMA_VERSION,
        extraction_version=EXTRACTION_VERSION,
        extractor_package_version=__version__,
        document_id=doc_id,
        source_filename=path.name,
        source_sha256=source_sha256,
        page_count=len(pages_out),
        pdf_metadata=meta,
        pages=pages_out,
        warnings=warnings,
    )


def write_artifact(artifact: ExtractionArtifact, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return out


def extract_to_json(pdf_path: str | Path, out_json: str | Path, **kwargs: object) -> Path:
    art = extract_pdf(pdf_path, **kwargs)  # type: ignore[arg-type]
    return write_artifact(art, out_json)


def load_artifact(path: str | Path) -> ExtractionArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExtractionArtifact.model_validate(data)
