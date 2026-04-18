from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rag_chunker import CHUNK_SCHEMA_VERSION, CHUNKER_VERSION, DEFAULT_MAX_CHUNK_CHARS, DEFAULT_OVERLAP_RATIO
from rag_chunker.models import ChunkRecord
from rag_extractor.extract import load_artifact
from rag_extractor.models import ExtractionArtifact, TableBlock, TextBlock

_HEADING_LINE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+.{3,100}"
    r"|[A-Z0-9™®© ,\-\(\)&]{6,100}$"
    r"|[A-Z][^.!?\n]{2,79}$"
    r")$"
)


def _heading_candidate(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or len(s) > 120:
        return False
    if s.endswith(".") and len(s) > 50:
        return False
    return bool(_HEADING_LINE.match(s))


def _push_stack(stack: list[str], title: str, max_depth: int = 6) -> None:
    title = title.strip()
    if not title:
        return
    stack.append(title)
    while len(stack) > max_depth:
        stack.pop(0)


def _strip_leading_headings(text: str, stack: list[str]) -> tuple[str, list[str]]:
    stack = list(stack)
    rest = text.strip()
    for _ in range(12):
        if not rest:
            break
        parts = rest.split("\n", 1)
        first = parts[0].strip()
        remainder = parts[1] if len(parts) > 1 else ""
        if not _heading_candidate(first):
            break
        if not remainder.strip():
            break
        _push_stack(stack, first)
        rest = remainder.strip()
    return rest, stack


def _chunk_sliding_ranges(text: str, max_chars: int, overlap_ratio: float) -> list[tuple[int, int, str]]:
    """Return (start, end, slice) with end exclusive; overlap applied between windows."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    overlap = max(1, int(max_chars * overlap_ratio))
    out: list[tuple[int, int, str]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        out.append((pos, end, text[pos:end]))
        if end >= len(text):
            break
        pos = max(0, end - overlap)
    return out


def _split_table_pieces(md: str, max_chars: int, overlap_ratio: float) -> list[str]:
    """Split pipe tables by row groups; repeat header in each piece; ~overlap_ratio of body rows overlap between pieces."""
    md = md.strip()
    if not md:
        return []
    if len(md) <= max_chars:
        return [md]
    lines = md.split("\n")
    sep_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"[-]{3,}", line) and "|" in line:
            sep_idx = i
            break
    if sep_idx <= 0:
        return [t[2] for t in _chunk_sliding_ranges(md, max_chars, overlap_ratio)]
    head = "\n".join(lines[: sep_idx + 1])
    body_lines = lines[sep_idx + 1 :]
    if not body_lines:
        return [md]
    overlap_rows = max(1, int(len(body_lines) * overlap_ratio))
    parts: list[str] = []
    buf: list[str] = []
    cur_len = len(head)

    def flush() -> None:
        nonlocal buf, cur_len
        if not buf:
            return
        parts.append(head + "\n" + "\n".join(buf))
        keep = buf[-overlap_rows:] if len(buf) > overlap_rows else list(buf)
        buf = keep
        cur_len = len(head) + sum(len(x) + 1 for x in buf)

    for row in body_lines:
        add_len = len(row) + 1
        if buf and cur_len + add_len > max_chars:
            flush()
        if not buf:
            cur_len = len(head)
        buf.append(row)
        cur_len = len(head) + sum(len(x) + 1 for x in buf)
    if buf:
        parts.append(head + "\n" + "\n".join(buf))

    return parts if parts else [md]


def _chunk_id(
    source_sha256: str,
    chunk_index: int,
    page_start: int,
    page_end: int,
    text: str,
) -> str:
    h = hashlib.sha256()
    h.update(CHUNKER_VERSION.encode())
    h.update(b"|")
    h.update(source_sha256.encode())
    h.update(b"|")
    h.update(str(chunk_index).encode())
    h.update(b"|")
    h.update(f"{page_start}:{page_end}".encode())
    h.update(b"|")
    h.update(text[:256].encode())
    return h.hexdigest()[:32]


def _embed_prefix(filename: str, section_path: list[str]) -> str:
    sec = " > ".join(section_path) if section_path else "(no section)"
    return f"[Source: {filename} | Section: {sec}]\n"


def chunk_artifact(
    artifact: ExtractionArtifact,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[ChunkRecord]:
    """
    Chunk by extraction block: long text uses sliding windows with overlap;
    large tables split by row groups with repeated header.
    """
    stack: list[str] = []
    records: list[ChunkRecord] = []
    chunk_index = 0

    for page in artifact.pages:
        for block in page.blocks:
            if isinstance(block, TextBlock):
                body, stack = _strip_leading_headings(block.text, stack)
                if not body.strip():
                    continue
                stripped = body.strip()
                sec = list(stack)
                for c0, c1, piece in _chunk_sliding_ranges(stripped, max_chunk_chars, overlap_ratio):
                    if not piece.strip():
                        continue
                    p0 = p1 = page.page_number
                    cid = _chunk_id(artifact.source_sha256, chunk_index, p0, p1, piece)
                    prefix = _embed_prefix(artifact.source_filename, sec)
                    records.append(
                        ChunkRecord(
                            schema_version=CHUNK_SCHEMA_VERSION,
                            chunker_version=CHUNKER_VERSION,
                            chunk_id=cid,
                            document_id=artifact.document_id,
                            source_sha256=artifact.source_sha256,
                            source_filename=artifact.source_filename,
                            extraction_version=artifact.extraction_version,
                            chunk_index=chunk_index,
                            page_start=p0,
                            page_end=p1,
                            section_path=sec,
                            block_ids=[block.block_id],
                            content_type="prose",
                            text_full=piece,
                            text_embed=prefix + piece,
                            char_start_in_doc=c0,
                            char_end_in_doc=c1,
                        )
                    )
                    chunk_index += 1
            elif isinstance(block, TableBlock):
                md = block.markdown.strip()
                if not md:
                    continue
                sec = list(stack)
                for piece in _split_table_pieces(md, max_chunk_chars, overlap_ratio):
                    if not piece.strip():
                        continue
                    p0 = p1 = page.page_number
                    cid = _chunk_id(artifact.source_sha256, chunk_index, p0, p1, piece)
                    prefix = _embed_prefix(artifact.source_filename, sec)
                    records.append(
                        ChunkRecord(
                            schema_version=CHUNK_SCHEMA_VERSION,
                            chunker_version=CHUNKER_VERSION,
                            chunk_id=cid,
                            document_id=artifact.document_id,
                            source_sha256=artifact.source_sha256,
                            source_filename=artifact.source_filename,
                            extraction_version=artifact.extraction_version,
                            chunk_index=chunk_index,
                            page_start=p0,
                            page_end=p1,
                            section_path=sec,
                            block_ids=[block.block_id],
                            content_type="table",
                            text_full=piece,
                            text_embed=prefix + piece,
                            char_start_in_doc=0,
                            char_end_in_doc=len(piece),
                        )
                    )
                    chunk_index += 1

    return records


def chunk_from_path(
    artifact_path: str | Path,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> tuple[list[ChunkRecord], ExtractionArtifact]:
    art = load_artifact(artifact_path)
    return chunk_artifact(art, max_chunk_chars=max_chunk_chars, overlap_ratio=overlap_ratio), art


def write_jsonl(chunks: list[ChunkRecord], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [c.model_dump_json() for c in chunks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_manifest(
    chunks: list[ChunkRecord],
    artifact: ExtractionArtifact,
    out_path: str | Path,
    *,
    max_chunk_chars: int,
    overlap_ratio: float,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "document_id": artifact.document_id,
        "source_sha256": artifact.source_sha256,
        "source_filename": artifact.source_filename,
        "extraction_version": artifact.extraction_version,
        "chunk_count": len(chunks),
        "max_chunk_chars": max_chunk_chars,
        "overlap_ratio": overlap_ratio,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
