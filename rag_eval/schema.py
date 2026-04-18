from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One line in eval JSONL."""

    id: str
    question: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    """If non-empty, a hit is relevant when ``chunk_id`` matches."""
    gold_pages: list[int] = Field(default_factory=list)
    """If non-empty, a hit is relevant when ``page_start <= p <= page_end`` for any ``p``."""
    gold_substrings: list[str] = Field(default_factory=list)
    """If non-empty, a hit is relevant when any substring appears in ``text_full`` (case-insensitive)."""

    def relevance(self, chunk_id: str, text_full: str, page_start: int, page_end: int) -> bool:
        if self.gold_chunk_ids and chunk_id in self.gold_chunk_ids:
            return True
        if self.gold_pages:
            for p in self.gold_pages:
                if page_start <= p <= page_end:
                    return True
        if self.gold_substrings:
            low = text_full.lower()
            if any(s.lower() in low for s in self.gold_substrings if s):
                return True
        return False
