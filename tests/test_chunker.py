from __future__ import annotations

import unittest

from rag_chunker.chunker import _chunk_sliding_ranges, chunk_artifact
from rag_extractor.models import BoundingBox, ExtractionArtifact, PageExtraction, TextBlock


class TestChunkSliding(unittest.TestCase):
    def test_overlap_twenty_percent(self) -> None:
        text = "x" * 100
        ranges = _chunk_sliding_ranges(text, max_chars=40, overlap_ratio=0.2)
        self.assertGreaterEqual(len(ranges), 2)
        overlap = 8  # 20% of 40
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[0][1], 40)
        self.assertEqual(ranges[1][0], 40 - overlap)


class TestChunkArtifact(unittest.TestCase):
    def test_one_text_block_chunks(self) -> None:
        bb = BoundingBox(x0=0, y0=0, x1=1, y1=1)
        block = TextBlock(block_id="p0001_b0000", page_number=1, bbox=bb, text="Hello world.\n\n" * 50)
        art = ExtractionArtifact(
            schema_version="extraction.v1",
            extraction_version="t",
            extractor_package_version="0",
            document_id="doc1",
            source_filename="t.pdf",
            source_sha256="a" * 64,
            page_count=1,
            pdf_metadata={},
            pages=[PageExtraction(page_number=1, width_pt=100, height_pt=100, blocks=[block])],
        )
        chunks = chunk_artifact(art, max_chunk_chars=80, overlap_ratio=0.2)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.text_embed.startswith("[Source:") for c in chunks))


if __name__ == "__main__":
    unittest.main()
