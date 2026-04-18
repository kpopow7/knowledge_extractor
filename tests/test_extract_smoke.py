import unittest
from pathlib import Path

from rag_extractor.extract import extract_pdf


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "pdfs" / "MPM_PS_US_MAR2026_02252026.pdf"


class TestExtractSmoke(unittest.TestCase):
    @unittest.skipUnless(FIXTURE.is_file(), "fixture PDF missing")
    def test_fixture_pages_and_tables(self) -> None:
        art = extract_pdf(FIXTURE)
        self.assertEqual(art.page_count, 36)
        self.assertEqual(art.source_filename, FIXTURE.name)
        self.assertTrue(art.source_sha256)
        table_pages = sum(
            1 for p in art.pages for b in p.blocks if getattr(b, "type", None) == "table"
        )
        self.assertGreater(table_pages, 0)


if __name__ == "__main__":
    unittest.main()
