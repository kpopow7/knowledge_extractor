"""Phase 2: chunk extraction artifacts for embedding / retrieval."""

__version__ = "0.1.0"

CHUNKER_VERSION = "2026.04.17"
CHUNK_SCHEMA_VERSION = "chunk.v1"

# Defaults tuned for technical manuals (adjust per embedding model context window)
DEFAULT_MAX_CHUNK_CHARS = 2500
DEFAULT_OVERLAP_RATIO = 0.2
