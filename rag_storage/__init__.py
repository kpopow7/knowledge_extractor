"""Pluggable persistence: local vs S3 blobs, SQLite vs Postgres registry + vector index."""

from rag_storage.env import load_dotenv_from_project

load_dotenv_from_project()

__all__ = ["load_dotenv_from_project"]
