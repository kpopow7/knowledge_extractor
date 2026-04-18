from __future__ import annotations

import os


def database_url() -> str | None:
    u = (os.environ.get("DATABASE_URL") or "").strip()
    return u or None


def use_postgres() -> bool:
    return database_url() is not None


def s3_bucket() -> str | None:
    b = (os.environ.get("RAG_S3_BUCKET") or os.environ.get("AWS_S3_BUCKET") or "").strip()
    return b or None


def use_s3_blobs() -> bool:
    return s3_bucket() is not None


def s3_key_prefix() -> str:
    return (os.environ.get("RAG_S3_PREFIX") or "").strip().strip("/")
