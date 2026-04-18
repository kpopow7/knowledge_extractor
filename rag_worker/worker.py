"""Run: ``python -m rag_worker.worker`` (requires ``REDIS_URL``)."""

from __future__ import annotations

import os

import rag_storage  # noqa: F401 — loads .env before REDIS_URL / logging env
import sys

from redis import Redis
from rq import Queue, Worker

from rag_api.logging_config import configure_logging


def main() -> int:
    configure_logging()
    url = (os.environ.get("REDIS_URL") or "").strip()
    if not url:
        print("REDIS_URL is not set.", file=sys.stderr)
        return 1
    redis_conn = Redis.from_url(url)
    queue = Queue("rag", connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
