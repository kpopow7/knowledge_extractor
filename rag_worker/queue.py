from __future__ import annotations

import logging
import os
from pathlib import Path

from redis import Redis
from rq import Queue

from rag_api import job_store

log = logging.getLogger(__name__)


def redis_url() -> str | None:
    u = (os.environ.get("REDIS_URL") or "").strip()
    return u or None


def enqueue_document_job(job_id: str, temp_pdf: Path, *, force: bool) -> None:
    """Push pipeline work to RQ (queue ``rag``). Caller must create the job row first."""
    url = redis_url()
    if not url:
        raise RuntimeError("REDIS_URL is not set")
    conn = Redis.from_url(url)
    q = Queue("rag", connection=conn)
    job = q.enqueue(
        "rag_worker.tasks.process_document_job",
        job_id,
        str(temp_pdf.resolve()),
        force,
        job_timeout=3600,
        failure_ttl=86400,
    )
    log.info("enqueued RQ job %s for pipeline job %s", job.id, job_id)
    job_store.set_status(job_id, "queued")


def use_redis_queue() -> bool:
    return redis_url() is not None
