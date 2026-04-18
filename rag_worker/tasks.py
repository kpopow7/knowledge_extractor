from __future__ import annotations

import logging
from pathlib import Path

from rag_worker.pipeline import run_document_pipeline

log = logging.getLogger(__name__)


def process_document_job(job_id: str, temp_pdf: str, force: bool) -> None:
    """RQ worker entrypoint (import path ``rag_worker.tasks.process_document_job``)."""
    run_document_pipeline(job_id, Path(temp_pdf), force=force)
