from __future__ import annotations

import logging
import os
import sys

from rag_api.request_context import request_id_var


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_var.get()
        record.request_id = rid if rid else "-"
        return True


def configure_logging() -> None:
    level_name = (os.environ.get("RAG_API_LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(level)
        logging.getLogger(name).propagate = False

    logging.getLogger("httpx").setLevel(logging.WARNING)
