from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)


def configure_prometheus(app: "FastAPI") -> None:
    """
    Expose ``GET /metrics`` in Prometheus text format (HTTP request metrics + process stats).
    Disabled when ``RAG_PROMETHEUS_METRICS`` is ``0`` / ``false`` / ``off``.
    """
    raw = (os.environ.get("RAG_PROMETHEUS_METRICS") or "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        log.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled")
        return

    # Match route name or path (see Instrumentator middleware); reduce noise on static/docs.
    inst = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=[
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/ui",
        ],
    )
    inst.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    log.info("Prometheus metrics at GET /metrics")
