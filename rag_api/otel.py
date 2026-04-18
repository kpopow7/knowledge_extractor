from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastapi import FastAPI


def configure_opentelemetry(app: "FastAPI") -> None:
    """
    Optional OTLP traces when ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or traces-specific) is set.
    Skips if packages are missing or ``OTEL_SDK_DISABLED=true``.
    """
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() in ("true", "1"):
        return
    endpoint = (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip() or (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or ""
    ).strip()
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        log.warning("OTEL endpoint set but OpenTelemetry packages are not installed")
        return

    service_name = (os.environ.get("OTEL_SERVICE_NAME") or "rag-api").strip()
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=os.environ.get("OTEL_FASTAPI_EXCLUDED_URLS", "health,metrics"),
    )
    log.info("OpenTelemetry tracing enabled (endpoint=%s)", endpoint)
