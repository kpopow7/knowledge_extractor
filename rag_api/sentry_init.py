from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def init_sentry() -> None:
    dsn = (os.environ.get("SENTRY_DSN") or "").strip()
    if not dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        log.warning("SENTRY_DSN set but sentry-sdk is not installed")
        return

    traces = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    profiles = float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))
    sentry_sdk.init(
        dsn=dsn,
        integrations=[
            FastApiIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        traces_sample_rate=traces,
        profiles_sample_rate=profiles,
        environment=os.environ.get("SENTRY_ENVIRONMENT") or os.environ.get("RAG_ENV", "development"),
    )
