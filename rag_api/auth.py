from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> None:
    """
    If ``RAG_API_KEYS`` is set to a non-empty comma-separated list, require a matching
    ``X-API-Key`` header or ``Authorization: Bearer <key>``. If unset or empty, no check.
    """
    raw = (os.environ.get("RAG_API_KEYS") or "").strip()
    if not raw:
        return
    allowed = {x.strip() for x in raw.split(",") if x.strip()}
    token = (x_api_key or "").strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token in allowed:
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key")
