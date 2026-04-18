from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_admin(
    x_admin_api_key: str | None = Header(None, alias="X-Admin-API-Key"),
    authorization: str | None = Header(None),
) -> None:
    """
    Separate from tenant API keys: set ``RAG_ADMIN_API_KEYS`` (comma-separated).
    Accepts ``X-Admin-API-Key`` or ``Authorization: Bearer <key>``.
    """
    raw = (os.environ.get("RAG_ADMIN_API_KEYS") or "").strip()
    if not raw:
        raise HTTPException(
            status_code=403,
            detail="Admin API disabled (set RAG_ADMIN_API_KEYS).",
        )
    allowed = {x.strip() for x in raw.split(",") if x.strip()}
    token = (x_admin_api_key or "").strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token in allowed:
        return
    raise HTTPException(status_code=401, detail="Invalid admin API key")
