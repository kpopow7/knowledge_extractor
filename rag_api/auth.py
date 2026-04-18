from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from rag_api.tenants import TenantContext, auth_configured, resolve_api_key


@dataclass
class AuthContext:
    """Resolved caller: open (anonymous) or authenticated tenant."""

    tenant: TenantContext | None = None
    anonymous: bool = False


def extract_token(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> str | None:
    token = (x_api_key or "").strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    return token or None


def extract_token_from_request(request: Request) -> str | None:
    """Same resolution as headers, for rate-limit keying."""
    x = (request.headers.get("X-API-Key") or "").strip()
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        t = auth[7:].strip()
        if t:
            return t
    return x or None


def require_auth(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> AuthContext:
    """
    If no API keys / tenant file is configured, allow anonymous access (single-tenant dev).
    Otherwise require ``X-API-Key`` or ``Authorization: Bearer`` matching ``RAG_API_KEYS``
    and/or ``RAG_TENANTS_FILE``.
    """
    if not auth_configured():
        return AuthContext(tenant=None, anonymous=True)

    token = extract_token(x_api_key, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    ctx = resolve_api_key(token)
    if ctx is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return AuthContext(tenant=ctx, anonymous=False)


def require_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> AuthContext:
    """Backward-compatible name: validates auth and returns tenant context."""
    return require_auth(x_api_key, authorization)
