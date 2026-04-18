from __future__ import annotations

from slowapi.util import get_remote_address
from starlette.requests import Request


def rate_limit_key(request: Request) -> str:
    """Prefer tenant id when a valid API key is present; otherwise client IP."""
    from rag_api.auth import extract_token_from_request
    from rag_api.tenants import resolve_api_key

    tok = extract_token_from_request(request)
    if tok:
        ctx = resolve_api_key(tok)
        if ctx:
            return f"t:{ctx.tenant_id}"
    return get_remote_address(request)
