from __future__ import annotations

from fastapi import HTTPException

from rag_api import usage_store
from rag_api.auth import AuthContext


def _limit_for(op: str, auth: AuthContext) -> int | None:
    if auth.anonymous or auth.tenant is None:
        return None
    t = auth.tenant
    if op == "ask":
        return t.asks_per_day
    if op == "retrieve":
        return t.retrieves_per_day
    if op == "ingest":
        return t.ingests_per_day
    return None


def _current_count(op: str, tenant_id: str) -> int:
    asks, retrieves, ingests = usage_store.get_counts(tenant_id)
    if op == "ask":
        return asks
    if op == "retrieve":
        return retrieves
    return ingests


def assert_under_quota(auth: AuthContext, op: str) -> None:
    """Raise 429 if this tenant is already at the daily cap for ``op``."""
    lim = _limit_for(op, auth)
    if lim is None or auth.tenant is None:
        return
    cur = _current_count(op, auth.tenant.tenant_id)
    if cur >= lim:
        raise HTTPException(
            status_code=429,
            detail=f"Daily {op} quota exceeded for tenant {auth.tenant.tenant_id} ({lim}/day).",
        )


def record_success(auth: AuthContext, op: str) -> None:
    """Increment usage after a successful operation."""
    if auth.anonymous or auth.tenant is None:
        return
    field = {"ask": "asks", "retrieve": "retrieves", "ingest": "ingests"}.get(op)
    if not field:
        return
    usage_store.increment(auth.tenant.tenant_id, field)
