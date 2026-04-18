from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TenantContext:
    """Resolved tenant for quota + rate-limit grouping."""

    tenant_id: str
    asks_per_day: int | None = None
    retrieves_per_day: int | None = None
    ingests_per_day: int | None = None


# key -> TenantContext
_by_api_key: dict[str, TenantContext] | None = None
_cache_sig: object | None = None


def _current_signature() -> tuple:
    path = (os.environ.get("RAG_TENANTS_FILE") or "").strip()
    mtime: float | None = None
    if path:
        p = Path(path)
        if p.is_file():
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = None
    keys = (os.environ.get("RAG_API_KEYS") or "").strip()
    dq = (
        (os.environ.get("RAG_DEFAULT_QUOTA_ASKS_PER_DAY") or "").strip(),
        (os.environ.get("RAG_DEFAULT_QUOTA_RETRIEVES_PER_DAY") or "").strip(),
        (os.environ.get("RAG_DEFAULT_QUOTA_INGESTS_PER_DAY") or "").strip(),
    )
    return (path, mtime, keys, dq)


def _parse_default_quotas() -> TenantContext:
    def _i(name: str) -> int | None:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return None
        try:
            v = int(raw, 10)
            return v if v >= 0 else None
        except ValueError:
            return None

    return TenantContext(
        tenant_id="default",
        asks_per_day=_i("RAG_DEFAULT_QUOTA_ASKS_PER_DAY"),
        retrieves_per_day=_i("RAG_DEFAULT_QUOTA_RETRIEVES_PER_DAY"),
        ingests_per_day=_i("RAG_DEFAULT_QUOTA_INGESTS_PER_DAY"),
    )


def _quotas_from_obj(q: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(q, dict):
        return None, None, None

    def one(k: str) -> int | None:
        v = q.get(k)
        if v is None:
            return None
        try:
            n = int(v)
            return n if n >= 0 else None
        except (TypeError, ValueError):
            return None

    return (
        one("asks_per_day"),
        one("retrieves_per_day"),
        one("ingests_per_day"),
    )


def _load_tenants_file(path: Path) -> dict[str, TenantContext]:
    out: dict[str, TenantContext] = {}
    data = json.loads(path.read_text(encoding="utf-8"))
    tenants = data.get("tenants") if isinstance(data, dict) else None
    if not isinstance(tenants, list):
        return out
    for row in tenants:
        if not isinstance(row, dict):
            continue
        tid = (row.get("tenant_id") or "").strip()
        if not tid:
            continue
        keys = row.get("api_keys")
        if not isinstance(keys, list):
            continue
        qobj = row.get("quotas") or row.get("quota")
        a, r, i = _quotas_from_obj(qobj if isinstance(qobj, dict) else {})
        ctx = TenantContext(tenant_id=tid, asks_per_day=a, retrieves_per_day=r, ingests_per_day=i)
        for k in keys:
            if isinstance(k, str) and k.strip():
                out[k.strip()] = ctx
    return out


def reload_registry() -> None:
    """Load ``RAG_TENANTS_FILE`` (if set) plus implicit default-tenant quotas from env."""
    global _by_api_key
    path_raw = (os.environ.get("RAG_TENANTS_FILE") or "").strip()
    default_ctx = _parse_default_quotas()
    merged: dict[str, TenantContext] = {}

    if path_raw:
        path = Path(path_raw)
        if path.is_file():
            try:
                merged.update(_load_tenants_file(path))
            except (OSError, json.JSONDecodeError, ValueError) as e:
                log.warning("Could not read RAG_TENANTS_FILE %s: %s", path, e)
        else:
            log.warning("RAG_TENANTS_FILE set but not found: %s", path)

    raw_keys = (os.environ.get("RAG_API_KEYS") or "").strip()
    if raw_keys:
        allowed = {x.strip() for x in raw_keys.split(",") if x.strip()}
        for k in allowed:
            if k not in merged:
                merged[k] = default_ctx

    _by_api_key = merged


def registry() -> dict[str, TenantContext]:
    """Return key → tenant map (reload when env or tenant file changes)."""
    global _by_api_key, _cache_sig
    sig = _current_signature()
    if _by_api_key is None or sig != _cache_sig:
        reload_registry()
        _cache_sig = sig
    assert _by_api_key is not None
    return _by_api_key


def resolve_api_key(token: str) -> TenantContext | None:
    """Return tenant for this API key, or ``None`` if unknown."""
    reg = registry()
    return reg.get(token)


def auth_configured() -> bool:
    """True if any API key is required (flat list and/or tenant file keys)."""
    if (os.environ.get("RAG_TENANTS_FILE") or "").strip():
        return True
    return bool((os.environ.get("RAG_API_KEYS") or "").strip())


def any_quota_enabled(ctx: TenantContext) -> bool:
    return any(
        x is not None
        for x in (ctx.asks_per_day, ctx.retrieves_per_day, ctx.ingests_per_day)
    )
