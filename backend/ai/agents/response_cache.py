"""Two-tier cache for grounded LLM answers.

L1 = in-process LRU (fast, per-instance). L2 = Supabase ``llm_response_cache``
(survives restarts, shared across instances). Keyed by a caller-supplied string
— include the date so keys expire daily. Best-effort: any L2 failure degrades
to L1-only so a DB hiccup never breaks a feature.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_L1: "OrderedDict[str, tuple]" = OrderedDict()   # key -> (expiry_monotonic, payload)
_L1_MAX = 512


def seconds_to_ist_eod(min_seconds: int = 300) -> float:
    """Seconds until 23:59:59 IST today — daily cache keys expire at IST EOD."""
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    eod = now.replace(hour=23, minute=59, second=59, microsecond=0)
    return max(float(min_seconds), (eod - now).total_seconds())


def _sb():
    try:
        from ...api.app import get_supabase_admin
        return get_supabase_admin()
    except Exception:  # noqa: BLE001
        return None


def _l1_get(key: str) -> Optional[Dict[str, Any]]:
    hit = _L1.get(key)
    if not hit:
        return None
    expiry, payload = hit
    if time.monotonic() >= expiry:
        _L1.pop(key, None)
        return None
    _L1.move_to_end(key)
    return payload


def _l1_set(key: str, payload: Dict[str, Any], ttl_seconds: float) -> None:
    _L1[key] = (time.monotonic() + ttl_seconds, payload)
    _L1.move_to_end(key)
    while len(_L1) > _L1_MAX:
        _L1.popitem(last=False)


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    """Return the cached payload dict or None. Checks L1 then L2."""
    hit = _l1_get(key)
    if hit is not None:
        return hit
    sb = _sb()
    if sb is None:
        return None
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = (
            sb.table("llm_response_cache")
            .select("payload")
            .eq("cache_key", key)
            .gt("expires_at", now)
            .limit(1)
            .execute()
        )
        data = rows.data or []
        if data:
            payload = data[0].get("payload")
            if isinstance(payload, dict):
                _l1_set(key, payload, 3600)   # short L1 TTL; L2 is the source of truth
                return payload
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm_response_cache L2 get failed: %s", exc)
    return None


def cache_set(key: str, payload: Dict[str, Any], *, ttl_seconds: float,
              surface: str = "", model: str = "") -> None:
    """Write to L1 and best-effort upsert L2."""
    _l1_set(key, payload, ttl_seconds)
    sb = _sb()
    if sb is None:
        return
    try:
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(seconds=ttl_seconds)).isoformat()
        sb.table("llm_response_cache").upsert({
            "cache_key": key,
            "surface": surface,
            "payload": payload,
            "model": model,
            "created_at": now.isoformat(),
            "expires_at": expires,
        }, on_conflict="cache_key").execute()
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm_response_cache L2 set failed: %s", exc)
