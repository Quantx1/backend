"""
system_flags — platform-wide ops toggles reader.

Every order-placing path should consult ``is_globally_halted()`` before
firing. Reads the ``system_flags`` table with a short in-memory TTL so
checks cost microseconds in the hot path instead of a round-trip per
trade.

Flipping the flag invalidates the cache for all processes eventually
(TTL expiry), so a 15-second lag between admin click and a worker honoring
the change is acceptable. Admin UI uses the source-of-truth endpoint,
not this cache.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 15.0


def _read_flag_row(key: str, *, supabase_client=None) -> Dict[str, Any]:
    if supabase_client is None:
        try:
            from ..core.database import get_supabase_admin
            supabase_client = get_supabase_admin()
        except Exception:
            return {}
    try:
        rows = (
            supabase_client.table("system_flags")
            .select("value")
            .eq("key", key)
            .limit(1)
            .execute()
        )
        row = (rows.data or [None])[0]
        if not row:
            return {}
        value = row.get("value") or {}
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        logger.debug("system_flags read failed for %s: %s", key, exc)
        return {}


def get_flag(key: str, *, supabase_client=None, force_refresh: bool = False) -> Dict[str, Any]:
    """Return the JSON value of a system_flag, empty dict on miss.
    Cached for ``_CACHE_TTL_SECONDS``. Caller should pass
    ``force_refresh=True`` only from admin-facing paths."""
    now = time.time()
    if not force_refresh:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
            if cached and cached[1] > now:
                return cached[0]

    value = _read_flag_row(key, supabase_client=supabase_client)
    with _CACHE_LOCK:
        _CACHE[key] = (value, now + _CACHE_TTL_SECONDS)
    return value


def is_globally_halted(*, supabase_client=None) -> bool:
    """Fail-closed read: any error reads as NOT halted so we don't
    accidentally freeze the platform over a transient DB hiccup.
    The admin UI is the authoritative check, not this."""
    value = get_flag("global_kill_switch", supabase_client=supabase_client)
    return bool(value.get("active", False))


def global_halt_reason(*, supabase_client=None) -> Optional[str]:
    value = get_flag("global_kill_switch", supabase_client=supabase_client)
    reason = value.get("reason")
    return str(reason) if reason else None


def invalidate_cache(key: Optional[str] = None) -> None:
    """Drop the cache for a single flag, or clear everything."""
    with _CACHE_LOCK:
        if key is None:
            _CACHE.clear()
        else:
            _CACHE.pop(key, None)


__all__ = [
    "get_flag",
    "global_halt_reason",
    "invalidate_cache",
    "is_globally_halted",
]
