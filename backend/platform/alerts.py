"""
alert_prefs — event × channel resolver used by the realtime bus and
every feature emitter.

Callsite pattern (realtime.py, push_service, feature code):

    from backend.services.alert_prefs import channels_for_event
    channels = await channels_for_event(user_id, "target_hit")
    if "push" in channels: ...
    if "telegram" in channels: ...

Defaults fall through from DEFAULT_PREFS in api/alerts_routes.py so a
missing row never drops a critical notification silently.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Set

from ..api.alerts_routes import CHANNELS, DEFAULT_PREFS

logger = logging.getLogger(__name__)


def _merge(stored: Optional[Dict]) -> Dict[str, Dict[str, bool]]:
    stored = stored or {}
    out: Dict[str, Dict[str, bool]] = {}
    for event_key, default_row in DEFAULT_PREFS.items():
        row = dict(default_row)
        if event_key in stored and isinstance(stored[event_key], dict):
            for ch in CHANNELS:
                if ch in stored[event_key]:
                    row[ch] = bool(stored[event_key][ch])
        out[event_key] = row
    return out


def _read_prefs_sync(supabase_client, user_id: str) -> Dict[str, Dict[str, bool]]:
    try:
        rows = (
            supabase_client.table("user_profiles")
            .select("alert_preferences")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        if rows.data:
            return _merge(rows.data[0].get("alert_preferences"))
    except Exception as exc:
        logger.debug("alert_prefs lookup failed %s: %s", user_id, exc)
    return _merge(None)


async def channels_for_event(
    user_id: str,
    event: str,
    *,
    supabase_client=None,
) -> Set[str]:
    """Return the set of channels enabled for (user, event).
    Async-safe — wraps the sync Supabase client via ``to_thread``."""
    if event not in DEFAULT_PREFS:
        # Unknown events fall through the matrix entirely so legacy
        # emitters don't silently drop when they use a new key.
        return set(CHANNELS)

    if supabase_client is None:
        from ..core.database import get_supabase_admin
        supabase_client = get_supabase_admin()

    prefs = await asyncio.to_thread(_read_prefs_sync, supabase_client, user_id)
    row = prefs.get(event, DEFAULT_PREFS.get(event, {}))
    return {ch for ch, on in row.items() if on}


def channels_for_event_sync(
    user_id: str,
    event: str,
    *,
    supabase_client=None,
) -> Set[str]:
    """Sync variant for callers inside threads / non-async contexts."""
    if event not in DEFAULT_PREFS:
        return set(CHANNELS)
    if supabase_client is None:
        from ..core.database import get_supabase_admin
        supabase_client = get_supabase_admin()
    prefs = _read_prefs_sync(supabase_client, user_id)
    row = prefs.get(event, DEFAULT_PREFS.get(event, {}))
    return {ch for ch, on in row.items() if on}


__all__ = ["channels_for_event", "channels_for_event_sync"]
