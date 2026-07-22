"""Movers with WHY — cause-attribution for the day's EOD movers.

Kills the reflexive "why is this stock up 8%?" Google search: for each of the
day's EOD close-to-close movers, fan out the existing multi-source headline
fetch and attach the freshest headline as the driver — or say, honestly,
"no identifiable news" (which is itself a signal: pure momentum/sector beta).

Zero LLM tokens; a handful of RSS/API fetches per symbol, cached 1 hour
in-process. Movers come from settled EOD candles (SEBI-safe); headlines are
public information.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}
_TTL_S = 3600

_NO_NEWS = "No identifiable news — likely momentum / sector move"


async def _driver_for(symbol: str) -> Dict[str, Any]:
    """Freshest recent headline for a symbol, or an honest empty."""
    try:
        from ...ai.sentiment.news_providers import fetch_all_sources
        items = await fetch_all_sources(symbol, lookback_days=2, max_per_source=6)
    except Exception as e:  # noqa: BLE001
        logger.debug("movers-why fetch failed for %s: %s", symbol, e)
        items = []
    best = None
    for it in items:
        title = (it.get("title") or "").strip()
        if len(title) < 15:
            continue
        if best is None or (it.get("published") or "") > (best.get("published") or ""):
            best = it
    if not best:
        return {"driver": _NO_NEWS, "source": None, "link": None, "has_news": False}
    return {
        "driver": (best.get("title") or "").strip()[:140],
        "source": best.get("source"),
        "link": best.get("link"),
        "has_news": True,
    }


async def movers_why() -> Dict[str, Any]:
    """EOD movers, each annotated with its probable cause. Cached 1h."""
    hit = _CACHE.get("w")
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]

    from ..briefing.market_briefing import _eod_movers
    movers = await asyncio.to_thread(_eod_movers, 4)
    if not movers:
        return {"items": [], "note": "EOD · settled close"}

    drivers = await asyncio.gather(*(_driver_for(m["symbol"]) for m in movers))
    items = [{**m, **d} for m, d in zip(movers, drivers)]
    out = {"items": items, "note": "EOD · settled close · headlines are public information"}
    _CACHE["w"] = (time.monotonic(), out)
    return out
