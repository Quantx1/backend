"""AI Setup Finder — unified count of the 4 canonical swing setups.

REUSES the existing live-screener scanners (NO new detection logic). Maps the
four canonical setup families to the best-fit scanner ids in the registry
(see ``data/screener/filters.py`` + the ``/scan/{scanner_id}`` dispatcher in
``api/screener_routes.py``), runs each, and returns labeled counts + the
matched symbol list per bucket so the UI can render "Breakout (23)" chips.

Deterministic, 0 tokens — there is no LLM anywhere in this path. Honest-empty
per the no-fallbacks contract: a category that the scanner returns nothing for
(or that errors) comes back with ``count: 0`` + an empty symbol list rather
than a fabricated number, and ``total`` is the sum of real matches.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 600  # the 4 scans are expensive (whole-universe) — share results 10 min

# The 4 canonical setup families → the existing scanner that best detects each.
# Order is the display order. Each entry: (key, label, scanner_id).
#   breakout    → #58 Breakout w/ Volume (52w high + vol 2×)
#   pullback    → #59 Pullback to EMA21 (uptrend dip at the key short-MA)
#   trend       → #54 MA Stack Bullish (price > EMA21 > SMA50 > SMA200)
#   reversal    → #57 Oversold Bounce Setup (RSI<35 but above SMA200)
# These ids are real, registered filters in SCANNER_FILTERS — we don't add or
# reimplement any detection; the Setup Finder is purely an aggregation surface.
SETUP_MAP: List[tuple] = [
    ("breakout", "Breakout", 58),
    ("pullback", "Pullback", 59),
    ("trend", "Trend continuation", 54),
    ("reversal", "Reversal", 57),
]


def _symbols_from_scan(result: Any) -> List[str]:
    """Pull the matched symbols out of a ``run_scanner`` payload.

    Scanner results are ``{"results": [{"symbol": ..., ...}, ...]}`` (see
    ``data/screener/formatting.format_for_frontend``). Be defensive: a failed
    scan returns ``success: False`` with an empty ``results`` list, and any
    unexpected shape collapses to an empty list (honest-empty, never raises).
    """
    if not isinstance(result, dict):
        return []
    rows = result.get("results") or []
    out: List[str] = []
    seen = set()
    for r in rows:
        sym = (r or {}).get("symbol") if isinstance(r, dict) else None
        if not sym:
            continue
        sym = str(sym).strip()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


async def find_setups(universe: Optional[str] = None) -> Dict[str, Any]:
    """Run the 4 canonical setup scanners and return labeled counts.

    Returns::

        {
          "setups": [
            {"key": "breakout", "label": "Breakout", "count": 23, "symbols": [...]},
            ...4 buckets, always in canonical order...
          ],
          "total": <sum of the four counts>,
          "ok": <True if at least one scanner returned without raising>,
        }

    ``universe`` maps to the live-screener exchange/index code (defaults to
    NSE / Nifty500 == the scanner's own defaults). A scanner that errors is
    isolated to its own bucket (count 0) so one bad category never zeroes the
    whole card. ``ok`` is False only when EVERY scanner raised — the signal the
    UI uses to render nothing instead of a misleading "0 setups".
    """
    from ...ai.agents.response_cache import cache_get, cache_set
    from ...data.screener.engine import get_live_screener

    screener = get_live_screener()
    # The legacy engine takes (scanner_id, exchange, index). "N"/"12" == the
    # NSE Nifty500 default that the rest of the screener UI uses. We expose a
    # single `universe` knob; for v1 it just selects the index breadth.
    index = "0" if (universe or "").lower() in {"all", "nse_all", "full"} else "12"

    # The 4 whole-universe scans cost minutes on a cold cache — serve repeats
    # from the shared cache so the card loads instantly for everyone after the
    # first run of a window. Only successful runs are cached (self-heal).
    ck = f"setupfinder:{index}"
    hit = cache_get(ck)
    if hit and hit.get("ok"):
        return hit

    async def _run_one(key: str, label: str, scanner_id: int) -> Dict[str, Any]:
        symbols: List[str] = []
        ok = False
        try:
            result = await screener.run_scanner(scanner_id, "N", index)
            ok = True
            symbols = _symbols_from_scan(result)
        except Exception as exc:  # isolate per-bucket failure
            logger.debug("setup_finder scanner %s (%s) failed: %s", scanner_id, key, exc)
        return {"key": key, "label": label, "count": len(symbols),
                "symbols": symbols, "_ok": ok}

    # Concurrent — wall-clock = slowest single scan, not the sum of four.
    buckets = await asyncio.gather(
        *[_run_one(key, label, sid) for key, label, sid in SETUP_MAP])

    oks = [b.pop("_ok") for b in buckets]   # pop from EVERY bucket (no short-circuit)
    any_ok = any(oks)
    total = sum(b["count"] for b in buckets)
    out = {"setups": list(buckets), "total": total, "ok": any_ok}
    if any_ok:
        cache_set(ck, out, ttl_seconds=_CACHE_TTL_S, surface="setup_finder", model="")
    return out
