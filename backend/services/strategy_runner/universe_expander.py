"""Universe expander — PR-FAN.

Map a strategy's symbolic universe ("nifty50", "nifty100", "single") to
a concrete list of symbols. Wraps the existing ``ai.qlib.load_universe``
for the universe tiers; handles ``single`` and sector universes locally.

Cached in-process — universe membership is stable within a trading day.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)


# Sector universe → hardcoded symbol lists. These are kept small + curated
# so the runner doesn't iterate the entire NSE looking for sector membership.
# When we get real sector classification feeds, replace with a DB query.
_SECTOR_SYMBOLS = {
    "sector:IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS",
        "PERSISTENT", "COFORGE", "OFSS",
    ],
    "sector:BANK": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "INDUSINDBK", "PNB", "BANKBARODA", "FEDERALBNK", "IDFCFIRSTB",
    ],
    "sector:AUTO": [
        "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT",
        "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY", "BHARATFORG", "MOTHERSON",
    ],
    "sector:PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
        "AUROPHARMA", "TORNTPHARM", "BIOCON", "ZYDUSLIFE", "ALKEM",
    ],
    "sector:FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "GODREJCP", "COLPAL", "MARICO", "TATACONSUM", "UBL",
    ],
    "sector:METAL": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL",
        "SAIL", "NMDC", "COALINDIA", "NATIONALUM", "WELCORP",
    ],
    "sector:ENERGY": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "HINDPETRO",
        "GAIL", "PETRONET", "OIL", "MGL", "IGL",
    ],
    "sector:INFRA": [
        "LT", "ULTRACEMCO", "ADANIPORTS", "ADANIENT", "GMRINFRA",
        "IRB", "NCC", "GRINFRA", "PNCINFRA", "KEC",
    ],
}


@lru_cache(maxsize=16)
def expand_universe(name: str, single_symbol: Optional[str] = None) -> List[str]:
    """Return the concrete symbol list for a strategy.universe value.

    Args:
        name: one of ``single`` | ``nifty50`` | ``nifty100`` | ``nifty500`` |
              ``sector:IT`` | ``sector:BANK`` | ...
        single_symbol: required when ``name == "single"`` — the symbol
                       to evaluate (NIFTY, BANKNIFTY, or a stock).
    """
    name = (name or "").strip()

    if name == "single":
        if not single_symbol:
            raise ValueError("expand_universe(name='single') requires single_symbol")
        return [single_symbol.upper()]

    if name in _SECTOR_SYMBOLS:
        return list(_SECTOR_SYMBOLS[name])

    if name in ("nifty50", "nifty100", "nifty500"):
        try:
            from ...ai.qlib import load_universe
        except Exception as exc:
            logger.warning("qlib universe load failed: %s — returning empty list", exc)
            return []
        try:
            return list(load_universe(name))
        except Exception as exc:
            logger.warning("load_universe(%s) failed: %s", name, exc)
            return []

    logger.warning("expand_universe: unknown universe '%s'", name)
    return []


def clear_universe_cache() -> None:
    """Invalidate the in-process universe cache. Called at start of day
    so morning index reshuffles aren't masked."""
    expand_universe.cache_clear()
