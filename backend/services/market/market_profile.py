"""Market Profile / TPO (#21) — time-at-price distribution.

Distinct from the volume-at-price panel: this counts TIME (periods) at each
price. Each bar is one bracket; we bin the price range and count how many bars
trade in each bin (TPOs), then derive the Point of Control (most-traded price)
and the 70% Value Area (VAH/VAL). Honestly a daily-bracket TPO (one period per
day), not 30-minute intraday brackets. `compute_tpo` is pure (tested).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def compute_tpo(bars: List[Dict[str, Any]], bins: int = 24) -> Dict[str, Any]:
    """bars: [{high, low}]. -> {profile:[{price,tpo}], poc, vah, val, total_tpo}."""
    lows = [b["low"] for b in bars if b.get("low") is not None]
    highs = [b["high"] for b in bars if b.get("high") is not None]
    empty = {"profile": [], "poc": None, "vah": None, "val": None, "total_tpo": 0}
    if not lows or not highs:
        return empty
    lo, hi = min(lows), max(highs)
    if hi <= lo:
        return empty
    step = (hi - lo) / bins
    counts = [0] * bins
    for b in bars:
        if b.get("low") is None or b.get("high") is None:
            continue
        i0 = max(0, int((b["low"] - lo) / step))
        i1 = min(bins - 1, int((b["high"] - lo) / step))
        for i in range(i0, i1 + 1):
            counts[i] += 1
    profile = [{"price": round(lo + (i + 0.5) * step, 2), "tpo": counts[i]} for i in range(bins)]
    total = sum(counts)
    poc_i = max(range(bins), key=lambda i: counts[i])

    # Value area: expand from POC outward until 70% of TPOs are enclosed.
    target = total * 0.70
    inc = counts[poc_i]
    lo_i = hi_i = poc_i
    while inc < target and (lo_i > 0 or hi_i < bins - 1):
        below = counts[lo_i - 1] if lo_i > 0 else -1
        above = counts[hi_i + 1] if hi_i < bins - 1 else -1
        if above >= below:
            hi_i += 1
            inc += counts[hi_i]
        else:
            lo_i -= 1
            inc += counts[lo_i]
    return {"profile": profile, "poc": profile[poc_i]["price"],
            "vah": profile[hi_i]["price"], "val": profile[lo_i]["price"], "total_tpo": total}


def market_profile(symbol: str, days: int = 60, bins: int = 24) -> Dict[str, Any]:
    sym = symbol.strip().upper()
    out: Dict[str, Any] = {"symbol": sym, "profile": [], "poc": None, "vah": None, "val": None}
    try:
        from ...data.market import get_market_data_provider
        df = get_market_data_provider().get_historical(sym, period="3mo", interval="1d")
        if df is None or len(df) == 0:
            return out
        df.columns = [c.lower() for c in df.columns]
        df = df.tail(days)
        bars = [{"high": float(r["high"]), "low": float(r["low"])}
                for _, r in df.iterrows() if r["high"] == r["high"] and r["low"] == r["low"]]
        tpo = compute_tpo(bars, bins=bins)
        out.update(tpo)
        out["symbol"] = sym
    except Exception as e:
        logger.debug("market_profile failed for %s: %s", sym, e)
    return out
