"""Footprint / Cumulative Volume Delta (#21) — bar-level proxy.

True tick footprint needs a per-trade feed (the tick_collector scaffold isn't
live). This computes the well-known OHLCV proxy: bar delta = volume × the
close-location value ((close-low)-(high-close))/(high-low) ∈ [-1,1] — buying when
the bar closes near its high, selling near its low. Cumulative delta = the
running sum (the CVD line); buy% = where the close sits in the range. Honestly
a daily-bar approximation, not tick-level. All pure (tested).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _clv(high: float, low: float, close: float) -> Optional[float]:
    rng = high - low
    if rng <= 0:
        return None
    return ((close - low) - (high - close)) / rng   # [-1, 1]


def bar_delta(high: float, low: float, close: float, volume: float) -> float:
    """Volume-weighted close-location value (buy +, sell −)."""
    clv = _clv(high, low, close)
    if clv is None or not volume:
        return 0.0
    return round(clv * float(volume), 2)


def buy_pct(high: float, low: float, close: float) -> float:
    """Share of the bar that is buying pressure (close location), 0-100."""
    clv = _clv(high, low, close)
    return 50.0 if clv is None else round((clv + 1) / 2 * 100, 1)


def compute_cvd(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """bars: [{date, high, low, close, volume}] oldest->newest -> per-bar delta,
    cumulative delta (CVD), buy%."""
    out: List[Dict[str, Any]] = []
    cum = 0.0
    for b in bars:
        d = bar_delta(b["high"], b["low"], b["close"], b.get("volume") or 0)
        cum += d
        out.append({"date": b.get("date"), "delta": d, "cvd": round(cum, 2),
                    "buy_pct": buy_pct(b["high"], b["low"], b["close"])})
    return out


def footprint(symbol: str, days: int = 60) -> Dict[str, Any]:
    """CVD line + latest delta/buy% + trend (rising/falling) for a symbol."""
    sym = symbol.strip().upper()
    out: Dict[str, Any] = {"symbol": sym, "cvd": [], "latest": None, "trend": None}
    try:
        from ...data.market import get_market_data_provider
        df = get_market_data_provider().get_historical(sym, period="3mo", interval="1d")
        if df is None or len(df) == 0:
            return out
        df.columns = [c.lower() for c in df.columns]
        bars: List[Dict[str, Any]] = []
        for idx, r in df.iterrows():
            try:
                bars.append({
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    "high": float(r["high"]), "low": float(r["low"]),
                    "close": float(r["close"]), "volume": float(r.get("volume", 0) or 0),
                })
            except Exception:
                continue
        series = compute_cvd(bars)
        if series:
            out["latest"] = series[-1]
            out["cvd"] = [{"date": s["date"], "cvd": s["cvd"]} for s in series[-days:]]
            if len(series) >= 10:
                out["trend"] = "rising" if series[-1]["cvd"] > series[-10]["cvd"] else "falling"
    except Exception as e:
        logger.debug("footprint failed for %s: %s", sym, e)
    return out
