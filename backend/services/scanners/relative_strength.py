"""True Relative Strength vs NIFTY (multi-window) — #7.

Replaces the universe-median proxy with a real benchmark-relative return:
RS = stock %-change over a window MINUS NIFTY %-change over the same window.
Positive = outperforming the index. `compute_rel_return` is pure (tested);
the wrappers read closes through the market provider (candles read-through for
the stock, Kite for the index).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_WINDOWS = (20, 50, 120)            # ~1m / ~2.5m / ~6m trading days
_CACHE: Dict[str, tuple] = {}       # symbol -> (ts, result)
_TTL_S = 1800


def compute_rel_return(stock_closes: Sequence[float], bench_closes: Sequence[float],
                       window: int) -> Optional[float]:
    """Relative return over `window` bars: stock %chg − benchmark %chg.
    None when either series is too short or a base price is zero."""
    if (not stock_closes or not bench_closes
            or len(stock_closes) <= window or len(bench_closes) <= window):
        return None
    s0, s1 = stock_closes[-1 - window], stock_closes[-1]
    b0, b1 = bench_closes[-1 - window], bench_closes[-1]
    if not s0 or not b0:
        return None
    return round((s1 / s0 - 1) * 100 - (b1 / b0 - 1) * 100, 2)


def _closes(symbol: str, period: str = "6mo") -> List[float]:
    try:
        from ...data.market import get_market_data_provider
        df = get_market_data_provider().get_historical(symbol, period=period, interval="1d")
        if df is None or len(df) == 0:
            return []
        df.columns = [c.lower() for c in df.columns]
        return [float(c) for c in df["close"].tolist() if c == c]
    except Exception as e:
        logger.debug("rs closes failed for %s: %s", symbol, e)
        return []


def symbol_rs(symbol: str, *, benchmark: str = "NIFTY") -> Dict:
    """Multi-window RS vs the benchmark for one symbol (cached 30m)."""
    sym = symbol.strip().upper()
    hit = _CACHE.get(sym)
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]
    stock = _closes(sym)
    bench = _closes(benchmark)
    rs = {f"rs_{w}d": compute_rel_return(stock, bench, w) for w in _WINDOWS}
    rs["benchmark"] = benchmark
    rs["outperforming"] = bool((rs.get("rs_50d") or 0) > 0)
    out = {"symbol": sym, **rs}
    if stock and bench:
        _CACHE[sym] = (time.monotonic(), out)
    return out
