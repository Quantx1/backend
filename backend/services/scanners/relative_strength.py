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


def _ratio_series(stock: Sequence[float], bench: Sequence[float], points: int = 120) -> List[float]:
    """RS line: stock/bench price ratio over the last `points` sessions,
    rebased to 100 at the window start (the classic Mansfield-style read —
    rising = outperforming NIFTY, regardless of absolute direction)."""
    n = min(len(stock), len(bench), points)
    if n < 10:
        return []
    s, b = stock[-n:], bench[-n:]
    if not s[0] or not b[0]:
        return []
    base = s[0] / b[0]
    return [round((si / bi) / base * 100, 2) for si, bi in zip(s, b) if bi]


def _outperf_streak(stock: Sequence[float], bench: Sequence[float]) -> Optional[int]:
    """Consecutive sessions the 20d RS has been positive (negative → negated).
    None when history is too short."""
    n = min(len(stock), len(bench))
    if n < 40:
        return None
    streak = 0
    sign = None
    for i in range(n - 1, 20, -1):
        rel = compute_rel_return(stock[: i + 1], bench[: i + 1], 20)
        if rel is None:
            break
        cur = rel > 0
        if sign is None:
            sign = cur
        if cur != sign:
            break
        streak += 1
        if streak >= 60:
            break
    return streak if sign else -streak if streak else 0


def symbol_rs(symbol: str, *, benchmark: str = "NIFTY") -> Dict:
    """Multi-window RS vs the benchmark for one symbol (cached 30m), plus the
    RS ratio line (rebased 100) and the 20d-RS streak for the full card."""
    sym = symbol.strip().upper()
    hit = _CACHE.get(sym)
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]
    stock = _closes(sym)
    bench = _closes(benchmark)
    rs = {f"rs_{w}d": compute_rel_return(stock, bench, w) for w in _WINDOWS}
    rs["benchmark"] = benchmark
    rs["outperforming"] = bool((rs.get("rs_50d") or 0) > 0)
    out = {
        "symbol": sym,
        **rs,
        "ratio_line": _ratio_series(stock, bench),
        "streak_20d": _outperf_streak(stock, bench),
    }
    if stock and bench:
        _CACHE[sym] = (time.monotonic(), out)
    return out
