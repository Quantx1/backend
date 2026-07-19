"""Probability Engine (#17) — empirical setup success rates.

Replaces the fabricated inline 'probability' arithmetic with REAL historical
outcomes: find every past occurrence of a setup in the symbol's own candle
history and measure how often price followed through within a horizon. So
"similar setups advanced >= 2% within 10 days 63% of the time" is actually true
(with the sample size shown). `_followthrough` is pure (tested).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _followthrough(idxs: List[int], closes: List[float], horizon: int, target: float) -> Dict[str, Any]:
    """Of the bars at `idxs`, the fraction that advanced >= target% within the
    next `horizon` bars."""
    n = len(closes)
    occ = succ = 0
    for i in idxs:
        if i + horizon >= n:
            continue
        entry = closes[i]
        fut = closes[i + 1:i + 1 + horizon]
        if not entry or not fut:
            continue
        occ += 1
        if (max(fut) - entry) / entry * 100 >= target:
            succ += 1
    return {"occurrences": occ, "success": succ,
            "prob_pct": round(succ / occ * 100, 1) if occ else None}


def setup_probabilities(symbol: str, *, horizon: int = 10, target: float = 2.0) -> Dict[str, Any]:
    """Empirical follow-through probabilities for the symbol's setups, plus
    whether each setup is active on the latest bar."""
    sym = symbol.strip().upper()
    out: Dict[str, Any] = {"symbol": sym, "horizon_days": horizon, "target_pct": target, "setups": []}
    try:
        from ...data.market import get_market_data_provider
        from ml.features.indicators import compute_all_indicators
        df = get_market_data_provider().get_historical(sym, period="1y", interval="1d")
        if df is None or len(df) < 60:
            return out
        df.columns = [c.lower() for c in df.columns]
        ind = compute_all_indicators(df)
        closes = [float(c) for c in ind["close"].tolist()]
        highs = [float(h) for h in ind["high"].tolist()]
        rsi = [float(r) if r == r else None for r in ind["rsi_14"].tolist()] if "rsi_14" in ind.columns else [None] * len(closes)
        n = len(closes)

        # Breakout: close above the prior 20-day high.
        bo_idx = [i for i in range(20, n) if closes[i] > max(highs[i - 20:i])]
        bo = _followthrough(bo_idx, closes, horizon, target)
        bo_active = bool(bo_idx and bo_idx[-1] == n - 1)
        out["setups"].append({"name": "20-day breakout", "active_now": bo_active, **bo})

        # Oversold bounce: RSI < 30.
        os_idx = [i for i in range(n) if rsi[i] is not None and rsi[i] < 30]
        osb = _followthrough(os_idx, closes, horizon, target)
        os_active = bool(os_idx and os_idx[-1] == n - 1)
        out["setups"].append({"name": "oversold (RSI<30) bounce", "active_now": os_active, **osb})

        # Trend continuation: price above the rising 50-DMA.
        if "sma_50" in ind.columns:
            sma = [float(s) if s == s else None for s in ind["sma_50"].tolist()]
            tc_idx = [i for i in range(1, n)
                      if sma[i] is not None and sma[i - 1] is not None
                      and closes[i] > sma[i] and sma[i] > sma[i - 1]]
            tc = _followthrough(tc_idx, closes, horizon, target)
            tc_active = bool(tc_idx and tc_idx[-1] == n - 1)
            out["setups"].append({"name": "uptrend continuation", "active_now": tc_active, **tc})
    except Exception as e:
        logger.debug("probability engine failed for %s: %s", sym, e)
    return out
