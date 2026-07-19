"""
Volume Profile — backend POC / value area / HVN / LVN from real OHLCV.

Closes the audit gap "volume profile is frontend-only, from daily OHLCV,
with no HVN/LVN labels". This is the deterministic backend computation:
distribute each bar's volume across the price bins it spanned (a TPO-style
approximation from OHLCV), then derive Point of Control, the 70% value area
(VAH/VAL), and High/Low-Volume Nodes.

Pure + unit-tested core (``compute_volume_profile``); the service wrapper
fetches candles via the market provider. No LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

VALUE_AREA_FRACTION = 0.70


def compute_volume_profile(
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
    *,
    bins: int = 24,
) -> Dict[str, Any]:
    """Volume-at-price profile from OHLCV bars. Pure; no I/O.

    Each bar spreads its volume evenly across the price bins its [low, high]
    range overlaps. Returns POC (max-volume price), value area (VAH/VAL band
    holding 70% of volume around POC), and HVN/LVN node prices. Honest-empty
    ({} fields None) when there aren't enough bars or no price range.
    """
    n = min(len(highs), len(lows), len(volumes))
    if n < 5 or bins < 4:
        return {"bins": [], "poc": None, "vah": None, "val": None, "hvn": [], "lvn": []}

    hi = max(float(highs[i]) for i in range(n))
    lo = min(float(lows[i]) for i in range(n))
    if hi <= lo:
        return {"bins": [], "poc": None, "vah": None, "val": None, "hvn": [], "lvn": []}

    width = (hi - lo) / bins
    edges = [lo + width * k for k in range(bins + 1)]
    centers = [round((edges[k] + edges[k + 1]) / 2, 2) for k in range(bins)]
    vol_at = [0.0] * bins

    for i in range(n):
        bl, bh, bv = float(lows[i]), float(highs[i]), float(volumes[i])
        if bv <= 0 or bh < bl:
            continue
        lo_idx = max(0, min(bins - 1, int((bl - lo) / width)))
        hi_idx = max(0, min(bins - 1, int((bh - lo) / width)))
        span = hi_idx - lo_idx + 1
        share = bv / span
        for k in range(lo_idx, hi_idx + 1):
            vol_at[k] += share

    total = sum(vol_at)
    if total <= 0:
        return {"bins": [], "poc": None, "vah": None, "val": None, "hvn": [], "lvn": []}

    poc_idx = max(range(bins), key=lambda k: vol_at[k])

    # Value area: expand out from POC (greedily taking the heavier neighbour)
    # until 70% of total volume is enclosed.
    lo_i = hi_i = poc_idx
    acc = vol_at[poc_idx]
    target = total * VALUE_AREA_FRACTION
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        left = vol_at[lo_i - 1] if lo_i > 0 else -1.0
        right = vol_at[hi_i + 1] if hi_i < bins - 1 else -1.0
        if right >= left:
            hi_i += 1
            acc += vol_at[hi_i]
        else:
            lo_i -= 1
            acc += vol_at[lo_i]

    mean_v = total / bins
    hvn = [centers[k] for k in range(bins) if vol_at[k] >= mean_v * 1.5]
    lvn = [centers[k] for k in range(bins) if 0 < vol_at[k] <= mean_v * 0.4]

    profile = [
        {"price": centers[k], "volume": round(vol_at[k], 2),
         "pct": round(vol_at[k] / total * 100, 2)}
        for k in range(bins)
    ]
    return {
        "bins": profile,
        "poc": centers[poc_idx],
        "vah": round(edges[hi_i + 1], 2),
        "val": round(edges[lo_i], 2),
        "hvn": hvn,
        "lvn": lvn,
        "value_area_pct": round(acc / total * 100, 1),
    }


def volume_profile(symbol: str, *, lookback_days: int = 60, bins: int = 24) -> Dict[str, Any]:
    """Volume profile for a symbol from daily candles (market provider)."""
    sym = (symbol or "").strip().upper()
    highs: List[float] = []
    lows: List[float] = []
    vols: List[float] = []
    try:
        from ...data.market import get_market_data_provider
        period = "1y" if lookback_days > 180 else "6mo" if lookback_days > 90 else "3mo"
        df = get_market_data_provider().get_historical(sym, period=period, interval="1d")
        if df is not None and len(df):
            df.columns = [c.lower() for c in df.columns]
            tail = df.tail(lookback_days)
            if {"high", "low", "volume"} <= set(tail.columns):
                highs = [float(x) for x in tail["high"].tolist()]
                lows = [float(x) for x in tail["low"].tolist()]
                vols = [float(x) for x in tail["volume"].tolist()]
    except Exception as exc:  # noqa: BLE001
        logger.debug("volume_profile fetch failed for %s: %s", sym, exc)

    prof = compute_volume_profile(highs, lows, vols, bins=bins)
    return {"symbol": sym, "lookback_days": lookback_days, "bars": len(vols), **prof}


__all__ = ["compute_volume_profile", "volume_profile", "VALUE_AREA_FRACTION"]
