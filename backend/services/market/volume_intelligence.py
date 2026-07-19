"""Volume Intelligence (#9) — one unified read of a stock's volume + delivery.

Combines: today's volume vs 20-day average (the spike), its percentile over ~60
sessions (how unusual), and the delivery-% trend vs its own average (the piece
that was missing — only absolute delivery existed). Classifies the day as
accumulation / churn / high-activity / quiet, with deterministic drivers and an
optional grounded narrative. `compute_volume_intel` is pure (tested).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def compute_volume_intel(volumes: Sequence[float],
                         deliveries: Optional[Sequence[Optional[float]]] = None) -> Dict[str, Any]:
    """Pure: spike (x avg), percentile, delivery trend, and a signal label."""
    out: Dict[str, Any] = {
        "today_volume": None, "avg_volume_20d": None, "x_avg": None,
        "vol_percentile": None, "delivery_today": None, "avg_delivery": None,
        "delivery_trend": None, "signal": "normal",
    }
    vols = [float(v) for v in (volumes or []) if v and v > 0]
    if len(vols) < 5:
        return out
    today = vols[-1]
    prior = vols[-21:-1] if len(vols) >= 21 else vols[:-1]
    avg = sum(prior) / len(prior) if prior else None
    out["today_volume"] = round(today)
    if avg:
        out["avg_volume_20d"] = round(avg)
        out["x_avg"] = round(today / avg, 2)
    window = vols[-60:]
    out["vol_percentile"] = round(sum(1 for v in window if v < today) / len(window) * 100)

    if deliveries:
        ds = [float(d) for d in deliveries if d is not None]
        if ds:
            out["delivery_today"] = round(ds[-1], 1)
            prior_d = ds[-21:-1] if len(ds) >= 21 else ds[:-1]
            if prior_d:
                avgd = sum(prior_d) / len(prior_d)
                out["avg_delivery"] = round(avgd, 1)
                out["delivery_trend"] = round(ds[-1] - avgd, 1)

    x, dt, ad = out["x_avg"], out["delivery_today"], out["avg_delivery"]
    if x and x >= 2:
        if dt is not None and ((ad and dt >= ad * 1.1) or dt >= 60):
            out["signal"] = "accumulation"   # heavy volume on strong delivery
        elif dt is not None and dt < 35:
            out["signal"] = "churn"          # heavy volume, weak delivery (intraday churn)
        else:
            out["signal"] = "high_activity"
    elif x and x < 0.5:
        out["signal"] = "quiet"
    return out


def _drivers(v: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if v.get("x_avg"):
        out.append(f"Volume {v['x_avg']}× the 20-day average "
                   f"({v.get('vol_percentile')}th percentile of the last ~60 sessions).")
    if v.get("delivery_today") is not None:
        line = f"Delivery {v['delivery_today']}%"
        if v.get("delivery_trend") is not None:
            arrow = "above" if v["delivery_trend"] >= 0 else "below"
            line += f" ({abs(v['delivery_trend'])}pp {arrow} its average)"
        out.append(line + ".")
    sig = v.get("signal")
    label = {"accumulation": "Heavy volume on strong delivery — looks like accumulation.",
             "churn": "Heavy volume but weak delivery — intraday churn, not conviction.",
             "high_activity": "Above-average participation today.",
             "quiet": "Unusually quiet — well below average volume."}.get(sig)
    if label:
        out.append(label)
    return out


def volume_intel(symbol: str, *, use_llm: bool = False, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Volume Intelligence for a symbol: volume from the market provider (any
    stock), delivery from candles (backfilled bonus). Drivers deterministic;
    narrative grounded + cached per symbol/day when use_llm."""
    sym = symbol.strip().upper()
    volumes: List[float] = []
    deliveries: List[Optional[float]] = []
    try:
        from ...data.market import get_market_data_provider
        df = get_market_data_provider().get_historical(sym, period="3mo", interval="1d")
        if df is not None and len(df):
            df.columns = [c.lower() for c in df.columns]
            if "volume" in df.columns:
                volumes = [float(x) for x in df["volume"].tolist() if x == x]
    except Exception as e:
        logger.debug("volume_intel volumes failed for %s: %s", sym, e)
    try:
        from ...core.database import get_supabase_admin
        from ...data.ohlc_store import read_candles
        rows = read_candles(get_supabase_admin(), sym, "1d", limit=80)
        deliveries = [r.get("delivery_pct") for r in rows]
    except Exception:
        pass

    intel = compute_volume_intel(volumes, deliveries)
    drivers = _drivers(intel)
    narrative = None
    if use_llm and drivers:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            {"symbol": sym, **intel}, f"What does today's volume and delivery say about {sym}?",
            cache_key=f"volintel:{sym}:{date.today().isoformat()}", user_id=user_id)
    return {"symbol": sym, **intel, "drivers": drivers, "narrative": narrative}
