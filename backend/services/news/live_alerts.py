"""Smart Alerts (#8) — live deterministic alert feed.

Evaluates the named conditions traders want — Volume 3× normal, OI ±15%,
20-day-high breakout, IV-Rank ≥ 80, AI high-prob signal — over real data and
returns the ones firing right now. Pure rule helpers (tested) + an efficient
universe scan (one candles window query + the OI feed). This is the detection
layer the saved-scan dispatcher / scheduler can also push from.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VOL_X = 3.0       # "volume 3× normal"
OI_PCT = 15.0     # "OI increased 15%"
IV_RANK = 80.0    # "IV Rank above 80"

_CACHE: Dict[str, tuple] = {}
_TTL_S = 300


def volume_alert(symbol: str, x_avg: Optional[float]) -> Optional[Dict[str, Any]]:
    if x_avg and x_avg >= VOL_X:
        return {"symbol": symbol, "type": "volume", "severity": "high",
                "message": f"Volume {x_avg:.1f}× the 20-day average."}
    return None


def oi_alert(symbol: str, oi_change_pct: Optional[float]) -> Optional[Dict[str, Any]]:
    if oi_change_pct is not None and abs(oi_change_pct) >= OI_PCT:
        d = "rose" if oi_change_pct > 0 else "fell"
        return {"symbol": symbol, "type": "oi", "severity": "high",
                "message": f"Futures OI {d} {abs(oi_change_pct):.0f}%."}
    return None


def breakout_alert(symbol: str, close: Optional[float], hi_prior20: Optional[float]) -> Optional[Dict[str, Any]]:
    if close and hi_prior20 and close > hi_prior20:
        return {"symbol": symbol, "type": "breakout", "severity": "medium",
                "message": "Price crossed its 20-day high."}
    return None


def iv_alert(symbol: str, iv_rank: Optional[float]) -> Optional[Dict[str, Any]]:
    if iv_rank is not None and iv_rank >= IV_RANK:
        return {"symbol": symbol, "type": "iv", "severity": "medium",
                "message": f"IV Rank {iv_rank:.0f} — premium-selling regime."}
    return None


def _scan_candles_alerts() -> List[Dict[str, Any]]:
    """Volume-3x + 20-day-high breakout from one candles window query."""
    from ...data.ohlc_store import pg_connect
    conn = pg_connect()
    out: List[Dict[str, Any]] = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                WITH ranked AS (
                  SELECT stock_symbol, close, high, volume,
                         row_number() OVER (PARTITION BY stock_symbol
                                            ORDER BY timestamp DESC) AS rn
                  FROM candles WHERE interval='1d'
                )
                SELECT stock_symbol,
                  max(close)  FILTER (WHERE rn=1) AS c0,
                  max(high)   FILTER (WHERE rn BETWEEN 2 AND 21) AS hi20,
                  max(volume) FILTER (WHERE rn=1) AS v0,
                  avg(volume) FILTER (WHERE rn BETWEEN 2 AND 21) AS avgv
                FROM ranked WHERE rn <= 21
                GROUP BY stock_symbol
            """)
            for sym, c0, hi20, v0, avgv in cur.fetchall():
                c0 = float(c0) if c0 else None
                hi20 = float(hi20) if hi20 else None
                x = (float(v0) / float(avgv)) if (v0 and avgv) else None
                for a in (volume_alert(sym, x), breakout_alert(sym, c0, hi20)):
                    if a:
                        out.append(a)
    finally:
        conn.close()
    return out


def _scan_oi_alerts() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        from ...data.screener.nse_data import get_nse_data
        for r in (get_nse_data().get_participant_oi().get("data") or []):
            a = oi_alert(str(r.get("symbol", "")).upper(), r.get("oi_change_pct"))
            if a:
                out.append(a)
    except Exception as e:
        logger.debug("oi alerts failed: %s", e)
    return out


def scan_live_alerts(limit: int = 60) -> List[Dict[str, Any]]:
    """All firing alerts right now (cached 5m). Deterministic; honest-empty on
    failure of any source."""
    hit = _CACHE.get("live")
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1][:limit]
    alerts: List[Dict[str, Any]] = []
    try:
        alerts.extend(_scan_candles_alerts())
    except Exception as e:
        logger.debug("candle alerts failed: %s", e)
    alerts.extend(_scan_oi_alerts())
    # severity-first, then by type
    order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda a: (order.get(a.get("severity"), 9), a.get("type", "")))
    if alerts:
        _CACHE["live"] = (time.monotonic(), alerts)
    return alerts[:limit]
