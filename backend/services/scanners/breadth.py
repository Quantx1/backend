"""Market breadth (#breadth) — true Advance/Decline + A/D line.

The dashboard's "breadth" was a sector-change average. This computes the real
advancing-vs-declining issue count per day across the universe (close vs prior
close) and the cumulative A/D line. `cumulative_ad` is pure (tested); the daily
counts come from one candles window query (direct Postgres).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}
_TTL_S = 600


def cumulative_ad(daily: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """daily: [{date, adv, dec}] oldest->newest. Adds net + cumulative ad_line."""
    out: List[Dict[str, Any]] = []
    cum = 0
    for r in daily:
        adv, dec = int(r.get("adv") or 0), int(r.get("dec") or 0)
        cum += adv - dec
        out.append({"date": r.get("date"), "adv": adv, "dec": dec,
                    "net": adv - dec, "ad_line": cum})
    return out


def _daily_counts(days: int) -> List[Dict[str, Any]]:
    from ...data.ohlc_store import pg_connect
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                WITH d AS (
                  SELECT timestamp::date AS dt, close,
                         lag(close) OVER (PARTITION BY stock_symbol
                                          ORDER BY timestamp) AS prev
                  FROM candles
                  WHERE interval='1d' AND timestamp >= now() - (%s || ' days')::interval
                )
                SELECT dt,
                  count(*) FILTER (WHERE close > prev) AS adv,
                  count(*) FILTER (WHERE close < prev) AS dec
                FROM d WHERE prev IS NOT NULL
                GROUP BY dt ORDER BY dt
            """, (days + 5,))
            return [{"date": dt.isoformat(), "adv": adv, "dec": dec}
                    for dt, adv, dec in cur.fetchall()]
    finally:
        conn.close()


def breadth(days: int = 120) -> Dict[str, Any]:
    """Today's A/D + ratio + the cumulative A/D line (last `days`). Cached 10m."""
    hit = _CACHE.get("b")
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]
    out: Dict[str, Any] = {"today": None, "ratio": None, "ad_line": []}
    try:
        line = cumulative_ad(_daily_counts(days))
        if line:
            today = line[-1]
            out["today"] = today
            out["ratio"] = round(today["adv"] / today["dec"], 2) if today["dec"] else None
            out["ad_line"] = [{"date": r["date"], "ad_line": r["ad_line"]} for r in line[-days:]]
            _CACHE["b"] = (time.monotonic(), out)
    except Exception as e:
        logger.debug("breadth failed: %s", e)
    return out
