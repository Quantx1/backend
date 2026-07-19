"""Sector rotation (multi-period RRG) — #8.

Leaders/laggards by RELATIVE strength + momentum, not just today's heatmap.
Per sector: short (~5d) and long (~20d) average return vs the market average ->
RS-long / RS-short -> RRG quadrant (Leading / Weakening / Lagging / Improving).
`classify_quadrant` + `aggregate` are pure (tested); the reads come from the
candles table (one window query via direct Postgres) + the instruments sector map.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}
_TTL_S = 1800


def classify_quadrant(rs_long: Optional[float], rs_short: Optional[float]) -> str:
    """RRG quadrant from long/short relative strength (vs market avg)."""
    if rs_long is None or rs_short is None:
        return "n/a"
    if rs_long >= 0 and rs_short >= 0:
        return "leading"      # strong and still strengthening
    if rs_long >= 0 and rs_short < 0:
        return "weakening"    # was strong, momentum fading
    if rs_long < 0 and rs_short < 0:
        return "lagging"      # weak and still weak
    return "improving"        # was weak, turning up


def aggregate(rows: List[Dict], sector_by_symbol: Dict[str, str]) -> List[Dict]:
    """rows: [{symbol, ret_5d, ret_20d}] -> per-sector RRG rows (RS vs market)."""
    r5 = [r["ret_5d"] for r in rows if r.get("ret_5d") is not None]
    r20 = [r["ret_20d"] for r in rows if r.get("ret_20d") is not None]
    if not r5 or not r20:
        return []
    mkt5, mkt20 = sum(r5) / len(r5), sum(r20) / len(r20)
    by_sec: Dict[str, Dict] = {}
    for r in rows:
        sec = sector_by_symbol.get(r["symbol"])
        if not sec:
            continue
        b = by_sec.setdefault(sec, {"r5": [], "r20": []})
        if r.get("ret_5d") is not None:
            b["r5"].append(r["ret_5d"])
        if r.get("ret_20d") is not None:
            b["r20"].append(r["ret_20d"])
    out: List[Dict] = []
    for sec, b in by_sec.items():
        if not b["r5"] or not b["r20"]:
            continue
        s5, s20 = sum(b["r5"]) / len(b["r5"]), sum(b["r20"]) / len(b["r20"])
        rs_short, rs_long = round(s5 - mkt5, 2), round(s20 - mkt20, 2)
        out.append({
            "sector": sec, "count": len(b["r20"]),
            "ret_5d": round(s5, 2), "ret_20d": round(s20, 2),
            "rs_short": rs_short, "rs_long": rs_long,
            "quadrant": classify_quadrant(rs_long, rs_short),
        })
    out.sort(key=lambda x: x["rs_long"], reverse=True)
    return out


def _read_returns() -> List[Dict]:
    """Per-symbol 5d / 20d return from candles (one window query, direct PG)."""
    from ...data.ohlc_store import pg_connect
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                WITH ranked AS (
                  SELECT stock_symbol, close,
                         row_number() OVER (PARTITION BY stock_symbol
                                            ORDER BY timestamp DESC) AS rn
                  FROM candles WHERE interval='1d'
                )
                SELECT stock_symbol,
                  max(close) FILTER (WHERE rn=1)  AS c0,
                  max(close) FILTER (WHERE rn=6)  AS c5,
                  max(close) FILTER (WHERE rn=21) AS c20
                FROM ranked WHERE rn IN (1, 6, 21)
                GROUP BY stock_symbol
            """)
            out: List[Dict] = []
            for sym, c0, c5, c20 in cur.fetchall():
                c0 = float(c0) if c0 else None
                c5 = float(c5) if c5 else None
                c20 = float(c20) if c20 else None
                out.append({
                    "symbol": sym,
                    "ret_5d": round((c0 / c5 - 1) * 100, 2) if (c0 and c5) else None,
                    "ret_20d": round((c0 / c20 - 1) * 100, 2) if (c0 and c20) else None,
                })
            return out
    finally:
        conn.close()


def _sector_map() -> Dict[str, str]:
    from ...core.database import get_supabase_admin
    sb = get_supabase_admin()
    out: Dict[str, str] = {}
    start = 0
    while True:
        chunk = (sb.table("instruments").select("symbol,sector")
                 .eq("instrument_type", "EQ").range(start, start + 999).execute().data or [])
        for r in chunk:
            if r.get("sector"):
                out[r["symbol"]] = r["sector"]
        if len(chunk) < 1000:
            break
        start += 1000
    return out


def sector_rotation() -> List[Dict]:
    hit = _CACHE.get("rot")
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]
    try:
        out = aggregate(_read_returns(), _sector_map())
        if out:
            _CACHE["rot"] = (time.monotonic(), out)
        return out
    except Exception as e:
        logger.debug("sector_rotation failed: %s", e)
        return []
