"""IV-history store + IV Rank / IV Percentile (Volatility Intelligence, #13).

ATM IV is accumulated forward (one row per symbol per day) from option-chain
snapshots; rank/percentile are computed over the trailing window. Honest: nulls
+ a `days` count until enough history exists (IV history can't be backfilled —
it needs option-chain history we only get going forward). `compute_iv_rank_
percentile` is pure (tested); the store wrappers are thin + best-effort.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_DAYS = 20  # below this, rank/percentile aren't meaningful


def compute_iv_rank_percentile(series: List[float], current_iv: float,
                               *, min_days: int = MIN_DAYS) -> Dict:
    """IV Rank = (cur-min)/(max-min)*100; IV Percentile = % of trailing days
    below current. Honest-null until `min_days` of history. Pure."""
    n = len(series)
    out: Dict = {
        "iv_rank": None, "iv_percentile": None, "days": n,
        "current_iv": round(current_iv, 4) if current_iv else None,
    }
    if not current_iv or current_iv <= 0 or n < min_days:
        return out
    lo, hi = min(series), max(series)
    out["iv_rank"] = round((current_iv - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0
    out["iv_percentile"] = round(sum(1 for v in series if v < current_iv) / n * 100, 1)
    return out


def record_atm_iv(symbol: str, atm_iv: Optional[float], *, on: Optional[date] = None) -> bool:
    """Upsert today's ATM IV for a symbol (one row per symbol per day)."""
    if not symbol or not atm_iv or atm_iv <= 0:
        return False
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        sb.table("iv_history").upsert(
            {"symbol": symbol.upper(), "trade_date": (on or date.today()).isoformat(),
             "atm_iv": round(float(atm_iv), 6), "source": "kite_snapshot"},
            on_conflict="symbol,trade_date",
        ).execute()
        return True
    except Exception as e:
        logger.debug("record_atm_iv failed for %s: %s", symbol, e)
        return False


def _read_series(symbol: str, days: int = 252) -> List[float]:
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        rows = (sb.table("iv_history").select("atm_iv,trade_date")
                .eq("symbol", symbol.upper())
                .order("trade_date", desc=True).limit(days).execute().data or [])
        return [float(r["atm_iv"]) for r in reversed(rows) if r.get("atm_iv") is not None]
    except Exception as e:
        logger.debug("read iv series failed for %s: %s", symbol, e)
        return []


def iv_rank_percentile(symbol: str, current_iv: Optional[float],
                       *, lookback_days: int = 252) -> Dict:
    """Record today's IV (forward accumulation) and return rank/percentile."""
    if current_iv:
        record_atm_iv(symbol, current_iv)
    return compute_iv_rank_percentile(_read_series(symbol, lookback_days), current_iv or 0.0)
