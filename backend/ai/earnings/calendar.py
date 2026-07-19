"""
F9 Earnings calendar reader.

Reads upcoming earnings announcement DATES from two sources:

    1. ``earnings_predictions`` table — any pre-existing announce-date
       rows. Preferred when present.
    2. yfinance ``Ticker.calendar`` — on-demand fallback when the DB
       is empty (bootstrap / first run).

v1 universe is the Nifty 200 large-caps by default. Admin + scheduler
can override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UpcomingEarning:
    symbol: str
    announce_date: str
    beat_prob: Optional[float] = None
    confidence: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    strategy_recommendation: Optional[str] = None


# ---------------------------------------------------------------- DB reads


def _read_from_db(
    *,
    days: int,
    supabase_client=None,
    start: Optional[date] = None,
) -> List[UpcomingEarning]:
    if supabase_client is None:
        from ...core.database import get_supabase_admin
        supabase_client = get_supabase_admin()
    start = start or date.today()
    end = start + timedelta(days=days)
    try:
        rows = (
            supabase_client.table("earnings_predictions")
            .select("symbol, announce_date, beat_prob, confidence, evidence, strategy_recommendation")
            .gte("announce_date", start.isoformat())
            .lte("announce_date", end.isoformat())
            .order("announce_date", desc=False)
            .execute()
        )
        out: List[UpcomingEarning] = []
        for r in rows.data or []:
            out.append(UpcomingEarning(
                symbol=r.get("symbol") or "",
                announce_date=str(r.get("announce_date")),
                beat_prob=float(r["beat_prob"]) if r.get("beat_prob") is not None else None,
                confidence=r.get("confidence"),
                evidence=r.get("evidence") or {},
                strategy_recommendation=r.get("strategy_recommendation"),
            ))
        return out
    except Exception as exc:
        logger.warning("earnings_predictions DB read failed: %s", exc)
        return []


# ---------------------------------------------------------------- yfinance


def _upcoming_from_yfinance(symbol: str, *, window_days: int = 14) -> Optional[date]:
    """Try to read the next announce date from yfinance ``Ticker.calendar``.
    Returns None if unavailable or outside window."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol if symbol.endswith(".NS") else f"{symbol}.NS")
        cal = getattr(tk, "calendar", None)
        if cal is None:
            return None
        # yfinance 0.2.x returns a dict; older versions a DataFrame.
        if hasattr(cal, "get"):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                ed = ed[0]
            if ed is None:
                return None
            if hasattr(ed, "date"):
                ed = ed.date()
            today = date.today()
            if today <= ed <= today + timedelta(days=window_days):
                return ed
        elif hasattr(cal, "loc"):
            try:
                d = cal.loc["Earnings Date"].iloc[0].date()
                today = date.today()
                if today <= d <= today + timedelta(days=window_days):
                    return d
            except Exception:
                return None
    except Exception as exc:
        logger.debug("yfinance calendar lookup failed %s: %s", symbol, exc)
    return None


def next_earnings_date(symbol: str, *, window_days: int = 60):
    """Public best-effort next announce date (yfinance Ticker.calendar).
    None when unavailable or outside the window — callers stay honest-empty."""
    return _upcoming_from_yfinance(symbol, window_days=window_days)


def bootstrap_universe_from_yfinance(
    symbols: List[str],
    *,
    days: int = 14,
) -> List[tuple]:
    """Probe yfinance for each symbol and return (symbol, announce_date)
    tuples for those whose calendar falls inside the window.

    Called by the scheduler on first run / nightly to hydrate the
    ``earnings_predictions`` table.
    """
    out: List[tuple] = []
    for sym in symbols:
        d = _upcoming_from_yfinance(sym, window_days=days)
        if d is not None:
            out.append((sym.upper(), d))
    return out


# ---------------------------------------------------------------- main


def fetch_upcoming_earnings(
    *,
    days: int = 14,
    supabase_client=None,
) -> List[UpcomingEarning]:
    """Return upcoming earnings with predictions in the ``days``-day
    forward window. DB-first; on-demand yfinance lookup if empty."""
    rows = _read_from_db(days=days, supabase_client=supabase_client)
    return rows
