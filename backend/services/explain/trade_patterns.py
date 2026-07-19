"""AI Trade Journal — behavioral pattern mining (#23).

Mines a trader's OWN closed trades for patterns the audit found missing:
win-rate by session (time-of-day), weekday, holding period, and best/worst
symbols — then a grounded one-liner ("you perform best on momentum trades
9:30-11:00"). `mine_patterns` is pure over normalized records (tested); the
reader normalizes timestamps to IST.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SESSIONS = [("Open (9:15-11)", 555, 660), ("Mid (11-1)", 660, 780), ("Close (1-3:30)", 780, 930)]
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_INTRADAY_MAX_MIN = 375  # one trading day


def _bucket(records: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    by: Dict[str, Dict[str, float]] = {}
    for r in records:
        k = key_fn(r)
        if k is None:
            continue
        b = by.setdefault(k, {"n": 0, "wins": 0, "pnl": 0.0})
        b["n"] += 1
        b["wins"] += 1 if r["pnl"] > 0 else 0
        b["pnl"] += r["pnl"]
    return [{"label": k, "n": int(v["n"]), "win_rate": round(v["wins"] / v["n"] * 100),
             "total_pnl": round(v["pnl"], 2)} for k, v in by.items()]


def mine_patterns(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """records: [{symbol, pnl, minute_of_day(IST), weekday, hold_min}]."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    wins = [r for r in records if r["pnl"] > 0]
    losses = [r for r in records if r["pnl"] <= 0]

    def session(r):
        m = r.get("minute_of_day")
        if m is None:
            return None
        for label, lo, hi in _SESSIONS:
            if lo <= m < hi:
                return label
        return None

    def weekday(r):
        w = r.get("weekday")
        return _WEEKDAYS[w] if (w is not None and 0 <= w < 5) else None

    def hold(r):
        h = r.get("hold_min")
        return None if h is None else ("Intraday" if h < _INTRADAY_MAX_MIN else "Swing (>1 day)")

    by_sym: Dict[str, Dict[str, float]] = {}
    for r in records:
        b = by_sym.setdefault(r["symbol"], {"n": 0, "pnl": 0.0})
        b["n"] += 1
        b["pnl"] += r["pnl"]
    syms = sorted(({"symbol": k, "n": int(v["n"]), "total_pnl": round(v["pnl"], 2)}
                   for k, v in by_sym.items()), key=lambda x: x["total_pnl"], reverse=True)

    return {
        "n": n,
        "win_rate": round(len(wins) / n * 100),
        "avg_win": round(sum(r["pnl"] for r in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(r["pnl"] for r in losses) / len(losses), 2) if losses else 0.0,
        "by_session": sorted(_bucket(records, session), key=lambda x: x["win_rate"], reverse=True),
        "by_weekday": _bucket(records, weekday),
        "by_hold": _bucket(records, hold),
        "best_symbols": syms[:3],
        "worst_symbols": [s for s in syms[::-1] if s["total_pnl"] < 0][:3],
    }


def _to_ist(ts: Optional[str]):
    if not ts:
        return None
    try:
        from datetime import datetime
        from ...data.market_calendar import IST
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(IST)
    except Exception:
        return None


def journal_insights(user_id: str, *, use_llm: bool = False) -> Dict[str, Any]:
    """Read the user's closed trades, mine patterns + optional grounded summary."""
    records: List[Dict[str, Any]] = []
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        rows = (sb.table("trades")
                .select("symbol,net_pnl,created_at,closed_at,status")
                .eq("user_id", user_id).eq("status", "closed")
                .order("closed_at", desc=True).limit(500).execute().data or [])
        for r in rows:
            pnl = r.get("net_pnl")
            if pnl is None:
                continue
            ent = _to_ist(r.get("created_at"))
            ext = _to_ist(r.get("closed_at"))
            records.append({
                "symbol": r.get("symbol") or "?",
                "pnl": float(pnl),
                "minute_of_day": (ent.hour * 60 + ent.minute) if ent else None,
                "weekday": ent.weekday() if ent else None,
                "hold_min": (ext - ent).total_seconds() / 60 if (ent and ext) else None,
            })
    except Exception as e:
        logger.debug("journal_insights read failed: %s", e)

    stats = mine_patterns(records)
    narrative = None
    if use_llm and stats.get("n", 0) >= 5:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            stats, "What are this trader's strongest and weakest patterns? Give one or two "
            "specific, actionable sentences (e.g. best session, best/worst symbols).")
    return {"stats": stats, "narrative": narrative}
