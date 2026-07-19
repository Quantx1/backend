"""AI Trading Coach — behavioral flags over the user's own closed trades.

Detects the three classic discipline failures with deterministic math only:

    - revenge trading      (re-opening within minutes of a losing close)
    - overtrading          (a day's trade count spiking vs the trailing median)
    - holding losers       (losers held far longer than winners)

The detectors are PURE over the same closed-trade rows the journal reader
loads (symbol / net_pnl / executed_at / created_at / closed_at) — open time is
``executed_at`` falling back to ``created_at``, mirroring trade_review. Rows
missing timestamps simply don't participate (honest fallback, never invented).
``coach_flags`` returns [] below 10 closed trades (honest minimum).
``coach_review`` is the reader (same loader shape as trade_patterns.
journal_insights) and wires the OPTIONAL grounded coaching note — the LLM only
narrates the flags; it never decides them.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MIN_TRADES = 10          # honest minimum before coaching anyone
_MIN_PER_DAY = 4          # an overtrading spike needs at least this many trades
_MIN_PRIOR_DAYS = 3       # ... and enough history for the trailing median
_MIN_POPULATION = 3       # loss-holding needs >=3 winners AND >=3 losers
_HOLD_RATIO = 1.5
_REVENGE_OCCASIONS = 2


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(ts: Any) -> Optional[datetime]:
    """ISO timestamp → aware datetime (naive assumed UTC). None when missing."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _open_ts(r: Dict[str, Any]) -> Optional[datetime]:
    """Open time = executed_at, else created_at (the fields the trades row has)."""
    return _parse_ts(r.get("executed_at") or r.get("created_at"))


def detect_revenge(records: List[Dict[str, Any]], window_minutes: int = 30) -> Dict[str, Any]:
    """Occasions where a NEW trade was opened within ``window_minutes`` after a
    LOSING close. Flagged at >= 2 occasions. Rows without timestamps are
    skipped honestly."""
    loss_closes: List[datetime] = []
    opens: List[datetime] = []
    evaluable = 0
    for r in records:
        o = _open_ts(r)
        c = _parse_ts(r.get("closed_at"))
        pnl = _num(r.get("net_pnl"))
        if o is None and c is None:
            continue
        evaluable += 1
        if o is not None:
            opens.append(o)
        if c is not None and pnl is not None and pnl < 0:
            loss_closes.append(c)

    occasions = 0
    for o in opens:
        for c in loss_closes:
            mins = (o - c).total_seconds() / 60.0
            if 0 < mins <= window_minutes:
                occasions += 1  # each new open counts once
                break
    return {
        "flagged": occasions >= _REVENGE_OCCASIONS,
        "occasions": occasions,
        "window_minutes": window_minutes,
        "evaluable": evaluable,
    }


def detect_overtrading(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Days whose trade count >= 2x the trailing median daily count (and at
    least ``_MIN_PER_DAY`` trades that day). Needs >= ``_MIN_PRIOR_DAYS`` prior
    active days so the median means something."""
    counts: Dict[Any, int] = {}
    for r in records:
        o = _open_ts(r)
        if o is None:
            continue
        d = o.date()
        counts[d] = counts.get(d, 0) + 1

    days = sorted(counts)
    spikes: List[Dict[str, Any]] = []
    for i, d in enumerate(days):
        prior = [counts[p] for p in days[:i]]
        if len(prior) < _MIN_PRIOR_DAYS:
            continue
        med = median(prior)
        if counts[d] >= _MIN_PER_DAY and med > 0 and counts[d] >= 2 * med:
            spikes.append({"day": d.isoformat(), "trades": counts[d],
                           "trailing_median": round(float(med), 1)})
    return {"flagged": bool(spikes), "spike_days": spikes, "days_observed": len(days)}


def detect_loss_holding(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Avg hold of losers >= 1.5x avg hold of winners. Needs >= 3 of each with
    real open+close timestamps; scratches (pnl == 0) belong to neither."""
    win_holds: List[float] = []
    loss_holds: List[float] = []
    for r in records:
        o = _open_ts(r)
        c = _parse_ts(r.get("closed_at"))
        pnl = _num(r.get("net_pnl"))
        if o is None or c is None or pnl is None:
            continue
        hold = (c - o).total_seconds() / 60.0
        if hold < 0:
            continue
        if pnl > 0:
            win_holds.append(hold)
        elif pnl < 0:
            loss_holds.append(hold)

    out: Dict[str, Any] = {"flagged": False, "winners": len(win_holds), "losers": len(loss_holds)}
    if len(win_holds) < _MIN_POPULATION or len(loss_holds) < _MIN_POPULATION:
        return out
    avg_w = sum(win_holds) / len(win_holds)
    avg_l = sum(loss_holds) / len(loss_holds)
    out["avg_winner_hold_min"] = round(avg_w, 1)
    out["avg_loser_hold_min"] = round(avg_l, 1)
    if avg_w > 0:
        out["ratio"] = round(avg_l / avg_w, 2)
        out["flagged"] = avg_l >= _HOLD_RATIO * avg_w
    return out


def coach_flags(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all detectors → {flags: [{key, label, detail}], stats: {...}}.
    Honest minimum: no coaching below 10 closed trades."""
    n = len(records)
    if n < _MIN_TRADES:
        return {"flags": [], "stats": {"n": n}}

    rev = detect_revenge(records)
    over = detect_overtrading(records)
    hold = detect_loss_holding(records)

    flags: List[Dict[str, str]] = []
    if rev["flagged"]:
        flags.append({
            "key": "revenge_trading",
            "label": "Revenge trading",
            "detail": (f"{rev['occasions']} times you opened a new trade within "
                       f"{rev['window_minutes']} min of closing a loser."),
        })
    if over["flagged"]:
        worst = max(over["spike_days"], key=lambda s: s["trades"])
        flags.append({
            "key": "overtrading",
            "label": "Overtrading",
            "detail": (f"{len(over['spike_days'])} day(s) at 2x+ your usual pace — "
                       f"e.g. {worst['trades']} trades on {worst['day']} vs a typical "
                       f"{worst['trailing_median']}."),
        })
    if hold["flagged"]:
        flags.append({
            "key": "holding_losers",
            "label": "Holding losers too long",
            "detail": (f"Losing trades are held {hold['ratio']}x longer than winners "
                       f"({hold['avg_loser_hold_min']} min vs {hold['avg_winner_hold_min']} min)."),
        })

    return {"flags": flags,
            "stats": {"n": n, "revenge": rev, "overtrading": over, "loss_holding": hold}}


def coach_review(user_id: str, *, use_llm: bool = False) -> Dict[str, Any]:
    """Read the user's closed trades (same loader shape as journal_insights),
    run the deterministic detectors, and OPTIONALLY narrate the flags with the
    grounded agent (user-triggered, cached per user+day)."""
    records: List[Dict[str, Any]] = []
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        rows = (sb.table("trades")
                .select("symbol,net_pnl,executed_at,created_at,closed_at,status")
                .eq("user_id", user_id).eq("status", "closed")
                .order("closed_at", desc=True).limit(500).execute().data or [])
        for r in rows:
            pnl = r.get("net_pnl")
            if pnl is None:
                continue
            records.append({
                "symbol": r.get("symbol") or "?",
                "net_pnl": float(pnl),
                "executed_at": r.get("executed_at"),
                "created_at": r.get("created_at"),
                "closed_at": r.get("closed_at"),
            })
    except Exception as e:
        logger.debug("coach_review read failed: %s", e)

    res = coach_flags(records)
    narrative = None
    if use_llm and res["flags"]:
        from ...ai.agents.grounded import grounded_reason
        from ...data.market_calendar import IST
        day = datetime.now(IST).date().isoformat()
        narrative = grounded_reason(
            {"flags": res["flags"], "stats": res["stats"]},
            "Coach this trader on these behavioral patterns — direct, kind, specific.",
            cache_key=f"coach:{user_id}:{day}",
            user_id=user_id,
        )
    return {"flags": res["flags"], "stats": res["stats"], "narrative": narrative}
