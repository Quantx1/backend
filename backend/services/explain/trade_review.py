"""Per-trade AI Trade Review — the post-mortem on a single closed trade.

Assembles REAL facts from the closed-trade row deterministically (entry/exit,
P&L, hold duration, exit reason, and risk metrics like R-multiple when a stop is
present), turns them into plain-English review bullets (always returned, 0
tokens), then OPTIONALLY narrates over them with the grounded agent (free-first
model, cached per trade) only when the user clicks "Get AI review".

`assemble_trade_facts` + `_review_points` are pure over a trade dict (tested);
`review_trade` is the reader that loads the row scoped to the user and wires the
optional narrative. Honest-empty: no closed trade → None.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _num(v: Any) -> Optional[float]:
    """Coerce to float, or None if missing/unparseable. Never fabricates."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _humanize_duration(opened: Optional[datetime], closed: Optional[datetime]) -> Optional[str]:
    """Plain-English hold duration. None when either timestamp is missing."""
    if not (opened and closed):
        return None
    secs = (closed - opened).total_seconds()
    if secs < 0:
        return None
    mins = secs / 60.0
    if mins < 60:
        m = int(round(mins))
        return f"{m} min" if m != 1 else "1 min"
    hours = mins / 60.0
    if hours < 24:
        h = int(round(hours))
        return f"{h} hr" if h != 1 else "1 hr"
    days = int(round(hours / 24.0))
    return f"{days} days" if days != 1 else "1 day"


def _signal_ref(trade: Dict[str, Any], key: str) -> Optional[float]:
    """Pull an originating-signal field from either a nested ``signals`` dict or
    a flat ``signal_<key>`` column. Returns None when absent — never invents."""
    sig = trade.get("signals")
    if isinstance(sig, dict) and sig.get(key) is not None:
        return _num(sig.get(key))
    return _num(trade.get(f"signal_{key}"))


def assemble_trade_facts(trade: Dict[str, Any]) -> Dict[str, Any]:
    """Real, deterministic facts about ONE closed trade. Only includes fields
    that exist on the row; never fabricates a number."""
    facts: Dict[str, Any] = {}

    sym = trade.get("symbol")
    if sym:
        facts["symbol"] = str(sym).upper()
    side = trade.get("direction")
    if side:
        facts["side"] = str(side).upper()
    if trade.get("exit_reason"):
        facts["exit_reason"] = str(trade.get("exit_reason"))

    entry = _num(trade.get("entry_price"))
    exit_p = _num(trade.get("exit_price"))
    if entry is not None:
        facts["entry_price"] = round(entry, 2)
    if exit_p is not None:
        facts["exit_price"] = round(exit_p, 2)

    gross = _num(trade.get("gross_pnl"))
    net = _num(trade.get("net_pnl"))
    if gross is not None:
        facts["gross_pnl"] = round(gross, 2)
    if net is not None:
        facts["net_pnl"] = round(net, 2)

    # P&L % — prefer the stored column, else derive from entry/exit + side.
    pnl_pct = _num(trade.get("pnl_percent"))
    if pnl_pct is None and entry and exit_p is not None and entry != 0:
        raw = (exit_p - entry) / entry * 100.0
        is_short = facts.get("side") == "SHORT"
        pnl_pct = -raw if is_short else raw
    if pnl_pct is not None:
        facts["pnl_pct"] = round(pnl_pct, 2)

    # hold duration — opened_at, else executed_at, else created_at.
    opened = _parse_ts(
        trade.get("opened_at") or trade.get("executed_at") or trade.get("created_at")
    )
    closed = _parse_ts(trade.get("closed_at"))
    dur = _humanize_duration(opened, closed)
    if dur:
        facts["hold_duration"] = dur

    # risk metrics — only when a real stop is present.
    stop = _num(trade.get("stop_loss"))
    if stop is not None and entry is not None and entry != 0:
        is_short = facts.get("side") == "SHORT"
        # risk per unit = distance from entry to stop (always positive).
        risk_per_unit = (entry - stop) if not is_short else (stop - entry)
        stop_dist_pct = abs((entry - stop) / entry) * 100.0
        facts["stop_distance_pct"] = round(stop_dist_pct, 2)
        # R-multiple from the % move when we have it (independent of qty).
        if risk_per_unit and risk_per_unit > 0 and exit_p is not None:
            move_per_unit = (exit_p - entry) if not is_short else (entry - exit_p)
            facts["r_multiple"] = round(move_per_unit / risk_per_unit, 2)

    # entry quality — how close the actual entry was to the signal's entry.
    sig_entry = _signal_ref(trade, "entry_price")
    if sig_entry is not None and entry is not None and sig_entry != 0:
        facts["entry_quality"] = {
            "signal_entry": round(sig_entry, 2),
            "actual_entry": round(entry, 2),
            "slippage_pct": round(abs((entry - sig_entry) / sig_entry) * 100.0, 2),
        }
    sig_stop = _signal_ref(trade, "stop_loss")
    if sig_stop is not None:
        facts.setdefault("signal", {})["stop_loss"] = round(sig_stop, 2)
    sig_target = _signal_ref(trade, "target") or _signal_ref(trade, "target_1")
    if sig_target is not None:
        facts.setdefault("signal", {})["target"] = round(sig_target, 2)

    return facts


def _review_points(facts: Dict[str, Any]) -> List[str]:
    """Deterministic plain-English review bullets — always available, 0 tokens."""
    out: List[str] = []

    reason = (facts.get("exit_reason") or "").lower()
    pnl_pct = facts.get("pnl_pct")
    pct_tail = ""
    if pnl_pct is not None:
        pct_tail = f" ({'+' if pnl_pct >= 0 else ''}{pnl_pct}%)"

    if "target" in reason:
        out.append(f"Hit target{pct_tail}.")
    elif "stop" in reason or reason in ("sl", "stoploss", "stop_loss"):
        out.append(f"Stopped out{pct_tail}.")
    elif reason:
        label = reason.replace("_", " ")
        out.append(f"Exited on {label}{pct_tail}.")
    elif pnl_pct is not None:
        out.append(f"Closed {'up' if pnl_pct >= 0 else 'down'}{pct_tail}.")

    net = facts.get("net_pnl")
    if net is not None:
        out.append(f"Net P&L {'+' if net >= 0 else ''}{net}.")

    if facts.get("hold_duration"):
        out.append(f"Held {facts['hold_duration']}.")

    r = facts.get("r_multiple")
    if r is not None:
        if r >= 0:
            out.append(f"Realized {r}R.")
        else:
            out.append(f"Lost {abs(round(r, 2))}R.")

    sd = facts.get("stop_distance_pct")
    if sd is not None:
        out.append(f"Stop was {sd}% from entry.")

    eq = facts.get("entry_quality") or {}
    if eq.get("slippage_pct") is not None:
        sl = eq["slippage_pct"]
        if sl <= 0.1:
            out.append("Entered right at the signal price.")
        else:
            out.append(f"Entry within {sl}% of the signal.")

    return out


def review_trade(trade_id: str, user_id: str, *, use_llm: bool = False) -> Optional[Dict[str, Any]]:
    """Load the user's CLOSED trade by id, build {trade, facts, points,
    narrative}. Narrative is the grounded agent, cached per trade, only when
    ``use_llm``. Honest-empty: no such closed trade → None."""
    trade: Optional[Dict[str, Any]] = None
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        # Same load shape as the trades/positions endpoints: scope by id +
        # user_id and read the originating signal's entry/stop/target so we can
        # score entry quality.
        res = (sb.table("trades")
               .select("*, signals(entry_price, stop_loss, target_1)")
               .eq("id", trade_id).eq("user_id", user_id)
               .eq("status", "closed").limit(1).execute())
        rows = res.data or []
        trade = rows[0] if rows else None
    except Exception as e:
        logger.debug("review_trade read failed for %s: %s", trade_id, e)
        return None

    if not trade:
        return None

    facts = assemble_trade_facts(trade)
    points = _review_points(facts)

    narrative = None
    if use_llm and points:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            facts,
            "Review this closed trade for the trader: what worked, what didn't, "
            "entry quality, and risk-management execution. 3-4 tight sentences.",
            cache_key="tradereview:" + str(trade_id),
            role="responder",
            user_id=user_id,
        )

    return {"trade": trade, "facts": facts, "points": points, "narrative": narrative}
