"""
Weekly portfolio review generator (N10).

For each Pro+ user, aggregate the past 7 days into a structured brief:

    - Closed trades  (count, win-rate, total P&L, best+worst)
    - Open positions (count, unrealized P&L)
    - Signals seen   (count, followed ratio)
    - Regime transitions detected by Regime
    - Portfolio performance vs Nifty

Then hand the brief to the LLM (via the OpenRouter gateway) which writes
a ~300-word second-person review. The review is stored in
``user_weekly_reviews`` keyed by
(user_id, week_of). The week_of anchor is the Monday of the reviewed
week so downstream queries are unambiguous.

Fallback: when the LLM is unavailable (no key, quota, transient),
a deterministic rule-based narrative is used instead so users still
get something. ``source`` in the stored row flags which path produced it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from ...data.market_calendar import IST

logger = logging.getLogger(__name__)


def _monday_of_week(today: Optional[date] = None) -> date:
    today = today or datetime.now(IST).date()
    return today - timedelta(days=today.weekday())


@dataclass
class WeeklyReview:
    user_id: str
    week_of: str              # ISO Monday
    content_markdown: str
    week_return_pct: Optional[float]
    nifty_return_pct: Optional[float]
    stats: Dict[str, Any] = field(default_factory=dict)
    source: str = "rule_based"   # 'llm' | 'rule_based'


# ============================================================================
# Aggregation
# ============================================================================


def _closed_trades_stats(sb, user_id: str, since: str) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("trades")
            .select("id, symbol, status, net_pnl, pnl_percent, closed_at, entry_price, exit_price")
            .eq("user_id", user_id)
            .eq("status", "closed")
            .gte("closed_at", since)
            .limit(200)
            .execute()
        )
        data = rows.data or []
    except Exception as exc:
        logger.debug("closed-trades lookup failed %s: %s", user_id, exc)
        data = []
    if not data:
        return {"count": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "total_return_pct": 0.0,
                "best": None, "worst": None}
    wins = [r for r in data if (r.get("pnl_percent") or 0) > 0]
    losses = [r for r in data if (r.get("pnl_percent") or 0) <= 0]
    best = max(data, key=lambda r: float(r.get("pnl_percent") or 0))
    worst = min(data, key=lambda r: float(r.get("pnl_percent") or 0))
    net_pnl = sum(float(r.get("net_pnl") or 0) for r in data)
    total_pct = sum(float(r.get("pnl_percent") or 0) for r in data)
    return {
        "count": len(data),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(data), 3) if data else 0.0,
        "net_pnl": round(net_pnl, 2),
        "total_return_pct": round(total_pct, 2),
        "best": {
            "symbol": best.get("symbol"),
            "pnl_percent": round(float(best.get("pnl_percent") or 0), 2),
        },
        "worst": {
            "symbol": worst.get("symbol"),
            "pnl_percent": round(float(worst.get("pnl_percent") or 0), 2),
        },
    }


def _open_positions_stats(sb, user_id: str) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("positions")
            .select("symbol, quantity, average_price, current_price, unrealized_pnl")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .limit(50)
            .execute()
        )
        data = rows.data or []
    except Exception:
        data = []
    unreal = sum(float(r.get("unrealized_pnl") or 0) for r in data)
    return {"count": len(data), "unrealized_pnl": round(unreal, 2)}


def _regime_transitions(sb, since: str) -> List[Dict[str, Any]]:
    try:
        rows = (
            sb.table("regime_history")
            .select("regime, detected_at, vix")
            .gte("detected_at", since)
            .order("detected_at")
            .execute()
        )
    except Exception:
        return []
    transitions: List[Dict[str, Any]] = []
    prev = None
    for r in rows.data or []:
        if prev and r.get("regime") != prev.get("regime"):
            transitions.append({
                "from": prev.get("regime"),
                "to": r.get("regime"),
                "at": r.get("detected_at"),
            })
        prev = r
    return transitions[-3:]  # last 3 max


def _nifty_return_7d(sb, since: str) -> Optional[float]:
    try:
        rows = (
            sb.table("regime_history")
            .select("nifty_close, detected_at")
            .gte("detected_at", since)
            .order("detected_at")
            .execute()
        )
        data = [r for r in (rows.data or []) if r.get("nifty_close")]
        if len(data) < 2:
            return None
        first, last = float(data[0]["nifty_close"]), float(data[-1]["nifty_close"])
        if first <= 0:
            return None
        return round(((last - first) / first) * 100.0, 2)
    except Exception:
        return None


def _signals_stats(sb, user_id: str, since: str) -> Dict[str, Any]:
    # Rough heuristic: how many signals were published in the window, and
    # how many the user executed trades on.
    try:
        sig_rows = (
            sb.table("signals")
            .select("id", count="exact")
            .gte("created_at", since)
            .execute()
        )
        signals_seen = int(getattr(sig_rows, "count", 0) or 0)
    except Exception:
        signals_seen = 0
    try:
        follow_rows = (
            sb.table("trades")
            .select("signal_id")
            .eq("user_id", user_id)
            .gte("created_at", since)
            .not_.is_("signal_id", "null")
            .execute()
        )
        followed = len({r["signal_id"] for r in (follow_rows.data or []) if r.get("signal_id")})
    except Exception:
        followed = 0
    return {"signals_seen": signals_seen, "signals_followed": followed}


# ============================================================================
# Narrative
# ============================================================================


async def _llm_narrative(stats: Dict[str, Any]) -> Optional[str]:
    try:
        from ..agents.llm import LLM
    except Exception as exc:
        logger.debug("LLM import failed: %s", exc)
        return None
    llm = LLM()
    if not llm.enabled:
        return None

    prompt = f"""Write a ~300-word weekly portfolio review for a retail Indian trader.

Use the second person ("you"). Be specific, direct, and encouraging without cheerleading.
Cover in order:
  1. One-sentence headline of the week (wins, losses, regime).
  2. Closed-trade performance: count, win-rate, best + worst.
  3. Benchmark vs Nifty (if available).
  4. Regime context from Regime (if transitions detected).
  5. One observation + one suggestion for next week.

Do NOT mention specific AI model names like TFT, Qlib, FinBERT, HMM, LightGBM, FinRL, XGBoost, Chronos, TimesFM, LSTM, or BiLSTM.
Use the internal engine names: Forecast, Alpha, Intraday, Trajectory, Regime, Mood, EarningsScout.
No disclaimers or "Dear [Name]".

Numbers:
{stats}
"""
    try:
        text = await llm.complete(prompt, temperature=0.4, top_p=0.9)
        text = (text or "").strip()
        # Guard against refusals.
        if len(text) < 80 or "I cannot" in text[:60].lower():
            return None
        return text
    except Exception as exc:
        logger.warning("LLM weekly review failed: %s", exc)
        return None


def _rule_based_narrative(stats: Dict[str, Any]) -> str:
    closed = stats.get("closed", {})
    positions = stats.get("positions", {})
    signals = stats.get("signals", {})
    nifty = stats.get("nifty_return_7d")
    transitions = stats.get("regime_transitions", [])

    lines: List[str] = []
    n = closed.get("count", 0)
    if n == 0:
        lines.append(
            "This week you did not close any trades. "
            "That is a fine choice when conviction is low — patience beats activity most weeks."
        )
    else:
        wr = round(closed.get("win_rate", 0) * 100)
        total = closed.get("total_return_pct", 0)
        best = closed.get("best") or {}
        worst = closed.get("worst") or {}
        lines.append(
            f"You closed {n} trade{'s' if n != 1 else ''} this week "
            f"at a {wr}% win rate, for a cumulative {total:+.2f}% return."
        )
        if best.get("symbol"):
            lines.append(
                f"Best: {best['symbol']} at {best.get('pnl_percent', 0):+.2f}%. "
                f"Worst: {worst.get('symbol', '—')} at {worst.get('pnl_percent', 0):+.2f}%."
            )

    if nifty is not None:
        week_return = closed.get("total_return_pct", 0)
        diff = week_return - nifty
        verdict = "beat" if diff > 0 else "trailed"
        lines.append(
            f"Nifty moved {nifty:+.2f}% this week. Your closed book "
            f"{verdict} the index by {abs(diff):.2f} pts."
        )

    if transitions:
        last = transitions[-1]
        lines.append(
            f"Regime flagged a {last['from']} → {last['to']} transition mid-week — "
            f"future signal sizes adjust automatically."
        )

    if positions.get("count", 0):
        lines.append(
            f"You carry {positions['count']} open position{'s' if positions['count'] != 1 else ''} "
            f"with unrealized P&L of ₹{positions.get('unrealized_pnl', 0):,.0f}."
        )

    if signals.get("signals_seen", 0):
        ratio = (
            signals.get("signals_followed", 0) / signals["signals_seen"]
            if signals["signals_seen"] else 0
        )
        lines.append(
            f"You acted on {signals.get('signals_followed', 0)} of the "
            f"{signals['signals_seen']} signals delivered this week "
            f"({round(ratio * 100)}% follow-through)."
        )

    # Suggestion — deterministic heuristic.
    if closed.get("count", 0) == 0 and signals.get("signals_seen", 0) > 0:
        lines.append(
            "Next week: try at least one paper trade from the top-confidence "
            "Forecast signal — building the execution muscle matters as much as picking."
        )
    elif closed.get("win_rate", 0) >= 0.6:
        lines.append(
            "Next week: the setup is working. Stay within your risk rails and let the edge compound."
        )
    elif closed.get("win_rate", 0) > 0:
        lines.append(
            "Next week: review the SL levels on your losing trades — "
            "tightening by 0.5x ATR often flips a losing strategy to flat."
        )
    else:
        lines.append(
            "Next week: reduce position size by 30% and focus on one sector "
            "until you hit a 3-trade winning streak."
        )

    return " ".join(lines)


# ============================================================================
# Driver
# ============================================================================


async def generate_review_for_user(
    *,
    user_id: str,
    supabase_client=None,
    week_of: Optional[date] = None,
    persist: bool = True,
    force_llm: bool = True,
) -> WeeklyReview:
    """Build this week's review for a single user. Safe to call on-demand
    (admin endpoint) or from the Sunday scheduler loop."""
    if supabase_client is None:
        from ...core.database import get_supabase_admin
        supabase_client = get_supabase_admin()

    week_of = week_of or _monday_of_week()
    # Window: Monday of the week at 00:00 IST → now.
    since = datetime.combine(week_of, datetime.min.time()).replace(tzinfo=IST).astimezone(timezone.utc).isoformat()

    closed = _closed_trades_stats(supabase_client, user_id, since)
    positions = _open_positions_stats(supabase_client, user_id)
    signals = _signals_stats(supabase_client, user_id, since)
    nifty = _nifty_return_7d(supabase_client, since)
    transitions = _regime_transitions(supabase_client, since)

    stats = {
        "closed": closed,
        "positions": positions,
        "signals": signals,
        "nifty_return_7d": nifty,
        "regime_transitions": transitions,
    }

    # AI Trading Coach behavioral flags — deterministic, best-effort. Lets the
    # narrative mention behaviors (revenge trading / overtrading / holding
    # losers too long) when present; never blocks the review on failure.
    try:
        from ...services.portfolio.coach_flags import coach_review
        behavior = await asyncio.to_thread(coach_review, user_id)
        stats["behavior_flags"] = behavior.get("flags", [])
    except Exception as exc:
        logger.debug("coach flags for weekly review failed %s: %s", user_id, exc)

    narrative: Optional[str] = None
    source = "rule_based"
    if force_llm:
        narrative = await _llm_narrative(stats)
        if narrative:
            source = "llm"
    if not narrative:
        narrative = _rule_based_narrative(stats)

    review = WeeklyReview(
        user_id=user_id,
        week_of=week_of.isoformat(),
        content_markdown=narrative,
        week_return_pct=closed.get("total_return_pct"),
        nifty_return_pct=nifty,
        stats=stats,
        source=source,
    )

    if persist:
        try:
            supabase_client.table("user_weekly_reviews").upsert({
                "user_id": user_id,
                "week_of": week_of.isoformat(),
                "content_markdown": narrative,
                "week_return_pct": closed.get("total_return_pct"),
                "nifty_return_pct": nifty,
                "generated_at": datetime.utcnow().isoformat(),
            }, on_conflict="user_id,week_of").execute()
        except Exception as exc:
            logger.error("weekly review persist failed %s: %s", user_id, exc)

    return review


async def generate_and_persist_all_pro(
    *,
    supabase_client=None,
    week_of: Optional[date] = None,
    concurrency: int = 4,
) -> Dict[str, Any]:
    """Fan-out the generator across every Pro+ user with the
    weekly_review feature opted in. Bounded concurrency to avoid
    the LLM (OpenRouter gateway) rate-limit."""
    if supabase_client is None:
        from ...core.database import get_supabase_admin
        supabase_client = get_supabase_admin()

    try:
        rows = (
            supabase_client.table("user_profiles")
            .select("id, tier, onboarding_completed")
            .in_("tier", ["pro", "elite"])
            .eq("onboarding_completed", True)
            .limit(5000)
            .execute()
        )
        users = rows.data or []
    except Exception as exc:
        logger.error("weekly review user query failed: %s", exc)
        return {"n_users": 0, "written": 0, "failed": 0}

    sem = asyncio.Semaphore(max(1, concurrency))
    written = 0
    failed = 0

    async def one(user_id: str):
        nonlocal written, failed
        async with sem:
            try:
                await generate_review_for_user(
                    user_id=user_id,
                    supabase_client=supabase_client,
                    week_of=week_of,
                    persist=True,
                )
                written += 1
            except Exception as exc:
                logger.warning("weekly review for %s failed: %s", user_id, exc)
                failed += 1

    await asyncio.gather(*[one(u["id"]) for u in users])
    return {"n_users": len(users), "written": written, "failed": failed}
