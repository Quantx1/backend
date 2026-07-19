"""
Dashboard API route — single overview endpoint that powers the
authenticated home view.

Aggregates positions, today's closed trades, top active signals, and
unread notification count in parallel-ish fashion (sequential awaits
with retry+timeout fallback so one slow Supabase query doesn't block
the entire dashboard).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dashboard"])


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_user_profile_dep():
    from .app import get_user_profile
    return get_user_profile


def _get_supabase_retry():
    from .app import supabase_query_with_retry
    return supabase_query_with_retry


@router.get("/api/dashboard/overview")
async def get_dashboard_overview(
    equity_days: int = 30,
    profile=Depends(_get_user_profile_dep()),
):
    """Aggregate the top-of-page dashboard payload.

    Returns positions + today's PnL + active signals + notification count
    + last ``equity_days`` of paper equity curve in ONE round-trip so the
    dashboard can render with a single fetch instead of 4 parallel ones.
    """
    user_id = profile["id"]
    today = date.today().isoformat()
    equity_cutoff = (date.today() - timedelta(days=equity_days)).isoformat()

    sb = _get_supabase_admin()
    supabase_query_with_retry = _get_supabase_retry()

    pos_list = await supabase_query_with_retry(
        lambda: sb.table("positions").select("*").eq("user_id", user_id).eq("is_active", True).limit(100).execute().data,
        timeout_fallback=[],
    )
    trades_data = await supabase_query_with_retry(
        lambda: sb.table("trades").select("net_pnl").eq("user_id", user_id).eq("status", "closed").gte("closed_at", today).limit(200).execute().data,
        timeout_fallback=[],
    )
    today_pnl = sum(float(t.get("net_pnl", 0) or 0) for t in trades_data)
    signals_list = await supabase_query_with_retry(
        lambda: sb.table("signals").select("*").eq("date", today).eq("status", "active").order("confidence", desc=True).limit(5).execute().data,
        timeout_fallback=[],
    )
    notif_data = await supabase_query_with_retry(
        lambda: sb.table("notifications").select("id").eq("user_id", user_id).eq("is_read", False).limit(100).execute().data,
        timeout_fallback=[],
    )
    notif_count = len(notif_data)

    # Equity curve — same query shape as /api/paper/v2/equity-curve so
    # consumers can drop in either source without remapping.
    equity_points = await supabase_query_with_retry(
        lambda: (
            sb.table("paper_snapshots")
            .select("snapshot_date, equity, cash, invested, drawdown_pct, nifty_close")
            .eq("user_id", user_id)
            .gte("snapshot_date", equity_cutoff)
            .order("snapshot_date", desc=False)
            .execute()
            .data
        ),
        timeout_fallback=[],
    )

    unrealized_pnl = sum(float(p.get("unrealized_pnl", 0) or 0) for p in pos_list)
    total_trades = profile.get("total_trades") or 0
    winning_trades = profile.get("winning_trades") or 0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        "stats": {
            # Defensive fallback only — the signup trigger seeds
            # ``user_profiles.capital = 100000`` (₹1L), so this default
            # should never actually fire. Previously this was 500000
            # which silently disagreed with the DB default and would
            # show the user a different starting figure if the row
            # ever rendered before the trigger settled.
            "capital": profile.get("capital", 100000),
            "total_pnl": profile.get("total_pnl", 0),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": round(win_rate, 2),
            "open_positions": len(pos_list),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "today_pnl": round(today_pnl, 2),
            "subscription_status": profile.get("subscription_status", "active"),
        },
        "recent_signals": signals_list,
        "active_positions": pos_list[:5],
        "notifications_count": notif_count,
        "equity_curve": {
            "days": equity_days,
            "points": equity_points or [],
        },
    }
