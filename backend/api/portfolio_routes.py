"""
Portfolio API routes — summary, history, performance metrics.

Read-only views over ``positions`` (live) + ``portfolio_history``
(daily snapshots) + ``trades`` (closed-trade aggregations).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Portfolio"])


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


def _get_user_profile_dep():
    from .app import get_user_profile
    return get_user_profile


@router.get("/api/portfolio")
async def get_portfolio(profile=Depends(_get_user_profile_dep())):
    """Get portfolio summary — capital deployed/available, unrealized P&L,
    positions split by segment."""
    try:
        supabase = _get_supabase_admin()
        user_id = profile["id"]

        positions = (
            supabase.table("positions")
            .select("*")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .limit(100)
            .execute()
        )
        pos_list = positions.data or []

        total_invested = sum(p["quantity"] * p["average_price"] for p in pos_list)
        total_current = sum(
            p["quantity"] * (p["current_price"] or p["average_price"])
            for p in pos_list
        )
        unrealized_pnl = total_current - total_invested
        margin_used = sum(p.get("margin_used", 0) or 0 for p in pos_list)

        return {
            "capital": profile["capital"],
            "deployed": round(total_invested, 2),
            "available": round(profile["capital"] - total_invested, 2),
            "margin_used": round(margin_used, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "positions": pos_list,
            "equity_positions": [p for p in pos_list if p["segment"] == "EQUITY"],
            "fo_positions": [p for p in pos_list if p["segment"] in ["FUTURES", "OPTIONS"]],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolio/history")
async def get_portfolio_history(
    days: int = 30,
    user=Depends(_get_current_user_dep()),
):
    """Get portfolio history — daily equity snapshots over the last N days."""
    try:
        supabase = _get_supabase_admin()
        start_date = (date.today() - timedelta(days=days)).isoformat()

        result = (
            supabase.table("portfolio_history")
            .select("*")
            .eq("user_id", user.id)
            .gte("date", start_date)
            .order("date")
            .limit(365)
            .execute()
        )
        return {"history": result.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolio/performance")
async def get_performance_metrics(user=Depends(_get_current_user_dep())):
    """Get portfolio performance metrics from closed trades —
    win rate, avg win/loss, profit factor, best/worst trade."""
    try:
        supabase = _get_supabase_admin()

        trades = (
            supabase.table("trades")
            .select("*")
            .eq("user_id", user.id)
            .eq("status", "closed")
            .order("closed_at", desc=True)
            .limit(1000)
            .execute()
        )

        if not trades.data:
            return {
                "total_trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                "profit_factor": 0, "total_pnl": 0, "best_trade": 0, "worst_trade": 0,
            }

        t_list = trades.data
        winners = [t for t in t_list if (t.get("net_pnl") or 0) > 0]
        losers = [t for t in t_list if (t.get("net_pnl") or 0) < 0]

        total_wins = sum(t.get("net_pnl", 0) for t in winners)
        total_losses = abs(sum(t.get("net_pnl", 0) for t in losers))

        return {
            "total_trades": len(t_list),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(t_list) * 100, 2) if t_list else 0,
            "avg_win": round(total_wins / len(winners), 2) if winners else 0,
            "avg_loss": round(total_losses / len(losers), 2) if losers else 0,
            "profit_factor": round(total_wins / total_losses, 2) if total_losses > 0 else 0,
            "total_pnl": round(sum(t.get("net_pnl", 0) for t in t_list), 2),
            "best_trade": round(max(t.get("net_pnl", 0) for t in t_list), 2),
            "worst_trade": round(min(t.get("net_pnl", 0) for t in t_list), 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
