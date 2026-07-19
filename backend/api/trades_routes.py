"""
Trades API routes — list, execute, approve, close, kill-switch.

Most complex chunk that came out of app.py: ``execute`` runs the full
risk-engine pipeline (signal-quality, portfolio-limit, loss-limit
checks → position sizing → trade insert → optional immediate
execution). ``kill-switch`` is the highest-stakes user action on the
platform — closes every active position and pauses trading.

The shared ``close_trade_record`` helper is module-public (no leading
underscore) because the positions/close endpoint also calls it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..core.config import settings
from ..schemas import CloseTrade, ExecuteTrade
from ..trading.risk import (
    Direction,
    RiskManagementEngine,
    RISK_PROFILES,
    Segment,
    Signal as RiskSignal,
)
from ..trading.execution import TradeExecutionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Trades"])


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


def _get_user_profile_dep():
    from .app import get_user_profile
    return get_user_profile


@router.get("/api/trades")
async def get_trades(
    status: Optional[str] = None,
    segment: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    user=Depends(_get_current_user_dep()),
):
    """Get user trades."""
    try:
        supabase = _get_supabase_admin()
        query = (
            supabase.table("trades")
            .select("*, signals(symbol, direction, confidence)")
            .eq("user_id", user.id)
        )
        if status:
            query = query.eq("status", status)
        if segment:
            query = query.eq("segment", segment)
        result = query.order("created_at", desc=True).limit(limit).execute()
        return {"trades": result.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/trades/journal-insights")
async def journal_insights_route(
    use_llm: bool = Query(default=False),
    user=Depends(_get_current_user_dep()),
):
    """AI Trade Journal pattern mining (#23) — win-rate by session / weekday /
    holding period + best/worst symbols from the user's own closed trades, plus
    an optional grounded one-liner of the strongest/weakest patterns."""
    import asyncio
    from ..services.explain.trade_patterns import journal_insights
    return await asyncio.to_thread(journal_insights, user.id, use_llm=use_llm)


@router.get("/api/trades/coach")
async def coach_review_route(
    use_llm: bool = Query(default=False),
    user=Depends(_get_current_user_dep()),
):
    """AI Trading Coach — deterministic behavioral flags (revenge trading,
    overtrading, holding losers too long) mined from the user's own closed
    trades, plus an optional grounded coaching note (user-triggered)."""
    import asyncio
    from ..services.portfolio.coach_flags import coach_review
    return await asyncio.to_thread(coach_review, user.id, use_llm=use_llm)


@router.get("/api/risk/status")
async def get_risk_status(user=Depends(_get_current_user_dep())):
    """User-level Risk Manager — deterministic status warnings.

    WARN ONLY: day-loss vs the user's daily limit, single-name >20%,
    sector >40%, and total exposure >100% of capital. Never blocks,
    sizes, or gates anything — the frontend renders these as amber
    lines and the order button always stays live."""
    import asyncio
    from ..services.portfolio.user_risk import risk_status
    return await asyncio.to_thread(risk_status, str(user.id))


@router.get("/api/trades/{trade_id}/analysis")
async def trade_review_route(
    trade_id: str,
    use_llm: bool = Query(default=False),
    user=Depends(_get_current_user_dep()),
):
    import asyncio
    from ..services.explain.trade_review import review_trade
    res = await asyncio.to_thread(review_trade, trade_id, user.id, use_llm=use_llm)
    if res is None:
        raise HTTPException(status_code=404, detail="Closed trade not found")
    return res


@router.post("/api/trades/execute")
async def execute_trade(data: ExecuteTrade, profile=Depends(_get_user_profile_dep())):
    """Execute a trade from a signal — runs the full risk pipeline."""
    # PR 96 — pulled out of the try block so the failure-path observability
    # hooks below can reference it even when the body raises early.
    user_id = str(profile.get("id") or "")
    execution_mode = "paper"
    try:
        supabase = _get_supabase_admin()

        # Phase 1.7 audit fix #1.6 — fail-fast on the GLOBAL kill switch
        # before any per-user state is even read. The auto-trader honors
        # this through TradeExecutionService.execute() (see service line
        # ~56), but the synchronous REST `/api/trades/execute` path
        # never consulted it. With this gap, an admin could flip the
        # global halt and the REST endpoint would happily route trades
        # to the live broker until the cache TTL rolled.
        from ..platform.system_flags import is_globally_halted, global_halt_reason
        if is_globally_halted(supabase_client=supabase):
            reason = (
                global_halt_reason(supabase_client=supabase)
                or "Live trading is currently halted by ops."
            )
            raise HTTPException(status_code=503, detail=reason)

        signal = supabase.table("signals").select("*").eq("id", data.signal_id).single().execute()
        if not signal.data:
            raise HTTPException(status_code=404, detail="Signal not found")
        sig = signal.data

        if sig.get("is_premium") and profile.get("subscription_status") not in ["active", "trial"]:
            raise HTTPException(status_code=403, detail="Premium subscription required")
        if sig["segment"] in ["FUTURES", "OPTIONS"] and not profile.get("fo_enabled"):
            raise HTTPException(status_code=403, detail="F&O trading not enabled")
        if profile["trading_mode"] == "signal_only":
            raise HTTPException(status_code=400, detail="Auto-trading not enabled")

        positions = (
            supabase.table("positions")
            .select("id")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .execute()
        )
        plan = profile.get("subscription_plans") or {}
        max_positions = plan.get("max_positions", profile.get("max_positions", 5))
        if len(positions.data) >= max_positions:
            raise HTTPException(status_code=400, detail=f"Max positions ({max_positions}) reached")

        if profile.get("kill_switch_active"):
            raise HTTPException(status_code=400, detail="Kill switch active. Trading paused.")
        paper_start = profile.get("paper_trading_started_at") or profile.get("created_at")
        paper_start_dt = (
            datetime.fromisoformat(paper_start.replace("Z", "+00:00"))
            if isinstance(paper_start, str)
            else paper_start
        )
        days_elapsed = (datetime.utcnow() - paper_start_dt).days if paper_start_dt else 0
        eligible_live = (
            profile.get("live_trading_whitelisted", False)
            and days_elapsed >= settings.PAPER_TRADE_DAYS
        )
        execution_mode = (
            "live"
            if (eligible_live or not settings.LIVE_TRADING_WHITELIST_ONLY)
            else "paper"
        )

        risk_engine = RiskManagementEngine(supabase)
        risk_profile = RISK_PROFILES.get(
            profile.get("risk_profile", "moderate"),
            RISK_PROFILES["moderate"],
        )

        entry_price = float(sig["entry_price"])
        stop_loss = float(data.custom_sl or sig["stop_loss"])
        target = float(data.custom_target or sig["target_1"])

        signal_obj = RiskSignal(
            symbol=sig["symbol"],
            segment=Segment[sig["segment"]],
            direction=Direction.LONG if sig["direction"] == "LONG" else Direction.SHORT,
            confidence=float(sig["confidence"]),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            lot_size=int(sig.get("lot_size") or 1),
            expiry=sig.get("expiry_date"),
            strike=sig.get("strike_price"),
            option_type=sig.get("option_type"),
        )

        ok, msg = risk_engine.check_signal_quality(signal_obj, risk_profile)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        ok, msg = await risk_engine.check_portfolio_limits(user_id, signal_obj, risk_profile)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        ok, msg = await risk_engine.check_loss_limits(user_id, risk_profile)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)

        # Available margin (only for live)
        available_margin = None
        if execution_mode == "live":
            try:
                from ..data.brokers.integration import BrokerFactory
                from ..data.brokers.credentials import decrypt_credentials
                conn = (
                    supabase.table("broker_connections")
                    .select("broker_name, access_token")
                    .eq("user_id", user_id)
                    .eq("status", "connected")
                    .single()
                    .execute()
                )
                if conn.data:
                    broker = BrokerFactory.create(
                        conn.data["broker_name"],
                        decrypt_credentials(conn.data["access_token"]),
                    )
                    if broker.login():
                        available_margin = broker.get_available_margin()
            except Exception as e:
                logger.warning(f"Margin fetch failed: {e}")

        capital = float(profile["capital"])
        pos = risk_engine.calculate_position_size(signal_obj, capital, risk_profile, available_margin)
        if not pos.approved:
            raise HTTPException(status_code=400, detail=pos.rejection_reason or "Position rejected")

        quantity = data.quantity or pos.quantity
        lots = data.lots or pos.lots
        margin_used = pos.margin_required

        trade = {
            "user_id": user_id,
            "signal_id": data.signal_id,
            "symbol": sig["symbol"],
            "exchange": sig.get("exchange", "NSE"),
            "segment": sig["segment"],
            "expiry_date": sig.get("expiry_date"),
            "strike_price": sig.get("strike_price"),
            "option_type": sig.get("option_type"),
            "lot_size": sig.get("lot_size"),
            "lots": lots,
            "direction": sig["direction"],
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_amount": pos.risk_amount,
            "position_value": quantity * entry_price,
            "margin_used": margin_used,
            "product_type": "CNC" if sig["segment"] == "EQUITY" else "NRML",
            "execution_mode": execution_mode,
            "status": "pending" if profile["trading_mode"] == "semi_auto" else "open",
        }

        result = supabase.table("trades").insert(trade).execute()
        trade_id = result.data[0]["id"]

        # Full-auto: create position immediately
        if profile["trading_mode"] == "full_auto":
            trade_executor = TradeExecutionService(_get_supabase_admin())
            if execution_mode == "live":
                await trade_executor.execute({**trade, "id": trade_id, "execution_mode": "live"})
            else:
                position = {
                    "user_id": user_id,
                    "trade_id": trade_id,
                    "symbol": sig["symbol"],
                    "exchange": sig.get("exchange", "NSE"),
                    "segment": sig["segment"],
                    "expiry_date": sig.get("expiry_date"),
                    "strike_price": sig.get("strike_price"),
                    "option_type": sig.get("option_type"),
                    "direction": sig["direction"],
                    "quantity": quantity,
                    "lots": lots,
                    "average_price": entry_price,
                    "current_price": entry_price,
                    "stop_loss": stop_loss,
                    "target": target,
                    "margin_used": margin_used,
                    "execution_mode": "paper",
                    "is_active": True,
                }
                supabase.table("positions").insert(position).execute()
                supabase.table("trades").update({
                    "status": "open",
                    "executed_at": datetime.utcnow().isoformat(),
                }).eq("id", trade_id).execute()

        # PR 96 — observability for trade execution. Branches by execution_mode
        # so the paper/live cohort split lines up with what the tier gate
        # actually allowed.
        try:
            from ..observability import EventName, track
            event = (
                EventName.SIGNAL_EXECUTED_LIVE
                if execution_mode == "live"
                else EventName.SIGNAL_EXECUTED_PAPER
            )
            track(event, user_id, {
                "success": True,
                "signal_id": str(data.signal_id),
                "trade_id": str(trade_id),
                "symbol": sig.get("symbol"),
                "direction": sig.get("direction"),
                "execution_mode": execution_mode,
                "quantity": quantity,
                "entry_price": float(entry_price or 0),
            })
        except Exception:
            pass

        return {
            "success": True,
            "trade_id": trade_id,
            "status": trade["status"],
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
        }
    except HTTPException as e:
        # PR 96 — log tier-gate / risk-engine refusals.
        try:
            from ..observability import EventName, track
            track(EventName.SIGNAL_EXECUTED_PAPER, user_id, {
                "success": False,
                "signal_id": str(getattr(data, "signal_id", "")),
                "blocked_reason": str(e.detail)[:200],
                "status_code": e.status_code,
            })
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        try:
            from ..observability import EventName, track
            track(EventName.SIGNAL_EXECUTED_PAPER, user_id, {
                "success": False,
                "signal_id": str(getattr(data, "signal_id", "")),
                "error": str(e)[:300],
            })
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/trades/{trade_id}/approve")
async def approve_trade(trade_id: str, user=Depends(_get_current_user_dep())):
    """Approve pending trade (semi-auto mode)."""
    try:
        supabase = _get_supabase_admin()

        # Phase 1.7 audit fix #1.6 — global kill switch also gates the
        # semi-auto approve path. TradeExecutionService.execute() honors
        # it but PAPER approves bypassed the service entirely.
        from ..platform.system_flags import is_globally_halted, global_halt_reason
        if is_globally_halted(supabase_client=supabase):
            reason = (
                global_halt_reason(supabase_client=supabase)
                or "Live trading is currently halted by ops."
            )
            raise HTTPException(status_code=503, detail=reason)

        trade = (
            supabase.table("trades")
            .select("*")
            .eq("id", trade_id)
            .eq("user_id", user.id)
            .single()
            .execute()
        )
        if not trade.data:
            raise HTTPException(status_code=404, detail="Trade not found")
        if trade.data["status"] != "pending":
            raise HTTPException(status_code=400, detail="Trade not pending")

        t = trade.data
        if t.get("execution_mode") == "live":
            trade_executor = TradeExecutionService(_get_supabase_admin())
            await trade_executor.execute({**t, "execution_mode": "live"})
        else:
            position = {
                "user_id": user.id,
                "trade_id": trade_id,
                "symbol": t["symbol"],
                "exchange": t.get("exchange", "NSE"),
                "segment": t["segment"],
                "expiry_date": t.get("expiry_date"),
                "strike_price": t.get("strike_price"),
                "option_type": t.get("option_type"),
                "direction": t["direction"],
                "quantity": t["quantity"],
                "lots": t.get("lots", 1),
                "average_price": t["entry_price"],
                "current_price": t["entry_price"],
                "stop_loss": t["stop_loss"],
                "target": t["target"],
                "margin_used": t.get("margin_used", 0),
                "execution_mode": "paper",
                "is_active": True,
            }
            supabase.table("positions").insert(position).execute()

            supabase.table("trades").update({
                "status": "open",
                "approved_at": datetime.utcnow().isoformat(),
                "executed_at": datetime.utcnow().isoformat(),
            }).eq("id", trade_id).execute()

        return {"success": True, "message": "Trade approved and executed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def close_trade_record(
    trade_id: str, data: CloseTrade, user_id: str,
) -> Dict[str, Any]:
    """Shared close-trade logic used by trades and positions endpoints.

    PR 99 — observability: tracks POSITION_CLOSED on success (both the
    live-broker branch and the paper / manual P&L branch) and on
    failure paths so the cohort report shows attempted vs successful
    exits. The two wrapper endpoints
    (``/api/trades/{id}/close`` + ``/api/positions/{id}/close``)
    delegate here, so instrumenting once covers both surfaces.
    """
    supabase = _get_supabase_admin()

    try:
        trade = (
            supabase.table("trades")
            .select("*")
            .eq("id", trade_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not trade.data or trade.data["status"] != "open":
            raise HTTPException(status_code=400, detail="Trade not found or not open")

        t = trade.data
        if t.get("execution_mode") == "live":
            pos = (
                supabase.table("positions")
                .select("*")
                .eq("trade_id", trade_id)
                .eq("is_active", True)
                .single()
                .execute()
            )
            if pos.data:
                trade_executor = TradeExecutionService(_get_supabase_admin())
                await trade_executor.close_position(
                    pos.data, data.exit_price or t["entry_price"], data.reason,
                )
                try:
                    from ..observability import EventName, track
                    track(EventName.POSITION_CLOSED, str(user_id), {
                        "success": True,
                        "trade_id": str(trade_id),
                        "symbol": t.get("symbol"),
                        "direction": t.get("direction"),
                        "execution_mode": "live",
                        "exit_reason": data.reason,
                    })
                except Exception:
                    pass
                return {"success": True}

        exit_price = data.exit_price or t["entry_price"]

        # Calculate P&L via the shared close-side math so this matches
        # what the scheduler writes when the same trade closes via
        # SL/target/EOD.
        from ..trading.pnl import compute_close_pnl
        result = compute_close_pnl(
            direction=t["direction"],
            entry_price=t["entry_price"],
            exit_price=exit_price,
            quantity=t["quantity"],
            segment=t["segment"],
        )
        gross_pnl = result["gross_pnl"]
        net_pnl = result["net_pnl"]
        pnl_percent = result["pnl_percent"]

        # Defense-in-depth: re-apply user_id on UPDATE even though the
        # SELECT above already gated ownership. If anyone refactors the
        # SELECT out, this filter prevents the trade close from writing
        # to another user's row. Same pattern as marketplace deployments
        # update (PR P1-22).
        supabase.table("trades").update({
            "status": "closed",
            "exit_price": exit_price,
            "gross_pnl": result["gross_pnl"],
            "charges": result["charges"],
            "net_pnl": result["net_pnl"],
            "pnl_percent": result["pnl_percent"],
            "exit_reason": data.reason,
            "closed_at": datetime.utcnow().isoformat(),
        }).eq("id", trade_id).eq("user_id", user_id).execute()

        supabase.table("positions").update({"is_active": False}).eq(
            "trade_id", trade_id
        ).eq("user_id", user_id).execute()

        try:
            from ..observability import EventName, track
            track(EventName.POSITION_CLOSED, str(user_id), {
                "success": True,
                "trade_id": str(trade_id),
                "symbol": t.get("symbol"),
                "direction": t.get("direction"),
                "execution_mode": t.get("execution_mode") or "paper",
                "exit_reason": data.reason,
                "net_pnl": round(net_pnl, 2),
                "pnl_percent": round(pnl_percent, 2),
            })
        except Exception:
            pass

        return {
            "success": True,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_percent": round(pnl_percent, 2),
        }
    except HTTPException as exc:
        try:
            from ..observability import EventName, track
            track(EventName.POSITION_CLOSED, str(user_id), {
                "success": False,
                "trade_id": str(trade_id),
                "blocked_reason": str(exc.detail)[:200],
                "status_code": exc.status_code,
            })
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            from ..observability import EventName, track
            track(EventName.POSITION_CLOSED, str(user_id), {
                "success": False,
                "trade_id": str(trade_id),
                "error": str(exc)[:300],
            })
        except Exception:
            pass
        raise


@router.post("/api/trades/{trade_id}/close")
async def close_trade(
    trade_id: str,
    data: CloseTrade = CloseTrade(),
    user=Depends(_get_current_user_dep()),
):
    """Close an open trade."""
    try:
        return await close_trade_record(trade_id, data, user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/trades/kill-switch")
async def kill_switch(user=Depends(_get_current_user_dep())):
    """Emergency kill switch: close all positions and pause trading."""
    # PR 96 — observability for the highest-stakes user action on the
    # platform. Fire on both success + failure so the audit picks up
    # failed attempts (e.g., broker disconnected during liquidation).
    user_id = str(getattr(user, "id", "") or "")
    positions_processed = 0
    try:
        supabase = _get_supabase_admin()
        # Durable pause: flip BOTH flags. kill_switch_active blocks new orders
        # via compliance_gate; auto_trader_enabled=False also stops the daily
        # AutoPilot rebalance from re-opening the positions we're about to close
        # (the rebalance query filters on auto_trader_enabled=True).
        supabase.table("user_profiles").update({
            "kill_switch_active": True,
            "auto_trader_enabled": False,
        }).eq("id", user.id).execute()

        positions = (
            supabase.table("positions")
            .select("*")
            .eq("user_id", user.id)
            .eq("is_active", True)
            .execute()
        )
        trade_executor = TradeExecutionService(_get_supabase_admin())
        for pos in positions.data or []:
            if pos.get("execution_mode") == "live":
                await trade_executor.close_position(
                    pos,
                    pos.get("current_price") or pos.get("average_price"),
                    "kill_switch",
                )
            else:
                await close_trade_record(
                    pos.get("trade_id"),
                    CloseTrade(exit_price=pos.get("current_price"), reason="kill_switch"),
                    user.id,
                )
            positions_processed += 1

        try:
            from ..observability import EventName, track
            track(EventName.KILL_SWITCH_FIRED, user_id, {
                "success": True,
                "positions_closed": positions_processed,
                "source": "user",
            })
        except Exception:
            pass

        return {"success": True, "message": "Kill switch activated. All positions closed."}
    except Exception as e:
        try:
            from ..observability import EventName, track
            track(EventName.KILL_SWITCH_FIRED, user_id, {
                "success": False,
                "positions_closed": positions_processed,
                "error": str(e)[:300],
                "source": "user",
            })
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))
