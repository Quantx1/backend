"""
Paper Trading API Routes — F11 acquisition engine.

All endpoints are authed (``Depends(get_current_user)``) and persist to
the four ``paper_*`` Supabase tables:

  paper_portfolios  — one row per user (cash balance)
  paper_positions   — open / closed positions per symbol
  paper_trades      — every BUY / SELL event
  paper_snapshots   — daily equity rollups (written by scheduler)

The legacy in-memory store + unauthed ``user_id``-in-body shape was
removed in PR P0-1 (full audit, 2026-05-06): it accepted any client-
supplied user_id and lost data on restart.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core.database import get_supabase_admin
from ..core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Paper Trading"])

INITIAL_CASH = 10_00_000.0  # ₹10 Lakh virtual seed


# ============================================================================
# REQUEST MODELS
# ============================================================================


class OrderRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str = "MARKET"


# ============================================================================
# HELPERS
# ============================================================================


async def _fetch_live_price(symbol: str) -> Optional[dict]:
    """Get live price via the configured market-data provider.

    Falls back to yfinance for the same kind of data (live quote) when
    the primary provider is down — that is a *data-source* fallback,
    not an AI-prediction substitute, so it doesn't violate the
    no-fallbacks rule. Returns None when both sources fail.
    """
    try:
        from ..data.market import get_market_data_provider
        provider = get_market_data_provider()
        quote = await provider.get_quote_async(symbol)
        if quote and quote.ltp > 0:
            return {
                "symbol": symbol,
                "price": quote.ltp,
                "name": symbol,
                "change": quote.change,
                "change_percent": quote.change_percent,
            }
    except Exception as e:
        logger.warning(f"MarketDataService unavailable for {symbol}: {e}")

    try:
        import yfinance as yf
        suffix = "" if "." in symbol else ".NS"
        ticker = yf.Ticker(f"{symbol}{suffix}")
        info = ticker.fast_info
        price = float(info.get("lastPrice", 0) or info.get("last_price", 0) or 0)
        prev = float(info.get("previousClose", 0) or info.get("previous_close", 0) or price)
        if price > 0:
            change = price - prev
            change_pct = (change / prev * 100) if prev else 0.0
            return {
                "symbol": symbol,
                "price": round(price, 2),
                "name": symbol,
                "change": round(change, 2),
                "change_percent": round(change_pct, 2),
            }
    except Exception as e:
        logger.warning(f"yfinance live-quote fallback failed for {symbol}: {e}")

    return None


def _ensure_portfolio(user_id: str) -> dict:
    """Return the user's paper_portfolios row, creating it on first
    access with the ``INITIAL_CASH`` seed."""
    sb = get_supabase_admin()
    existing = (
        sb.table("paper_portfolios")
        .select("user_id, cash, created_at, last_activity_at")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if existing:
        return existing[0]
    new_row = {
        "user_id": user_id,
        "cash": INITIAL_CASH,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    sb.table("paper_portfolios").insert(new_row).execute()
    # Seed a day-0 snapshot so the equity curve has a baseline point
    # immediately rather than waiting for the nightly scheduler. The
    # scheduler will start updating from this seed forward.
    try:
        sb.table("paper_snapshots").upsert({
            "user_id": user_id,
            "snapshot_date": date.today().isoformat(),
            "equity": INITIAL_CASH,
            "cash": INITIAL_CASH,
            "invested": 0,
            "drawdown_pct": 0,
            "nifty_close": None,
        }, on_conflict="user_id,snapshot_date").execute()
    except Exception as exc:
        logger.debug("paper day-0 snapshot seed failed: %s", exc)
    return new_row


def _open_positions(user_id: str) -> List[dict]:
    sb = get_supabase_admin()
    return (
        sb.table("paper_positions")
        .select("id, symbol, qty, entry_price, entry_date, stop_loss, target")
        .eq("user_id", user_id)
        .eq("status", "open")
        .execute()
        .data
        or []
    )


def _build_portfolio(
    portfolio_row: dict,
    positions: List[dict],
    live_prices: Dict[str, float],
) -> dict:
    """Format the {cash, holdings, total_*} payload the frontend renders."""
    holdings_list = []
    total_invested = 0.0
    total_current = 0.0
    cash = float(portfolio_row.get("cash") or 0)

    for pos in positions:
        sym = pos["symbol"]
        qty = int(pos["qty"])
        avg = float(pos["entry_price"])
        live = float(live_prices.get(sym, avg))
        invested = qty * avg
        current = qty * live
        pnl = current - invested
        pnl_pct = (pnl / invested * 100) if invested else 0.0
        holdings_list.append({
            "symbol": sym,
            "quantity": qty,
            "avg_price": round(avg, 2),
            "live_price": round(live, 2),
            "invested": round(invested, 2),
            "current_value": round(current, 2),
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_pct, 2),
        })
        total_invested += invested
        total_current += current

    total_pnl = total_current - total_invested
    portfolio_value = cash + total_current
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0

    return {
        "cash_balance": round(cash, 2),
        "holdings": holdings_list,
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_percent": round(total_pnl_pct, 2),
        "portfolio_value": round(portfolio_value, 2),
    }


# ============================================================================
# ROUTES
# ============================================================================


@router.get("/api/paper/portfolio")
async def get_portfolio(user=Depends(get_current_user)):
    """Get the authed user's paper portfolio with live-price valuation."""
    portfolio = _ensure_portfolio(user.id)
    positions = _open_positions(user.id)

    # Fan out live-price fetches (sequential — paper portfolios are
    # small enough that gather isn't worth the complexity).
    live_prices: Dict[str, float] = {}
    for pos in positions:
        sym = pos["symbol"]
        if sym in live_prices:
            continue
        data = await _fetch_live_price(sym)
        if data:
            live_prices[sym] = float(data["price"])

    return _build_portfolio(portfolio, positions, live_prices)


@router.get("/api/paper/orders")
async def get_orders(
    limit: int = 100,
    user=Depends(get_current_user),
):
    """Get the authed user's recent paper-trade history (newest first)."""
    sb = get_supabase_admin()
    rows = (
        sb.table("paper_trades")
        .select(
            "id, symbol, action, qty, price, pnl, pnl_pct, exit_reason, "
            "ai_note, executed_at"
        )
        .eq("user_id", user.id)
        .order("executed_at", desc=True)
        .limit(max(1, min(500, limit)))
        .execute()
        .data
        or []
    )
    return {"orders": rows}


@router.get("/api/paper/price/{symbol}")
async def get_stock_price(symbol: str):
    """Live quote lookup. Public — no PII, just market data."""
    sym = symbol.upper().strip()
    data = await _fetch_live_price(sym)
    if not data or data["price"] <= 0:
        raise HTTPException(status_code=404, detail=f"Stock {sym} not found")
    return data


@router.post("/api/paper/order")
async def place_order(req: OrderRequest, user=Depends(get_current_user)):
    """Place a buy or sell paper trade at market price.

    Thin wrapper around services.execution.paper_executor.execute_paper_order —
    the same function the strategy runner uses for auto-executions.
    Keeping both manual and auto paths on one implementation prevents
    cash / position drift between them.
    """
    if req.action not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="Action must be BUY or SELL")
    if req.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")

    symbol = req.symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    price_data = await _fetch_live_price(symbol)
    if not price_data or price_data["price"] <= 0:
        raise HTTPException(status_code=404, detail=f"Could not fetch price for {symbol}")
    price = float(price_data["price"])

    from ..services.execution.paper_executor import execute_paper_order

    result = execute_paper_order(
        supabase=get_supabase_admin(),
        user_id=user.id,
        symbol=symbol,
        action=req.action.lower(),  # type: ignore[arg-type]
        quantity=req.quantity,
        price=price,
        source="manual",
    )

    if not result.ok:
        # Map executor error codes to standard HTTP responses.
        reason_to_status = {
            "insufficient_cash": (400, "Insufficient cash balance"),
            "insufficient_holdings": (400, f"Insufficient holdings for {symbol}"),
            "invalid_action": (400, "Action must be BUY or SELL"),
            "invalid_quantity": (400, "Quantity must be positive"),
            "invalid_price": (404, f"Could not fetch price for {symbol}"),
        }
        status_code, detail = reason_to_status.get(
            result.reason or "", (400, result.reason or "order failed"),
        )
        raise HTTPException(status_code=status_code, detail=detail)

    return {
        "success": True,
        "executed_price": round(price, 2),
        "order": {
            "id": result.trade_id,
            "symbol": symbol,
            "action": req.action,
            "quantity": req.quantity,
            "price": round(price, 2),
            "total_value": round(price * req.quantity, 2),
            "status": "EXECUTED",
            "realized_pnl": result.realized_pnl,
            "realized_pnl_pct": result.realized_pnl_pct,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    }


@router.post("/api/paper/reset")
async def reset_account(user=Depends(get_current_user)):
    """Reset the authed user's paper account to the initial seed.

    Closes all open positions (status → 'closed') and resets cash to
    ``INITIAL_CASH``. Trade history in ``paper_trades`` is preserved
    so the user can see what they've done — only positions and cash
    revert.
    """
    sb = get_supabase_admin()
    now = datetime.utcnow().isoformat() + "Z"

    sb.table("paper_positions").update({"status": "closed"}).eq(
        "user_id", user.id
    ).eq("status", "open").execute()

    _ensure_portfolio(user.id)
    sb.table("paper_portfolios").update({
        "cash": INITIAL_CASH,
        "last_activity_at": now,
    }).eq("user_id", user.id).execute()

    return {"success": True, "message": "Account reset to ₹10,00,000"}


# ============================================================================
# v2 — analytics endpoints (unchanged from prior pass; already auth-gated)
# ============================================================================


@router.get("/api/paper/v2/equity-curve")
async def paper_equity_curve(
    days: int = 90,
    user=Depends(get_current_user),
) -> dict:
    """Return the user's per-day paper equity + Nifty close benchmark
    over the last N days. Feeds the ``EquityCurve`` chart."""
    client = get_supabase_admin()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        resp = (
            client.table("paper_snapshots")
            .select("snapshot_date, equity, cash, invested, drawdown_pct, nifty_close")
            .eq("user_id", user.id)
            .gte("snapshot_date", cutoff)
            .order("snapshot_date", desc=False)
            .execute()
        )
        rows = resp.data or []
    except Exception as exc:
        logger.warning("paper equity-curve query failed: %s", exc)
        rows = []

    if rows:
        base_equity = float(rows[0]["equity"])
        base_nifty = float(rows[0]["nifty_close"] or 0) or 1
        for r in rows:
            e = float(r["equity"])
            n = float(r["nifty_close"] or 0)
            r["return_pct"] = round((e / base_equity - 1) * 100, 4) if base_equity else 0
            r["nifty_pct"] = round((n / base_nifty - 1) * 100, 4) if base_nifty else 0

    latest = rows[-1] if rows else None
    return {
        "days": days,
        "points": rows,
        "latest": latest,
        "initial_equity": INITIAL_CASH,
    }


@router.get("/api/paper/v2/league")
async def paper_league(weeks: int = 1) -> dict:
    """Anonymized weekly paper-trading leaderboard. Top 20 by weekly
    return — each user's handle is hashed to a stable, masked string."""
    client = get_supabase_admin()
    cutoff = (date.today() - timedelta(days=weeks * 7)).isoformat()

    try:
        resp = (
            client.table("paper_snapshots")
            .select("user_id, snapshot_date, equity")
            .gte("snapshot_date", cutoff)
            .execute()
        )
        rows = resp.data or []
    except Exception as exc:
        logger.warning("paper league query failed: %s", exc)
        rows = []

    by_user: Dict[str, List[dict]] = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r)
    for snaps in by_user.values():
        snaps.sort(key=lambda s: s["snapshot_date"])

    leaderboard = []
    for user_id, snaps in by_user.items():
        if len(snaps) < 2:
            continue
        start_eq = float(snaps[0]["equity"])
        end_eq = float(snaps[-1]["equity"])
        if start_eq <= 0:
            continue
        ret_pct = (end_eq / start_eq - 1) * 100
        handle = "Trader" + hashlib.sha256(user_id.encode()).hexdigest()[:6].upper()
        leaderboard.append({
            "handle": handle,
            "return_pct": round(ret_pct, 2),
            "final_equity": round(end_eq, 2),
            "snapshots": len(snaps),
        })

    leaderboard.sort(key=lambda x: x["return_pct"], reverse=True)
    top = leaderboard[:20]
    for i, row in enumerate(top, start=1):
        row["rank"] = i

    return {
        "weeks": weeks,
        "top_20": top,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/api/paper/v2/achievements")
async def paper_achievements(user=Depends(get_current_user)) -> dict:
    """Streaks + badges for the current user. Pure derivations from
    ``paper_snapshots`` + ``paper_positions``."""
    client = get_supabase_admin()

    try:
        snaps_resp = (
            client.table("paper_snapshots")
            .select("snapshot_date, equity, drawdown_pct")
            .eq("user_id", user.id)
            .order("snapshot_date", desc=False)
            .execute()
        )
        snaps = snaps_resp.data or []
    except Exception as exc:
        logger.warning("paper achievements snaps query failed: %s", exc)
        snaps = []

    streak = 0
    for i in range(len(snaps) - 1, 0, -1):
        prev = float(snaps[i - 1]["equity"])
        curr = float(snaps[i]["equity"])
        if curr > prev:
            streak += 1
        else:
            break

    try:
        trades_resp = (
            client.table("paper_positions")
            .select("symbol, qty, entry_price, status")
            .eq("user_id", user.id)
            .eq("status", "closed")
            .execute()
        )
        trades = trades_resp.data or []
    except Exception:
        trades = []

    trade_count = len(trades)
    initial = INITIAL_CASH
    current_equity = float(snaps[-1]["equity"]) if snaps else initial
    total_return_pct = ((current_equity - initial) / initial) * 100 if initial else 0
    days_trading = len(snaps)

    badges = []
    if trade_count >= 1:
        badges.append({"key": "first_trade", "label": "First trade", "tier": "bronze"})
    if trade_count >= 10:
        badges.append({"key": "ten_trades", "label": "10 trades", "tier": "silver"})
    if trade_count >= 50:
        badges.append({"key": "fifty_trades", "label": "50 trades", "tier": "gold"})
    if streak >= 3:
        badges.append({"key": "three_streak", "label": "3-day streak", "tier": "bronze"})
    if streak >= 7:
        badges.append({"key": "seven_streak", "label": "Week streak", "tier": "silver"})
    if total_return_pct >= 5:
        badges.append({"key": "five_pct", "label": "+5% gain", "tier": "bronze"})
    if total_return_pct >= 10:
        badges.append({"key": "ten_pct", "label": "+10% gain", "tier": "silver"})
    if days_trading >= 30:
        badges.append({"key": "thirty_days", "label": "30 days active", "tier": "gold"})

    return {
        "streak_days": streak,
        "trade_count": trade_count,
        "days_trading": days_trading,
        "total_return_pct": round(total_return_pct, 2),
        "current_equity": round(current_equity, 2),
        "badges": badges,
        "go_live_eligible": days_trading >= 30,
    }
