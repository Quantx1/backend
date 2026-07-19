"""Paper-trade executor — shared by the manual /api/paper/order route
and the strategy runner's auto-execution path (PR-AM).

One function, ``execute_paper_order``, handles the full lifecycle:
  - ensures the user has a paper_portfolios row (lazy seed)
  - validates BUY against available cash
  - validates SELL against open quantity
  - inserts / updates the paper_positions row (averaging entry price
    for adds, marking 'closed' when SELL covers the full open qty)
  - updates paper_portfolios.cash (charges 10 bps roundtrip)
  - inserts a paper_trades event row with realized PnL on SELL
  - returns a ``PaperExecutionResult`` so callers can persist the order
    ID / realized PnL on their own audit rows

Use this everywhere instead of duplicating the order-placement logic.
The strategy runner depends on it to keep deploy → signal → entry →
exit honest: signals fired by the runner now actually move the user's
paper portfolio.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


# Same seed used by /api/paper/* — keep in sync if it ever changes there.
INITIAL_CASH = 10_00_000.0
# Combined STT + brokerage + GST approximation for paper realism.
ROUNDTRIP_CHARGE = 0.001


@dataclass
class PaperExecutionResult:
    """What ``execute_paper_order`` reports back to the caller."""
    ok: bool
    action: Literal["buy", "sell"]
    symbol: str
    quantity: int
    price: float
    position_id: Optional[str]
    trade_id: Optional[str]
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    cash_after: Optional[float] = None
    reason: Optional[str] = None  # set on ok=False with a short error code


def _ensure_portfolio(sb: Any, user_id: str) -> dict:
    """Lazy-seed the user's paper_portfolios row + a day-0 snapshot."""
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
    now = datetime.utcnow().isoformat() + "Z"
    new_row = {"user_id": user_id, "cash": INITIAL_CASH, "created_at": now}
    sb.table("paper_portfolios").insert(new_row).execute()
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


def execute_paper_order(
    *,
    supabase: Any,
    user_id: str,
    symbol: str,
    action: Literal["buy", "sell"],
    quantity: int,
    price: float,
    source: str = "manual",
) -> PaperExecutionResult:
    """Place a paper buy or sell at the provided price.

    ``source`` is recorded on the trade row so we can distinguish manual
    orders from strategy-driven auto-executions ('user_strategy').

    Failures return PaperExecutionResult(ok=False, reason=...). Callers
    decide whether to surface those (HTTP 4xx for the manual route)
    or silently log (strategy runner).
    """
    symbol = symbol.upper().strip()
    if action not in ("buy", "sell"):
        return PaperExecutionResult(ok=False, action=action, symbol=symbol,
                                    quantity=quantity, price=price,
                                    position_id=None, trade_id=None,
                                    reason="invalid_action")
    if quantity <= 0:
        return PaperExecutionResult(ok=False, action=action, symbol=symbol,
                                    quantity=quantity, price=price,
                                    position_id=None, trade_id=None,
                                    reason="invalid_quantity")
    if price <= 0:
        return PaperExecutionResult(ok=False, action=action, symbol=symbol,
                                    quantity=quantity, price=price,
                                    position_id=None, trade_id=None,
                                    reason="invalid_price")

    portfolio = _ensure_portfolio(supabase, user_id)
    cash = float(portfolio["cash"])
    now = datetime.utcnow().isoformat() + "Z"
    total_value = price * quantity

    existing = (
        supabase.table("paper_positions")
        .select("id, qty, entry_price")
        .eq("user_id", user_id)
        .eq("symbol", symbol)
        .eq("status", "open")
        .limit(1)
        .execute()
        .data
        or []
    )
    existing_pos = existing[0] if existing else None

    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None

    if action == "buy":
        cost = total_value * (1 + ROUNDTRIP_CHARGE / 2)
        if cost > cash:
            return PaperExecutionResult(
                ok=False, action="buy", symbol=symbol, quantity=quantity,
                price=price, position_id=None, trade_id=None,
                reason="insufficient_cash",
            )

        new_cash = cash - cost
        if existing_pos:
            old_qty = int(existing_pos["qty"])
            old_entry = float(existing_pos["entry_price"])
            new_qty = old_qty + quantity
            new_entry = (old_qty * old_entry + quantity * price) / new_qty
            supabase.table("paper_positions").update({
                "qty": new_qty,
                "entry_price": round(new_entry, 4),
            }).eq("id", existing_pos["id"]).execute()
            position_id = existing_pos["id"]
        else:
            ins = supabase.table("paper_positions").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "symbol": symbol,
                "qty": quantity,
                "entry_price": round(price, 4),
                "entry_date": now,
                "status": "open",
            }).execute()
            position_id = (ins.data or [{}])[0].get("id")

    else:  # sell
        if not existing_pos or int(existing_pos["qty"]) < quantity:
            return PaperExecutionResult(
                ok=False, action="sell", symbol=symbol, quantity=quantity,
                price=price, position_id=None, trade_id=None,
                reason="insufficient_holdings",
            )
        proceeds = total_value * (1 - ROUNDTRIP_CHARGE / 2)
        new_cash = cash + proceeds
        old_qty = int(existing_pos["qty"])
        old_entry = float(existing_pos["entry_price"])
        realized_pnl = round((price - old_entry) * quantity, 2)
        realized_pnl_pct = round(((price / old_entry) - 1) * 100, 4) if old_entry else None
        position_id = existing_pos["id"]

        if old_qty == quantity:
            supabase.table("paper_positions").update({"status": "closed"}).eq(
                "id", existing_pos["id"],
            ).execute()
        else:
            supabase.table("paper_positions").update({
                "qty": old_qty - quantity,
            }).eq("id", existing_pos["id"]).execute()

    supabase.table("paper_portfolios").update({
        "cash": round(new_cash, 2),
        "last_activity_at": now,
    }).eq("user_id", user_id).execute()

    trade_id = str(uuid.uuid4())
    trade_row = {
        "id": trade_id,
        "user_id": user_id,
        "position_id": position_id,
        "symbol": symbol,
        "action": action,
        "qty": quantity,
        "price": round(price, 4),
        "pnl": realized_pnl,
        "pnl_pct": realized_pnl_pct,
        "executed_at": now,
        "source": source,
    }
    try:
        supabase.table("paper_trades").insert(trade_row).execute()
    except Exception as exc:
        # Source column may not exist on older schemas — retry without it.
        if "source" in str(exc).lower():
            trade_row.pop("source", None)
            supabase.table("paper_trades").insert(trade_row).execute()
        else:
            raise

    return PaperExecutionResult(
        ok=True, action=action, symbol=symbol, quantity=quantity,
        price=price, position_id=position_id, trade_id=trade_id,
        realized_pnl=realized_pnl, realized_pnl_pct=realized_pnl_pct,
        cash_after=round(new_cash, 2),
    )
