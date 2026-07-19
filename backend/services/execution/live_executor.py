"""Live broker executor — strategy runner ↔ TradeExecutionService bridge (PR-AQ).

Mirrors the shape of services/paper_executor.py so the strategy runner
can call ``execute_live_order(action='buy', ...)`` exactly the way it
already calls ``execute_paper_order(...)``.

What this layer does (and does NOT do):
  - DOES: build a ``trades`` row, persist it, delegate to the existing
    ``TradeExecutionService`` which handles broker auth + place_order +
    positions row + Zerodha GTT for SL/target.
  - DOES NOT: hold any broker credentials, talk to broker APIs
    directly, manage WebSocket order status. Those live in
    data/brokers/integration.py and trading/execution.py.

Safety contract:
  1. **Broker check** — fast-fail if user has no connected broker.
  2. **Kill-switch + eligibility** — already enforced inside
     TradeExecutionService.execute(); we just propagate the result.
  3. **Idempotency** — refuse to BUY when an OPEN/PENDING trade
     already exists for (user, strategy, symbol). This stops a runner
     retry from doubling a live position.
  4. **SELL guard** — refuse if no open live position matches; never
     short by accident.

The runner calls this for every live entry/exit. Errors are logged
and surfaced as ok=False so the strategy_positions row still closes
cleanly (the strategy lifecycle stays consistent even when the
broker leg fails).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class LiveExecutionResult:
    """What ``execute_live_order`` reports back to the caller."""
    ok: bool
    action: Literal["buy", "sell"]
    symbol: str
    quantity: int
    price: float
    trade_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    position_id: Optional[str] = None
    realized_pnl: Optional[float] = None
    reason: Optional[str] = None  # set on ok=False with a short error code


def _has_broker_connection(supabase: Any, user_id: str) -> bool:
    try:
        r = (
            supabase.table("broker_connections")
            .select("id")
            .eq("user_id", user_id)
            .eq("status", "connected")
            .limit(1)
            .execute()
        )
        return bool(r.data)
    except Exception:
        return False


def _existing_open_live_trade(
    supabase: Any, user_id: str, strategy_id: str, symbol: str,
) -> Optional[dict]:
    """Idempotency check — is there already a live trade we'd be doubling?"""
    try:
        r = (
            supabase.table("trades")
            .select("id, status, broker_order_id")
            .eq("user_id", user_id)
            .eq("symbol", symbol)
            .eq("execution_mode", "live")
            .in_("status", ["pending", "approved", "open"])
            .limit(5)
            .execute()
        )
        for row in r.data or []:
            # Match strategy via the signal that fired the trade. We
            # don't have strategy_id directly on trades, but the runner
            # paths a strategy_id through the trade row's notes JSON.
            # Easier path: any open live trade on the same symbol is
            # treated as a conflict — refuse to layer a second live
            # entry on top.
            return row
    except Exception:
        pass
    return None


def _find_open_live_position(
    supabase: Any, user_id: str, symbol: str,
) -> Optional[dict]:
    """Find the live ``positions`` row to feed close_position()."""
    try:
        r = (
            supabase.table("positions")
            .select("*")
            .eq("user_id", user_id)
            .eq("symbol", symbol)
            .eq("execution_mode", "live")
            .eq("is_active", True)
            .order("last_updated", desc=True)
            .limit(1)
            .execute()
        )
        return (r.data or [None])[0]
    except Exception:
        return None


async def execute_live_order(
    *,
    supabase: Any,
    user_id: str,
    strategy_id: str,
    symbol: str,
    action: Literal["buy", "sell"],
    quantity: int,
    price: float,
    stop_loss: Optional[float] = None,
    target: Optional[float] = None,
    reason: str = "user_strategy",
) -> LiveExecutionResult:
    """Place a live broker order on behalf of a strategy fire.

    The price arg is the strategy's evaluation price — used for the
    trades.entry_price audit field. The actual broker fill price will
    differ (market order) and lands in trades.average_price after the
    broker fills.
    """
    symbol = symbol.upper().strip()
    if action not in ("buy", "sell"):
        return LiveExecutionResult(ok=False, action=action, symbol=symbol,
                                   quantity=quantity, price=price,
                                   reason="invalid_action")
    if quantity <= 0:
        return LiveExecutionResult(ok=False, action=action, symbol=symbol,
                                   quantity=quantity, price=price,
                                   reason="invalid_quantity")
    if price <= 0:
        return LiveExecutionResult(ok=False, action=action, symbol=symbol,
                                   quantity=quantity, price=price,
                                   reason="invalid_price")

    if not _has_broker_connection(supabase, user_id):
        return LiveExecutionResult(ok=False, action=action, symbol=symbol,
                                   quantity=quantity, price=price,
                                   reason="no_broker_connected")

    # Lazy import — TradeExecutionService loads instrument master which
    # is heavy; we don't want to pull it for the paper code path.
    from ...trading.execution import TradeExecutionService

    executor = TradeExecutionService(supabase)
    now = datetime.utcnow().isoformat() + "Z"

    if action == "buy":
        # ── SEBI algo-framework gate (entries only — exits below always
        # allowed so risk can be reduced even under a kill-switch). Refuses
        # new automated live exposure unless the operator is empanelled and
        # the durable pause is clear. See services/compliance_gate.py ──
        from ..compliance_gate import check_algo_order

        decision = check_algo_order(
            supabase=supabase, user_id=user_id, strategy_id=strategy_id,
            segment="equity", automated=True, live=True,
        )
        if not decision.allowed:
            logger.warning(
                "live_executor: algo order refused for strategy %s (%s)",
                strategy_id, decision.reason,
            )
            return LiveExecutionResult(
                ok=False, action="buy", symbol=symbol, quantity=quantity,
                price=price, reason=f"compliance_block:{decision.reason}",
            )

        # ── Idempotency: refuse to double an open live trade ──
        if _existing_open_live_trade(supabase, user_id, strategy_id, symbol):
            return LiveExecutionResult(
                ok=False, action="buy", symbol=symbol, quantity=quantity,
                price=price, reason="duplicate_open_trade",
            )

        trade_id = str(uuid.uuid4())
        trade_row = {
            "id": trade_id,
            "user_id": user_id,
            "symbol": symbol,
            "exchange": "NSE",
            "segment": "EQUITY",
            "direction": "LONG",
            "trade_type": "swing",
            "order_type": "MARKET",
            "product_type": "CNC",
            "execution_mode": "live",
            "quantity": quantity,
            "entry_price": round(price, 2),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "target": round(target, 2) if target else None,
            "status": "pending",
            "notes": f"strategy:{strategy_id}",
            "created_at": now,
        }
        try:
            supabase.table("trades").insert(trade_row).execute()
        except Exception as exc:
            logger.warning("live_executor: trades insert failed: %s", exc)
            return LiveExecutionResult(
                ok=False, action="buy", symbol=symbol, quantity=quantity,
                price=price, reason=f"trades_insert_failed: {exc}",
            )

        # Delegate to the existing live-trade executor. It handles:
        # broker auth, place_order, positions row, GTT for SL/target.
        result = await executor.execute({**trade_row})
        if not result.get("success"):
            # Mark trade rejected for audit; the runner's signal +
            # strategy_position rows still close cleanly.
            try:
                supabase.table("trades").update({
                    "status": "rejected",
                    "exit_reason": result.get("code") or "broker_error",
                }).eq("id", trade_id).execute()
            except Exception:
                pass
            return LiveExecutionResult(
                ok=False, action="buy", symbol=symbol, quantity=quantity,
                price=price, trade_id=trade_id,
                reason=result.get("code") or result.get("message", "broker_error"),
            )

        # Pull the broker_order_id back from the trade row (the executor
        # wrote it before returning).
        broker_order_id = None
        position_id = None
        try:
            updated = (
                supabase.table("trades")
                .select("broker_order_id")
                .eq("id", trade_id)
                .single()
                .execute()
            )
            broker_order_id = (updated.data or {}).get("broker_order_id")
            pos = (
                supabase.table("positions")
                .select("id")
                .eq("trade_id", trade_id)
                .limit(1)
                .execute()
            )
            position_id = (pos.data or [{}])[0].get("id")
        except Exception:
            pass

        return LiveExecutionResult(
            ok=True, action="buy", symbol=symbol, quantity=quantity,
            price=price, trade_id=trade_id,
            broker_order_id=broker_order_id,
            position_id=position_id,
        )

    # ── action == "sell" → close the live position ──
    position = _find_open_live_position(supabase, user_id, symbol)
    if not position:
        return LiveExecutionResult(
            ok=False, action="sell", symbol=symbol, quantity=quantity,
            price=price, reason="no_open_live_position",
        )

    result = await executor.close_position(position, exit_price=price, reason=reason)
    if not result.get("success"):
        return LiveExecutionResult(
            ok=False, action="sell", symbol=symbol, quantity=quantity,
            price=price, reason=result.get("message", "broker_error"),
            trade_id=position.get("trade_id"),
        )

    # Realized PnL — pull from the trade row the executor just updated.
    realized = None
    try:
        t = (
            supabase.table("trades")
            .select("net_pnl, broker_order_id")
            .eq("id", position.get("trade_id"))
            .single()
            .execute()
        )
        realized = float((t.data or {}).get("net_pnl") or 0)
    except Exception:
        pass

    return LiveExecutionResult(
        ok=True, action="sell", symbol=symbol, quantity=quantity,
        price=price, trade_id=position.get("trade_id"),
        position_id=position.get("id"),
        realized_pnl=realized,
    )
