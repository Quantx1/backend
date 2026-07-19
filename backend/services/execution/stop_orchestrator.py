"""Universal stop-loss placement at the broker.

T1.1 (2026-05-31) — closes the catastrophic gap where the live executor
silently failed to place broker-side stops for Upstox + Angel and only
DB-stored them for Zerodha when GTT failed.

Single entry point `place_stop_orders()` works for all 3 supported
brokers (Zerodha · Upstox · Angel) and returns a uniform structured
result so the caller can persist it on the position row.

Each broker has its own native mechanism:
  - Zerodha: native GTT (OCO two-leg: SL + Target)
  - Upstox:  SL-M order (broker-placed stop-loss-market)
  - Angel:   STOPLOSS_MARKET via SmartAPI

When broker-side placement succeeds, the position is protected even
through a Quant X server crash — the broker holds the stop and fires
it on price trigger regardless of our uptime. This is the core safety
guarantee for live trading.

Counterpart `cancel_stop_orders()` MUST be called from close_position()
to avoid orphaned stops at the broker after a manual exit.

Per memory `project_no_fallbacks_no_refunds_2026_04_19`: when stop
placement fails entirely (broker rejection, instrument unsupported), we
mark the position `stop_status='unprotected'` and surface it to the
user via the next alert — never silently continue as if stopped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StopResult:
    """Outcome of broker-side stop placement.

    `status` values:
      placed         — broker accepted the stop (GTT or SL-M live)
      unsupported    — broker doesn't support stops for this instrument
      failed         — broker rejected (insufficient funds, halt, etc.)
      unprotected    — placement attempt errored; user MUST be alerted
    """
    status: str = "unprotected"
    stop_broker_id: Optional[str] = None
    target_broker_id: Optional[str] = None
    error: Optional[str] = None
    method: Optional[str] = None       # 'gtt_oco' | 'sl_m' | 'gtt_single'

    def to_position_patch(self) -> Dict[str, Any]:
        """DB patch for the positions table — keep field names stable."""
        return {
            "stop_status": self.status,
            "stop_broker_id": self.stop_broker_id,
            "target_broker_id": self.target_broker_id,
            "stop_method": self.method,
            "stop_error": self.error,
        }


def place_stop_orders(
    broker: Any,
    broker_name: str,
    *,
    symbol: str,
    exchange: str,
    direction: str,                # 'LONG' | 'SHORT'
    quantity: int,
    stop_loss: Optional[float],
    target: Optional[float] = None,
) -> StopResult:
    """Place broker-side stops for an opened position.

    Returns a StopResult with status + the broker order/GTT IDs so the
    caller can persist them for later cancellation on close.
    """
    if not stop_loss or quantity <= 0:
        return StopResult(status="failed", error="missing_stop_or_qty")

    # Lazy import to avoid pulling broker SDKs into modules that don't need them
    try:
        from ...data.brokers.integration import GTTOrder
    except Exception as e:
        return StopResult(status="unprotected", error=f"integration_import_failed: {e}")

    exit_side = "SELL" if direction == "LONG" else "BUY"
    trigger_values = [stop_loss]
    orders = [{
        "transaction_type": exit_side,
        "quantity": quantity,
        "price": stop_loss,
    }]
    trigger_type = "single"

    # Two-leg OCO (SL + Target) is supported native by Zerodha. Upstox
    # and Angel only place the SL leg via SL-M; target handled by the
    # position monitor cron.
    if target and broker_name.lower() == "zerodha":
        trigger_values = [stop_loss, target]
        orders.append({
            "transaction_type": exit_side,
            "quantity": quantity,
            "price": target,
        })
        trigger_type = "two-leg"

    gtt = GTTOrder(
        symbol=symbol,
        exchange=exchange,
        trigger_type=trigger_type,
        trigger_values=trigger_values,
        orders=orders,
    )

    try:
        result = broker.place_gtt_order(gtt)
    except Exception as e:
        logger.error("stop_orchestrator: %s.place_gtt_order(%s) raised: %s",
                     broker_name, symbol, e)
        return StopResult(status="unprotected", error=str(e)[:200])

    # Parse the broker-specific status fields
    gtt_id = getattr(result, "gtt_id", None)
    status_raw = (getattr(result, "status", "") or "").lower()

    # Map broker status to our canonical states
    if status_raw in ("placed", "active", "sl_placed"):
        method = "gtt_oco" if trigger_type == "two-leg" else (
            "gtt_single" if broker_name.lower() == "zerodha" else "sl_m"
        )
        return StopResult(
            status="placed",
            stop_broker_id=gtt_id,
            target_broker_id=None,   # target lives inside the OCO group for Zerodha
            method=method,
        )
    if status_raw in ("failed", "sl_failed"):
        return StopResult(status="failed", error=status_raw, method=None)
    if status_raw == "skipped":
        # Zerodha enctoken-mode case — feature unsupported
        return StopResult(status="unsupported", error="broker_token_mode")
    # Unknown status — treat as unprotected (will be alerted)
    return StopResult(status="unprotected", error=f"unknown_status:{status_raw}")


def cancel_stop_orders(
    broker: Any,
    broker_name: str,
    *,
    stop_broker_id: Optional[str],
    target_broker_id: Optional[str] = None,
) -> Dict[str, bool]:
    """Cancel any broker-side stops attached to a position being closed.

    Returns {'stop': bool, 'target': bool} indicating each cancellation
    outcome. We never raise — orphaned cancellations are logged but not
    fatal, since the user has already initiated the exit.
    """
    out = {"stop": True, "target": True}
    if stop_broker_id:
        try:
            ok = _cancel_one(broker, broker_name, stop_broker_id)
            out["stop"] = ok
        except Exception as e:
            logger.warning("stop_orchestrator: cancel stop %s on %s failed: %s",
                           stop_broker_id, broker_name, e)
            out["stop"] = False
    if target_broker_id:
        try:
            out["target"] = _cancel_one(broker, broker_name, target_broker_id)
        except Exception as e:
            logger.warning("stop_orchestrator: cancel target %s on %s failed: %s",
                           target_broker_id, broker_name, e)
            out["target"] = False
    return out


def _cancel_one(broker: Any, broker_name: str, broker_id: str) -> bool:
    """Dispatch broker-specific cancel call."""
    bname = broker_name.lower()
    if bname == "zerodha":
        # Kite differentiates GTT from regular order cancellation.
        if hasattr(broker, "cancel_gtt_order"):
            return bool(broker.cancel_gtt_order(broker_id))
        # Older clients fall back to regular cancel
        return bool(broker.cancel_order(broker_id))
    # Upstox + Angel use regular order cancellation for their SL-M orders
    return bool(broker.cancel_order(broker_id))
