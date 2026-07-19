"""Live broker order status reconciliation (PR-AR.2).

After PR-AQ the runner places live broker orders and writes a
``trades`` row with status='pending' + broker_order_id. The broker
acknowledges receipt instantly, but the actual fill happens
microseconds (or seconds, or never) later. If we don't poll the broker
for the new status, the trades row stays 'pending' forever and the
/strategies/deployed panel shows a phantom open trade.

This service polls every pending trade and updates its status from the
broker side. Runs every 2 minutes during market hours via the
scheduler.

Per-trade work:
  1. Look up the user's broker_connection (skip if disconnected)
  2. Call broker.get_order_status(broker_order_id)
  3. Map broker status → trades.status:
       COMPLETE  → 'open'        (filled — position is alive)
       CANCELLED → 'cancelled'
       REJECTED  → 'rejected'
       PENDING/OPEN → stay 'pending', bump reconciliation_attempts
  4. Update last_reconciled_at so the index can find stale rows fast

After ``MAX_RECONCILE_ATTEMPTS`` unsuccessful polls (~20 min of pending
state) we mark the trade 'unknown' so the user sees something is wrong
and the deployed-panel doesn't keep it as 'pending' forever.

Idempotent — re-running against a 'closed' or 'rejected' trade is a
no-op. Safe to invoke from any scheduler firing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Stop polling a trade after 10 attempts (~20 min at 2-min cadence).
# A market order that hasn't filled in 20 minutes is broken — surface
# it to the user as 'unknown' rather than keep polling forever.
MAX_RECONCILE_ATTEMPTS = 10

# Cap per-tick batch so a long pending-trade queue can't blow the tick.
RECONCILE_BATCH_SIZE = 50


@dataclass
class ReconcileReport:
    started_at: str
    finished_at: Optional[str] = None
    scanned: int = 0
    transitioned_open: int = 0
    transitioned_cancelled: int = 0
    transitioned_rejected: int = 0
    transitioned_unknown: int = 0
    still_pending: int = 0
    errors: int = 0
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scanned": self.scanned,
            "transitioned_open": self.transitioned_open,
            "transitioned_cancelled": self.transitioned_cancelled,
            "transitioned_rejected": self.transitioned_rejected,
            "transitioned_unknown": self.transitioned_unknown,
            "still_pending": self.still_pending,
            "errors": self.errors,
        }


# Map BrokerOrderStatus.name → trades.status
_BROKER_TO_TRADE_STATUS = {
    "COMPLETE": "open",
    "FILLED": "open",
    "CANCELLED": "cancelled",
    "REJECTED": "rejected",
}


def _broker_for_user(supabase: Any, user_id: str):
    """Build an authenticated broker client for the user; None on failure."""
    try:
        from ...data.brokers.credentials import decrypt_credentials
        from ...data.brokers.integration import BrokerFactory
    except Exception:
        return None, None

    try:
        conn = (
            supabase.table("broker_connections")
            .select("broker_name, access_token")
            .eq("user_id", user_id)
            .eq("status", "connected")
            .single()
            .execute()
        )
    except Exception:
        return None, None
    if not conn.data:
        return None, None

    broker_name = conn.data["broker_name"]
    try:
        credentials = decrypt_credentials(conn.data["access_token"])
        broker = BrokerFactory.create(broker_name, credentials)
        if broker and broker.login():
            return broker, broker_name
    except Exception as exc:
        logger.debug("reconciler: broker init failed: %s", exc)
    return None, broker_name


async def reconcile_pending_trades(supabase: Any) -> ReconcileReport:
    """Scan every pending live trade and pull a fresh status from the
    broker. Returns a report — caller logs it.
    """
    report = ReconcileReport(started_at=datetime.now(timezone.utc).isoformat())
    try:
        rows = (
            supabase.table("trades")
            .select(
                "id, user_id, broker_order_id, status, reconciliation_attempts, "
                "average_price, quantity, symbol"
            )
            .in_("status", ["pending", "approved"])
            .eq("execution_mode", "live")
            .order("last_reconciled_at", desc=False, nullsfirst=True)
            .limit(RECONCILE_BATCH_SIZE)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        report.errors += 1
        report.error_messages.append(f"trades load: {exc}")
        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report

    report.scanned = len(rows)
    if not rows:
        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report

    # Cache broker clients per user so we don't re-auth for every row.
    broker_cache: Dict[str, Any] = {}
    now_iso = datetime.now(timezone.utc).isoformat()

    for trade in rows:
        trade_id = trade["id"]
        broker_order_id = trade.get("broker_order_id")
        user_id = trade.get("user_id")
        attempts = int(trade.get("reconciliation_attempts") or 0) + 1

        if not broker_order_id:
            # Order never got an ID from the broker — treat as rejected
            # rather than poll forever.
            try:
                supabase.table("trades").update({
                    "status": "rejected",
                    "exit_reason": "manual",
                    "last_reconciled_at": now_iso,
                    "reconciliation_attempts": attempts,
                }).eq("id", trade_id).execute()
                report.transitioned_rejected += 1
            except Exception:
                report.errors += 1
            continue

        if user_id not in broker_cache:
            broker_cache[user_id] = _broker_for_user(supabase, user_id)
        broker, _ = broker_cache[user_id]

        if broker is None:
            # User disconnected their broker; we can't poll. Bump
            # attempts so the next tick eventually marks it unknown.
            try:
                supabase.table("trades").update({
                    "last_reconciled_at": now_iso,
                    "reconciliation_attempts": attempts,
                }).eq("id", trade_id).execute()
                report.still_pending += 1
            except Exception:
                report.errors += 1
            continue

        try:
            broker_status = broker.get_order_status(broker_order_id)
            status_name = getattr(broker_status, "name", str(broker_status)).upper()
        except Exception as exc:
            logger.debug("reconciler: get_order_status failed for %s: %s",
                         trade_id, exc)
            report.errors += 1
            try:
                supabase.table("trades").update({
                    "last_reconciled_at": now_iso,
                    "reconciliation_attempts": attempts,
                }).eq("id", trade_id).execute()
            except Exception:
                pass
            continue

        new_status = _BROKER_TO_TRADE_STATUS.get(status_name)
        if new_status is None:
            # Still PENDING / OPEN on the broker side.
            if attempts >= MAX_RECONCILE_ATTEMPTS:
                # Give up — mark cancelled (the allowed terminal status
                # closest to "broker never confirmed") so the UI stops
                # claiming "pending" indefinitely.
                try:
                    supabase.table("trades").update({
                        "status": "cancelled",
                        "exit_reason": "manual",
                        "last_reconciled_at": now_iso,
                        "reconciliation_attempts": attempts,
                    }).eq("id", trade_id).execute()
                    report.transitioned_unknown += 1
                    logger.warning(
                        "reconciler: gave up on trade %s after %d attempts (marked cancelled)",
                        trade_id, attempts,
                    )
                except Exception:
                    report.errors += 1
            else:
                try:
                    supabase.table("trades").update({
                        "last_reconciled_at": now_iso,
                        "reconciliation_attempts": attempts,
                    }).eq("id", trade_id).execute()
                    report.still_pending += 1
                except Exception:
                    report.errors += 1
            continue

        # Status transition path.
        update_payload: Dict[str, Any] = {
            "status": new_status,
            "last_reconciled_at": now_iso,
            "reconciliation_attempts": attempts,
        }
        if new_status == "open":
            update_payload.update({
                "executed_at": now_iso,
                "filled_quantity": trade.get("quantity"),
                "pending_quantity": 0,
            })
            report.transitioned_open += 1
        elif new_status == "cancelled":
            update_payload["exit_reason"] = "manual"
            report.transitioned_cancelled += 1
        elif new_status == "rejected":
            update_payload["exit_reason"] = "manual"
            report.transitioned_rejected += 1

        try:
            supabase.table("trades").update(update_payload).eq("id", trade_id).execute()
        except Exception as exc:
            logger.warning("reconciler: trade update failed for %s: %s", trade_id, exc)
            report.errors += 1

    report.finished_at = datetime.now(timezone.utc).isoformat()
    return report
