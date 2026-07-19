"""
Trade execution service for scheduler-driven fills.
Creates positions and updates trades when broker execution is not integrated.
"""

import logging
from datetime import datetime, date
from typing import Dict, Any

from ..data.brokers.integration import (
    BrokerFactory,
    Order,
    TransactionType,
    OrderType,
    ProductType,
)
from ..data.brokers.credentials import decrypt_credentials
from ..core.config import settings
from .fo.instruments import InstrumentMaster

logger = logging.getLogger(__name__)


class TradeExecutionService:
    def __init__(self, supabase_admin):
        self.supabase = supabase_admin
        self.instrument_master = InstrumentMaster(settings.FNO_INSTRUMENTS_FILE)

    async def execute(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a pending trade by opening a position and updating trade status.
        This is a DB-level execution for environments without broker integration.
        """
        # PR 48 — global kill-switch gate. Paper trades are unaffected; only
        # live (real-money) execution halts when ops flip the flag.
        if trade.get("execution_mode") == "live":
            try:
                from ..platform.system_flags import is_globally_halted, global_halt_reason
                if is_globally_halted(supabase_client=self.supabase):
                    reason = global_halt_reason(supabase_client=self.supabase) or "ops halt"
                    logger.warning(
                        "Trade %s blocked — global kill switch active (%s)",
                        trade.get("id"), reason,
                    )
                    try:
                        self.supabase.table("trades").update({
                            "status": "rejected",
                            "exit_reason": "risk_limit",
                        }).eq("id", trade.get("id")).execute()
                    except Exception:
                        pass
                    return {
                        "success": False,
                        "message": f"Trading halted: {reason}",
                        "code": "global_kill_switch",
                    }
            except Exception as kill_exc:
                logger.debug("kill-switch check skipped: %s", kill_exc)

            # SEBI algo-framework gate — the UNIVERSAL choke point for automated
            # live entries routed through this shared executor (AutoPilot
            # rebalance, strategy fires, scheduler). Refuses automated live
            # orders unless the operator is empanelled + the durable pause is
            # clear. Exits go through close_position (not here) and stay allowed.
            # Manual human orders use the separate /broker/order path, not this.
            try:
                from ..services.compliance_gate import check_algo_order

                seg = (
                    "options"
                    if str(trade.get("segment", "")).upper() in ("OPTIONS", "FNO", "OPTION")
                    else "equity"
                )
                automated = str(trade.get("order_source", "algo")).lower() != "manual"
                decision = check_algo_order(
                    supabase=self.supabase,
                    user_id=str(trade.get("user_id") or ""),
                    strategy_id=str(trade.get("strategy_id") or trade.get("notes") or ""),
                    segment=seg,
                    automated=automated,
                    live=True,
                )
                if not decision.allowed:
                    logger.warning(
                        "Trade %s blocked by compliance gate (%s)",
                        trade.get("id"), decision.reason,
                    )
                    try:
                        self.supabase.table("trades").update({
                            "status": "rejected",
                            "exit_reason": "compliance_block",
                        }).eq("id", trade.get("id")).execute()
                    except Exception:
                        pass
                    return {
                        "success": False,
                        "message": f"Order blocked (compliance): {decision.reason}",
                        "code": f"compliance_block:{decision.reason}",
                    }
            except Exception as gate_exc:
                logger.debug("compliance gate check skipped: %s", gate_exc)

        try:
            if trade.get("execution_mode") == "live":
                return await self._execute_live_trade(trade)

            trade_id = trade.get("id")
            user_id = trade.get("user_id")

            if not trade_id or not user_id:
                return {"success": False, "message": "Missing trade or user id"}

            if trade.get("status") not in ["pending", "approved"]:
                return {"success": False, "message": "Trade is not pending/approved"}

            existing = self.supabase.table("positions").select("id").eq("trade_id", trade_id).execute()
            if existing.data:
                return {"success": True, "message": "Position already open"}

            entry_price = float(trade.get("average_price") or trade.get("entry_price") or 0)
            if entry_price <= 0:
                return {"success": False, "message": "Invalid entry price"}

            quantity = int(trade.get("quantity") or 0)
            if quantity <= 0:
                return {"success": False, "message": "Invalid quantity"}

            position = {
                "user_id": user_id,
                "trade_id": trade_id,
                "symbol": trade.get("symbol"),
                "exchange": trade.get("exchange", "NSE"),
                "segment": trade.get("segment", "EQUITY"),
                "expiry_date": trade.get("expiry_date"),
                "strike_price": trade.get("strike_price"),
                "option_type": trade.get("option_type"),
                "direction": trade.get("direction"),
                "quantity": quantity,
                "lots": trade.get("lots", 1),
                "average_price": entry_price,
                "current_price": entry_price,
                "current_value": quantity * entry_price,
                "stop_loss": trade.get("stop_loss"),
                "target": trade.get("target"),
                "margin_used": trade.get("margin_used"),
                "risk_amount": trade.get("risk_amount"),
                "execution_mode": trade.get("execution_mode", "paper"),
                "is_active": True,
                "last_updated": datetime.utcnow().isoformat(),
            }

            self.supabase.table("positions").insert(position).execute()

            self.supabase.table("trades").update({
                "status": "open",
                "executed_at": datetime.utcnow().isoformat(),
                "average_price": entry_price,
                "filled_quantity": quantity,
                "pending_quantity": 0,
            }).eq("id", trade_id).execute()

            return {"success": True, "message": "Trade executed"}
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"success": False, "message": str(e)}

    async def _execute_live_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Execute live trade via broker API."""
        trade_id = trade.get("id")
        user_id = trade.get("user_id")
        if not trade_id or not user_id:
            return {"success": False, "message": "Missing trade or user id"}

        # PR 130 — defense-in-depth eligibility check. Route layer already
        # gates by tier on its way in; this catches the AutoPilot / RL
        # paths that may not pass through the same routes.
        try:
            from .eligibility import check_live_trade_eligibility  # noqa: PLC0415
            elig = check_live_trade_eligibility(
                user_id=str(user_id),
                supabase=self.supabase,
            )
            if not elig.eligible:
                logger.warning(
                    "Live execution blocked for trade %s: %s (%s)",
                    trade_id, elig.code, elig.reason,
                )
                try:
                    self.supabase.table("trades").update({
                        "status": "rejected",
                        "exit_reason": elig.code or "ineligible",
                    }).eq("id", trade_id).execute()
                except Exception:
                    pass
                return {
                    "success": False,
                    "message": elig.reason or "Live execution not allowed",
                    "code": elig.code,
                }
        except Exception as exc:
            logger.debug("eligibility check skipped: %s", exc)

        conn = self.supabase.table("broker_connections").select(
            "broker_name, access_token"
        ).eq("user_id", user_id).eq("status", "connected").single().execute()

        if not conn.data:
            return {"success": False, "message": "No broker connected"}

        broker_name = conn.data["broker_name"]
        credentials = decrypt_credentials(conn.data["access_token"])

        broker = BrokerFactory.create(broker_name, credentials)
        if not broker.login():
            return {"success": False, "message": "Broker login failed"}

        direction = trade.get("direction")
        symbol = trade.get("symbol")
        exchange = trade.get("exchange", "NSE")
        instrument_token = None
        if trade.get("segment") == "FUTURES":
            resolved = self._resolve_futures_contract(trade)
            if resolved:
                symbol = resolved.get("tradingsymbol", symbol)
                exchange = resolved.get("exchange", exchange) or exchange
                instrument_token = resolved.get("instrument_token")
        qty = int(trade.get("quantity") or 0)
        if qty <= 0:
            return {"success": False, "message": "Invalid quantity"}

        order = Order(
            symbol=symbol,
            exchange=exchange,
            transaction_type=TransactionType.BUY if direction == "LONG" else TransactionType.SELL,
            quantity=qty,
            product=ProductType.CNC if trade.get("segment") == "EQUITY" else ProductType.NRML,
            order_type=OrderType.MARKET,
            price=0,
            instrument_token=instrument_token,
        )

        placed = broker.place_order(order)
        if placed.status.name == "REJECTED":
            return {"success": False, "message": f"Order rejected: {placed.message}"}

        # Update trade
        self.supabase.table("trades").update({
            "status": "open",
            "executed_at": datetime.utcnow().isoformat(),
            "average_price": trade.get("entry_price"),
            "filled_quantity": qty,
            "pending_quantity": 0,
            "broker_order_id": placed.order_id,
        }).eq("id", trade_id).execute()

        # Create position
        position = {
            "user_id": user_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "exchange": exchange,
            "segment": trade.get("segment", "EQUITY"),
            "expiry_date": trade.get("expiry_date"),
            "strike_price": trade.get("strike_price"),
            "option_type": trade.get("option_type"),
            "direction": direction,
            "quantity": qty,
            "lots": trade.get("lots", 1),
            "average_price": trade.get("entry_price"),
            "current_price": trade.get("entry_price"),
            "stop_loss": trade.get("stop_loss"),
            "target": trade.get("target"),
            "margin_used": trade.get("margin_used"),
            "risk_amount": trade.get("risk_amount"),
            "execution_mode": "live",
            "is_active": True,
            "last_updated": datetime.utcnow().isoformat(),
        }
        self.supabase.table("positions").insert(position).execute()

        # T1.1 — UNIVERSAL broker-side stop placement.
        # Previously only attempted for Zerodha; Upstox + Angel positions
        # had NO broker-placed protection. Now routed through the
        # stop_orchestrator which dispatches to native GTT (Zerodha) or
        # SL-M (Upstox/Angel). Result is persisted so close_position()
        # can cancel cleanly and so an unprotected state is alertable.
        from ..services.execution.stop_orchestrator import place_stop_orders
        stop_result = place_stop_orders(
            broker, broker_name,
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            quantity=qty,
            stop_loss=trade.get("stop_loss"),
            target=trade.get("target"),
        )
        # Persist on the position row so close_position can find the IDs.
        # Patch payload is a dict of new column values that match
        # the schema added in migration pr_t1_1_broker_stops.sql.
        try:
            self.supabase.table("positions").update(
                stop_result.to_position_patch()
            ).eq("trade_id", trade_id).execute()
        except Exception as e:
            logger.warning("stop_result persist failed for trade %s: %s", trade_id, e)
        # Mirror on trades.entry_gtt_id for backwards compat with older code.
        if stop_result.stop_broker_id:
            try:
                self.supabase.table("trades").update({
                    "entry_gtt_id": stop_result.stop_broker_id,
                }).eq("id", trade_id).execute()
            except Exception:
                pass
        if stop_result.status != "placed":
            logger.warning(
                "Position %s for user %s OPENED WITHOUT broker stop (status=%s, err=%s) — alerting user",
                symbol, user_id, stop_result.status, stop_result.error,
            )
            # Fire the fno_position_unprotected alert event so the user
            # learns IMMEDIATELY (push + telegram + whatsapp + email per
            # alerts_routes.py DEFAULT_PREFS — this is the highest-urgency
            # alert in the system).
            try:
                from ..platform.alerts import channels_for_event_sync
                channels = channels_for_event_sync(user_id, "fno_position_unprotected")
                title = f"⚠ Position UNPROTECTED at broker: {symbol}"
                message = (
                    f"Your {direction} {qty} {symbol} position was opened but the broker "
                    f"did NOT accept the stop-loss ({stop_result.status}: "
                    f"{stop_result.error or 'unknown'}). "
                    f"Stop your position manually or rely on our monitoring service."
                )
                # Push dispatch — we're already inside an async function, await directly.
                if "push" in channels:
                    try:
                        from ..platform.notifications import dispatch_push
                        await dispatch_push(
                            user_id=user_id, title=title, body=message,
                            url=f"/portfolio?position={trade_id}",
                        )
                    except Exception as pe:
                        logger.debug("unprotected-position push failed: %s", pe)
                # Insert into notifications table — SCHEMA: type/message/data
                # (not event/body/metadata — verified via information_schema 2026-05-31).
                try:
                    self.supabase.table("notifications").insert({
                        "user_id": user_id,
                        "type": "fno_position_unprotected",
                        "priority": "critical",
                        "title": title,
                        "message": message,
                        "channels": list(channels),
                        "data": {
                            "symbol": symbol,
                            "trade_id": trade_id,
                            "stop_status": stop_result.status,
                            "stop_error": stop_result.error,
                            "url": f"/portfolio?position={trade_id}",
                        },
                    }).execute()
                except Exception as ne:
                    logger.debug("notifications insert failed: %s", ne)
            except Exception as e:
                logger.warning("unprotected-position alert dispatch failed: %s", e)

        return {
            "success": True,
            "message": "Trade executed (live)",
            "stop_status": stop_result.status,
        }

    async def close_position(self, position: Dict[str, Any], exit_price: float, reason: str) -> Dict[str, Any]:
        """Close live position via broker, then update DB."""
        try:
            user_id = position.get("user_id")
            trade_id = position.get("trade_id")
            symbol = position.get("symbol")
            exchange = position.get("exchange", "NSE")
            instrument_token = None
            qty = int(position.get("quantity") or 0)
            direction = position.get("direction")

            if position.get("execution_mode") != "live":
                return {"success": False, "message": "Not a live position"}

            conn = self.supabase.table("broker_connections").select(
                "broker_name, access_token"
            ).eq("user_id", user_id).eq("status", "connected").single().execute()
            if not conn.data:
                return {"success": False, "message": "No broker connected"}

            broker_name = conn.data["broker_name"]
            credentials = decrypt_credentials(conn.data["access_token"])
            broker = BrokerFactory.create(broker_name, credentials)
            if not broker.login():
                return {"success": False, "message": "Broker login failed"}

            if position.get("segment") == "FUTURES":
                resolved = self._resolve_futures_contract(position)
                if resolved:
                    symbol = resolved.get("tradingsymbol", symbol)
                    exchange = resolved.get("exchange", exchange) or exchange
                    instrument_token = resolved.get("instrument_token")

            # T1.1 — cancel any broker-side stops attached to this position
            # BEFORE placing the market exit, otherwise both will fire and
            # we end up double-selling. Errors are logged but non-fatal —
            # the user has already initiated the exit.
            stop_broker_id = position.get("stop_broker_id")
            target_broker_id = position.get("target_broker_id")
            if stop_broker_id or target_broker_id:
                try:
                    from ..services.execution.stop_orchestrator import cancel_stop_orders
                    cancel_stop_orders(
                        broker, broker_name,
                        stop_broker_id=stop_broker_id,
                        target_broker_id=target_broker_id,
                    )
                except Exception as e:
                    logger.warning(
                        "cancel_stop_orders failed for position %s: %s — proceeding with market exit",
                        position.get("id"), e,
                    )

            order = Order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=TransactionType.SELL if direction == "LONG" else TransactionType.BUY,
                quantity=qty,
                product=ProductType.CNC if position.get("segment") == "EQUITY" else ProductType.NRML,
                order_type=OrderType.MARKET,
                price=0,
                instrument_token=instrument_token,
            )
            placed = broker.place_order(order)
            if placed.status.name == "REJECTED":
                return {"success": False, "message": f"Exit order rejected: {placed.message}"}

            pnl = (exit_price - position.get("average_price")) * \
                qty if direction == "LONG" else (position.get("average_price") - exit_price) * qty
            pnl_pct = (pnl / (qty * position.get("average_price"))) * 100 if qty else 0

            self.supabase.table("trades").update({
                "status": "closed",
                "exit_price": exit_price,
                "net_pnl": pnl,
                "pnl_percent": pnl_pct,
                "exit_reason": reason,
                "closed_at": datetime.utcnow().isoformat()
            }).eq("id", trade_id).execute()

            self.supabase.table("positions").update({
                "is_active": False
            }).eq("id", position["id"]).execute()

            return {"success": True, "message": "Position closed"}
        except Exception as e:
            logger.error(f"Close position failed: {e}")
            return {"success": False, "message": str(e)}

    def _resolve_futures_contract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve futures tradingsymbol/exchange/instrument_token from instrument master.
        """
        if not self.instrument_master.available():
            return {}
        underlying = data.get("symbol")
        if not underlying:
            return {}
        expiry_raw = data.get("expiry_date")
        expiry_date = None
        if isinstance(expiry_raw, date):
            expiry_date = expiry_raw
        elif isinstance(expiry_raw, str):
            try:
                expiry_date = datetime.fromisoformat(expiry_raw.replace("Z", "+00:00")).date()
            except Exception:
                expiry_date = None
        return self.instrument_master.get_futures_contract(underlying, on_date=expiry_date or date.today()) or {}
