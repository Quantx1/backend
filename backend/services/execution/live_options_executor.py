"""Live broker multi-leg options executor (PR-AV).

Bridge from the runner's options dispatch to the existing broker
``place_order`` primitives. Mirrors the shape of paper_options_executor
so the runner can swap call sites by status:

  paper:  open_paper_option_position(...)
  live:   open_live_option_position(...)

What this layer enforces:
  1. **Broker check** — fail fast when no connected broker.
  2. **Margin check** — broker.get_available_margin() must cover
     estimated_margin from the strategy. (BS-derived estimate; we don't
     hit the broker's margin calculator — that's per-broker and slow.)
  3. **Sequential leg placement** — each leg goes as a MARKET order.
     If a later leg fails we DO NOT auto-reverse the earlier ones —
     marking the position as 'partial' is safer than a panicked
     reverse order at unknown prices. The user resolves via the F&O
     panel's manual close.
  4. **Audit** — every leg gets a ``trades`` row with execution_mode
     ='live', segment='OPTIONS'; combined position lives in
     paper_option_positions (with metadata.live_broker_order_ids[]).

Tradingsymbol format:
  Zerodha format (NIFTY24N0024000CE) is used as the canonical here.
  We don't try every broker's quirks in v1 — Upstox/Angel routing
  for options is flagged as 'live_options_unsupported' until a
  per-broker symbol formatter ships.

NOT in this PR (next slice):
  - GTT placement of SL/target per leg (Zerodha supports two-leg GTT
    on single options; multi-leg requires per-leg GTTs which is
    fragile when legs share underlying risk).
  - OCO bracket orchestration.
  - Reverse-on-partial rollback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from ...ai.strategy.dsl import LegSpec, OptionType
from ...ai.strategy.options_resolver import resolve_legs

logger = logging.getLogger(__name__)


# Index → exchange. Index option contracts trade on NFO.
_INDEX_TO_EXCHANGE = {
    "NIFTY": "NFO",
    "BANKNIFTY": "NFO",
    "FINNIFTY": "NFO",
    "MIDCPNIFTY": "NFO",
    "SENSEX": "BFO",
}


@dataclass
class LiveOpenResult:
    ok: bool
    position_id: Optional[str] = None
    placed_legs: List[Dict[str, Any]] = field(default_factory=list)
    failed_legs: List[Dict[str, Any]] = field(default_factory=list)
    estimated_margin: Optional[float] = None
    available_margin: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class LiveCloseResult:
    ok: bool
    position_id: Optional[str] = None
    placed_legs: List[Dict[str, Any]] = field(default_factory=list)
    failed_legs: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None


_MONTH_3LETTER = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def _format_zerodha_options_symbol(
    underlying: str, expiry: date, strike: float, option_type: OptionType,
) -> str:
    """Zerodha weekly options tradingsymbol: NIFTY26N2724000PE.

    Format: {SYMBOL}{YY}{MONTH_CODE}{DD}{STRIKE}{CE|PE}
    Month code: 1-9 for Jan-Sep (single digit), O/N/D for Oct/Nov/Dec.
    Strike is integer (NIFTY/BANKNIFTY interval = 50/100).
    """
    underlying = underlying.upper().strip()
    y = expiry.strftime("%y")
    month_letter = {
        1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
        7: "7", 8: "8", 9: "9", 10: "O", 11: "N", 12: "D",
    }[expiry.month]
    d = expiry.strftime("%d")
    strike_str = str(int(round(strike)))
    return f"{underlying}{y}{month_letter}{d}{strike_str}{option_type.value}"


def _format_angel_options_symbol(
    underlying: str, expiry: date, strike: float, option_type: OptionType,
) -> str:
    """Angel One tradingsymbol: NIFTY27NOV2624000CE.

    Format: {SYMBOL}{DD}{MMM}{YY}{STRIKE}{CE|PE}
    Month is 3-letter (JAN/FEB/.../DEC). Day is 2-digit zero-padded.
    """
    underlying = underlying.upper().strip()
    d = expiry.strftime("%d")
    mmm = _MONTH_3LETTER[expiry.month]
    yy = expiry.strftime("%y")
    strike_str = str(int(round(strike)))
    return f"{underlying}{d}{mmm}{yy}{strike_str}{option_type.value}"


def _format_upstox_options_symbol(
    underlying: str, expiry: date, strike: float, option_type: OptionType,
) -> str:
    """Upstox tradingsymbol (NFO segment): NIFTY 27 NOV 24 CE 24000.

    Note: Upstox internally resolves via instrument_key. For the
    legacy place_order path (symbol+exchange), the broker accepts the
    NSE-style "{SYMBOL} {DD} {MMM} {YY} {CE|PE} {STRIKE}" format with
    spaces. We omit spaces here (broker normalises) → NIFTY27NOV24CE24000.
    """
    underlying = underlying.upper().strip()
    d = expiry.strftime("%d")
    mmm = _MONTH_3LETTER[expiry.month]
    yy = expiry.strftime("%y")
    strike_str = str(int(round(strike)))
    return f"{underlying}{d}{mmm}{yy}{option_type.value}{strike_str}"


# Dispatcher: broker_name → formatter. Kept here so the executor only
# imports one symbol when broker support grows.
_FORMATTERS = {
    "zerodha": _format_zerodha_options_symbol,
    "angelone": _format_angel_options_symbol,
    "angel": _format_angel_options_symbol,
    "upstox": _format_upstox_options_symbol,
}


def _format_options_symbol(
    broker_name: str, underlying: str, expiry: date,
    strike: float, option_type: OptionType,
) -> Optional[str]:
    """Returns the broker's tradingsymbol for the option, or None if
    the broker isn't supported yet."""
    fmt = _FORMATTERS.get(str(broker_name or "").lower())
    if fmt is None:
        return None
    return fmt(underlying, expiry, strike, option_type)


def _broker_for_user(supabase: Any, user_id: str):
    """Authenticated broker client + broker_name; (None, None) on failure."""
    try:
        from ...data.brokers.credentials import decrypt_credentials
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
        creds = decrypt_credentials(conn.data["access_token"])
        broker = __import__(
            "backend.data.brokers.integration", fromlist=["BrokerFactory"],
        ).BrokerFactory.create(broker_name, creds)
        if broker and broker.login():
            return broker, broker_name
    except Exception as exc:
        logger.debug("live_options_executor: broker init failed: %s", exc)
    return None, broker_name


async def open_live_option_position(
    *,
    supabase: Any,
    user_id: str,
    underlying: str,
    spot: float,
    sigma: float,
    legs: List[LegSpec],
    lots: int = 1,
    strategy_id: Optional[str] = None,
    template_slug: Optional[str] = None,
    today: Optional[date] = None,
    estimated_margin: Optional[float] = None,
) -> LiveOpenResult:
    """Open a multi-leg live position by placing each leg sequentially
    at the user's connected broker.

    Even on the live path we ALSO persist a ``paper_option_positions``
    row to track combined P&L + entry premium across legs, because that
    table is what the /strategies/deployed + /fo-strategies surfaces
    read. The position metadata records ``live=true`` and
    ``broker_order_ids`` so the UI knows what's real vs synthetic.
    """
    underlying = underlying.upper().strip()
    today = today or date.today()

    if not legs:
        return LiveOpenResult(ok=False, reason="no_legs")

    # ── SEBI algo-framework gate. Live options are refused unless the operator
    # has a real (non-synthetic) options backtest and flips ALLOW_LIVE_OPTIONS,
    # and — in production — is exchange-empanelled. Honours the kill-switch. ──
    from ..compliance_gate import check_algo_order

    decision = check_algo_order(
        supabase=supabase, user_id=user_id, strategy_id=str(strategy_id or ""),
        segment="options", automated=True, live=True,
    )
    if not decision.allowed:
        logger.warning(
            "live_options: order refused for %s/%s (%s)",
            underlying, strategy_id, decision.reason,
        )
        return LiveOpenResult(ok=False, reason=f"compliance_block:{decision.reason}")

    broker, broker_name = _broker_for_user(supabase, user_id)
    if broker is None:
        return LiveOpenResult(ok=False, reason="no_broker_connected")
    if str(broker_name or "").lower() not in _FORMATTERS:
        return LiveOpenResult(
            ok=False,
            reason=f"live_options_unsupported_for_{broker_name}",
        )

    # Pre-trade margin check.
    available = None
    try:
        available = float(broker.get_available_margin() or 0)
    except Exception as exc:
        logger.debug("live_options_executor: margin probe failed: %s", exc)

    if estimated_margin is not None and available is not None:
        if available < estimated_margin:
            return LiveOpenResult(
                ok=False, reason="insufficient_margin",
                estimated_margin=estimated_margin,
                available_margin=available,
            )

    # Resolve legs to concrete strikes/expiries (same resolver the paper
    # path uses, so the strike picks match what the user backtested).
    resolved = resolve_legs(
        legs, spot=spot, symbol=underlying, today=today, sigma=sigma,
    )

    exchange = _INDEX_TO_EXCHANGE.get(underlying, "NFO")

    # Place legs sequentially. T1.3 (2026-05-31) — for defined-risk
    # structures (2+ legs spanning both BUY and SELL), if any leg fails
    # after others fill we ROLL BACK the filled legs immediately via
    # opposite-side MARKET orders. Reasoning: a partial Bull Call Spread
    # is a naked long call (different risk profile); leaving the user
    # exposed is more dangerous than an immediate market reverse.
    # Naked single-leg trades don't trigger rollback by definition.
    from ...data.brokers.integration import (
        Order, OrderType, ProductType, TransactionType,
    )

    # Determine if this structure REQUIRES rollback on partial fill.
    sides_to_place = {str(leg.side.value).upper() for leg in resolved}
    is_defined_risk = len(resolved) >= 2 and len(sides_to_place) >= 2

    placed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    broker_order_ids: List[str] = []

    for leg in resolved:
        tradingsymbol = _format_options_symbol(
            broker_name, underlying, leg.expiry, leg.strike, leg.option_type,
        )
        if not tradingsymbol:
            failed.append({
                "tradingsymbol": None,
                "side": leg.side.value,
                "reason": f"no_formatter_for_{broker_name}",
            })
            continue
        # Lot size is embedded in the contract for options — pass
        # qty_lots × deployment-lots × lot_size as quantity.
        try:
            from .paper_options_executor import _lot_size_for
            lot_size = _lot_size_for(underlying)
        except Exception:
            lot_size = 1
        qty = lots * leg.qty_lots * lot_size

        order = Order(
            symbol=tradingsymbol,
            exchange=exchange,
            transaction_type=(
                TransactionType.BUY if str(leg.side.value).upper() == "BUY"
                else TransactionType.SELL
            ),
            quantity=qty,
            product=ProductType.NRML,  # carry overnight; index options usually NRML
            order_type=OrderType.MARKET,
            price=0,
        )
        try:
            result = broker.place_order(order)
            status = getattr(result.status, "name", str(result.status)).upper() \
                if result.status else "UNKNOWN"
            if status == "REJECTED":
                failed.append({
                    "tradingsymbol": tradingsymbol,
                    "side": leg.side.value,
                    "qty": qty,
                    "reason": result.message or "rejected_by_broker",
                })
            else:
                broker_order_ids.append(result.order_id or "")
                placed.append({
                    "tradingsymbol": tradingsymbol,
                    "side": leg.side.value,
                    "qty": qty,
                    "broker_order_id": result.order_id,
                    "status": status,
                })
        except Exception as exc:
            failed.append({
                "tradingsymbol": tradingsymbol,
                "side": leg.side.value,
                "qty": qty,
                "reason": str(exc)[:200],
            })

    # T1.3 — Rollback on partial fill for defined-risk structures.
    # If 2+ legs spanning BUY+SELL and we have BOTH placed and failed
    # legs, reverse the placed legs immediately via opposite-side
    # MARKET orders. Slippage at unknown prices is accepted because the
    # alternative (naked exposure) is worse.
    rolled_back: List[Dict[str, Any]] = []
    rollback_failed: List[Dict[str, Any]] = []
    rollback_triggered = False
    if is_defined_risk and placed and failed:
        rollback_triggered = True
        logger.warning(
            "live_options_executor: partial fill on defined-risk %s "
            "for user %s — rolling back %d placed legs",
            template_slug or "custom", user_id, len(placed),
        )
        for p_leg in placed:
            # Reverse: BUY -> SELL, SELL -> BUY
            original_side = (p_leg.get("side") or "").upper()
            reverse_side = (
                TransactionType.SELL if original_side == "BUY"
                else TransactionType.BUY
            )
            reverse_order = Order(
                symbol=p_leg["tradingsymbol"],
                exchange=exchange,
                transaction_type=reverse_side,
                quantity=p_leg.get("qty", 0),
                product=ProductType.NRML,
                order_type=OrderType.MARKET,
                price=0,
            )
            try:
                rb_result = broker.place_order(reverse_order)
                rb_status = getattr(rb_result.status, "name", str(rb_result.status)).upper() \
                    if rb_result.status else "UNKNOWN"
                if rb_status == "REJECTED":
                    rollback_failed.append({
                        "tradingsymbol": p_leg["tradingsymbol"],
                        "original_side": original_side,
                        "reason": rb_result.message or "rollback_rejected",
                    })
                else:
                    rolled_back.append({
                        "tradingsymbol": p_leg["tradingsymbol"],
                        "original_side": original_side,
                        "reverse_order_id": rb_result.order_id,
                    })
            except Exception as exc:
                rollback_failed.append({
                    "tradingsymbol": p_leg["tradingsymbol"],
                    "original_side": original_side,
                    "reason": str(exc)[:200],
                })
        # If rollback succeeded for ALL placed legs, the position is
        # effectively flat — clear the placed list so the position
        # persists as 'rolled_back' rather than 'partial'.
        if not rollback_failed:
            placed = []
            logger.info(
                "live_options_executor: rollback complete — %d legs reversed",
                len(rolled_back),
            )

    # Persist a paper_option_positions row tagged as live so the UI can
    # show combined P&L. Same shape as paper, with live metadata.
    from .paper_options_executor import open_paper_option_position
    if rollback_triggered and not rollback_failed:
        position_status = "rolled_back"
    elif rollback_triggered and rollback_failed:
        position_status = "rollback_failed_alert"  # MUST surface to user
    elif not failed:
        position_status = "open"
    else:
        position_status = "partial"
    paper_res = open_paper_option_position(
        supabase=supabase,
        user_id=user_id,
        underlying=underlying,
        spot=spot,
        sigma=sigma,
        legs=legs,
        lots=lots,
        strategy_id=strategy_id,
        template_slug=template_slug,
        source="live_user_strategy" if not failed else "live_partial",
        today=today,
    )

    # Tag the row with live metadata (T1.3 adds rollback diagnostics)
    if paper_res.ok and paper_res.position_id:
        try:
            supabase.table("paper_option_positions").update({
                "metadata": {
                    "live": True,
                    "broker_name": broker_name,
                    "broker_order_ids": broker_order_ids,
                    "failed_legs": failed,
                    "placed_legs": placed,
                    "estimated_margin": estimated_margin,
                    "available_margin_at_open": available,
                    "status_detail": position_status,
                    # T1.3 rollback audit
                    "rollback_triggered": rollback_triggered,
                    "rolled_back_legs": rolled_back,
                    "rollback_failed_legs": rollback_failed,
                    "is_defined_risk": is_defined_risk,
                },
            }).eq("id", paper_res.position_id).execute()
        except Exception as exc:
            logger.debug("live_options_executor: metadata tag failed: %s", exc)

    # Reason code so callers can render the right UX:
    #   None                             — fully placed, all legs live
    #   partial_fill_naked_position      — partial, no rollback (single-side structure)
    #   partial_fill_rolled_back         — partial, rollback succeeded → position is flat
    #   partial_fill_rollback_failed     — partial, rollback failed → URGENT user alert
    if not failed:
        outcome_reason = None
    elif rollback_triggered and not rollback_failed:
        outcome_reason = "partial_fill_rolled_back"
    elif rollback_triggered and rollback_failed:
        outcome_reason = "partial_fill_rollback_failed"
    else:
        outcome_reason = "partial_fill_naked_position"

    return LiveOpenResult(
        ok=not failed,
        position_id=paper_res.position_id,
        placed_legs=placed,
        failed_legs=failed,
        estimated_margin=estimated_margin,
        available_margin=available,
        reason=outcome_reason,
    )


async def close_live_option_position(
    *,
    supabase: Any,
    user_id: str,
    position_id: str,
    spot: Optional[float] = None,
    sigma: Optional[float] = None,
    reason: str = "manual",
    today: Optional[date] = None,
) -> LiveCloseResult:
    """Close every leg of a live position by placing the opposite MARKET
    order at the broker, then update the combined paper_option_positions
    row to closed via the paper executor's helper.
    """
    today = today or date.today()
    pos = (
        supabase.table("paper_option_positions")
        .select("*")
        .eq("id", position_id)
        .eq("user_id", user_id)
        .single()
        .execute()
        .data
    )
    if not pos:
        return LiveCloseResult(ok=False, reason="position_not_found")

    meta = pos.get("metadata") or {}
    if not meta.get("live"):
        return LiveCloseResult(ok=False, reason="not_a_live_position")

    legs = (
        supabase.table("paper_option_legs")
        .select("*")
        .eq("position_id", position_id)
        .execute()
        .data
        or []
    )
    if not legs:
        return LiveCloseResult(ok=False, reason="no_legs")

    broker, broker_name = _broker_for_user(supabase, user_id)
    if broker is None:
        return LiveCloseResult(ok=False, reason="no_broker_connected")
    if str(broker_name or "").lower() not in _FORMATTERS:
        return LiveCloseResult(
            ok=False, reason=f"live_options_unsupported_for_{broker_name}",
        )

    from ...data.brokers.integration import (
        Order, OrderType, ProductType, TransactionType,
    )

    placed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    underlying = pos["underlying"]
    exchange = _INDEX_TO_EXCHANGE.get(underlying, "NFO")

    for L in legs:
        expiry_d = date.fromisoformat(L["expiry_date"])
        from ...ai.strategy.dsl import OptionType as _OT
        opt = _OT.CE if str(L["option_type"]).upper() == "CE" else _OT.PE
        tradingsymbol = _format_options_symbol(
            broker_name, underlying, expiry_d, float(L["strike"]), opt,
        )
        if not tradingsymbol:
            failed.append({
                "leg_id": L["id"],
                "reason": f"no_formatter_for_{broker_name}",
            })
            continue
        qty = int(L["lots"]) * int(L["lot_size"])
        # Closing direction is opposite of the open side.
        was_buy = str(L["side"]).upper() == "BUY"
        order = Order(
            symbol=tradingsymbol,
            exchange=exchange,
            transaction_type=TransactionType.SELL if was_buy else TransactionType.BUY,
            quantity=qty,
            product=ProductType.NRML,
            order_type=OrderType.MARKET,
            price=0,
        )
        try:
            result = broker.place_order(order)
            status = getattr(result.status, "name", str(result.status)).upper() \
                if result.status else "UNKNOWN"
            if status == "REJECTED":
                failed.append({
                    "leg_id": L["id"],
                    "tradingsymbol": tradingsymbol,
                    "reason": result.message or "rejected",
                })
            else:
                placed.append({
                    "leg_id": L["id"],
                    "tradingsymbol": tradingsymbol,
                    "broker_order_id": result.order_id,
                    "status": status,
                })
        except Exception as exc:
            failed.append({
                "leg_id": L["id"],
                "tradingsymbol": tradingsymbol,
                "reason": str(exc)[:200],
            })

    # Mark the combined position closed via the paper executor (handles
    # leg.exit_price + trades row + realized_pnl bookkeeping).
    from .paper_options_executor import close_paper_option_position
    paper_res = close_paper_option_position(
        supabase=supabase,
        position_id=position_id,
        user_id=user_id,
        spot=spot,
        sigma=sigma,
        reason=reason,
        source="live_manual" if reason == "manual" else "live_user_strategy",
        today=today,
    )

    # Tag close-side broker metadata
    try:
        existing_meta = {**(meta or {}),
                         "close_placed_legs": placed,
                         "close_failed_legs": failed,
                         "closed_via": "live"}
        supabase.table("paper_option_positions").update({
            "metadata": existing_meta,
        }).eq("id", position_id).execute()
    except Exception:
        pass

    return LiveCloseResult(
        ok=paper_res.ok and not failed,
        position_id=position_id,
        placed_legs=placed,
        failed_legs=failed,
        reason=None if (paper_res.ok and not failed)
        else (paper_res.reason or "partial_close_manual_resolve"),
    )
