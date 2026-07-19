"""Paper multi-leg options executor (PR-AT).

Parallel of paper_executor.py but for multi-leg option positions
(Bull Call Spread, Iron Condor, Long Straddle, etc).

Public surface:
  open_paper_option_position(supabase, user_id, ...) → OpenResult
  close_paper_option_position(supabase, position_id, ...) → CloseResult
  mark_to_market(supabase, position) → MTMResult

Premium model:
  Black-Scholes mid price, same _bs_price helper used by the existing
  options backtest engine — keeps paper P&L coherent with what the
  backtest showed before deploy. Live option chain integration is a
  follow-up (broker chain API is per-broker; BS estimate gets us 90%
  of the way for free).

Lot-size lookup:
  Index lots (NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY/SENSEX) are
  well-known constants; equity lots come from the existing
  trading/fo/instruments.py InstrumentMaster.

Open math:
  Net premium = Σ (side_sign × premium_per_share) × lot_size × lots
    side BUY  → -1 (we pay)
    side SELL → +1 (we collect)
  Stored as a positive number when we paid, negative when we collected,
  on paper_option_positions.net_premium. (Convention chosen to match
  the trader's intuition: "I paid ₹X" → +X.)

Close math:
  Mark every leg to current spot via BS, then exit at the marked price.
  pnl = entry_premium_paid - current_exit_premium  (signs handled per leg)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from ...ai.strategy.dsl import LegSpec, OptionType
from ...ai.strategy.options_backtest import _bs_price
from ...ai.strategy.options_resolver import resolve_legs, ResolvedLeg

logger = logging.getLogger(__name__)


# Default risk-free rate (RBI repo proxy). Same as options_backtest.
RISK_FREE_RATE = 0.065

# Lot-size table for indices — these are NSE-published constants that
# rarely change. Equity lots come from InstrumentMaster.
INDEX_LOT_SIZES: Dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 15,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 75,
    "SENSEX": 10,
}


# ── results ──────────────────────────────────────────────────────────


@dataclass
class OpenResult:
    ok: bool
    position_id: Optional[str] = None
    trade_id: Optional[str] = None
    net_premium: Optional[float] = None
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    legs: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class CloseResult:
    ok: bool
    position_id: Optional[str] = None
    trade_id: Optional[str] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class MTMResult:
    current_value: float
    unrealized_pnl: float
    legs: List[Dict[str, Any]] = field(default_factory=list)
    # PR-AX — how the mark was sourced:
    #   'chain'  → broker option-chain LTPs (most accurate)
    #   'mixed'  → some legs chain, some BS (chain missing those strikes)
    #   'bs'     → all legs BS-estimated (no broker / chain unavailable)
    source: str = "bs"


# ── helpers ──────────────────────────────────────────────────────────


def _lot_size_for(underlying: str) -> int:
    """NIFTY → 50, BANKNIFTY → 15, RELIANCE → equity lot from master."""
    u = underlying.upper().strip()
    if u in INDEX_LOT_SIZES:
        return INDEX_LOT_SIZES[u]
    # Fallback to InstrumentMaster for equity F&O lots.
    try:
        from ...core.config import settings
        from ...trading.fo.instruments import InstrumentMaster
        master = InstrumentMaster(settings.FNO_INSTRUMENTS_FILE)
        if master.available():
            row = master.get_options_lot_size(u) if hasattr(master, "get_options_lot_size") else None
            if row:
                return int(row)
    except Exception:
        pass
    # Conservative default; better to under-size than over-size on a fallback.
    return 1


def _price_legs(
    resolved: List[ResolvedLeg],
    *,
    spot: float,
    today: date,
    sigma: float,
) -> List[Dict[str, Any]]:
    """Price every resolved leg via BS. Returns per-leg per-SHARE premium."""
    out: List[Dict[str, Any]] = []
    for leg in resolved:
        days_to_expiry = max((leg.expiry - today).days, 0)
        T = days_to_expiry / 365.25
        prem = _bs_price(
            spot, leg.strike, T, RISK_FREE_RATE, sigma,
            is_call=(leg.option_type == OptionType.CE),
        )
        out.append({
            "side": leg.side.value,
            "option_type": leg.option_type.value,
            "strike": float(leg.strike),
            "expiry": leg.expiry,
            "qty_lots": leg.qty_lots,
            "premium": round(prem, 4),
        })
    return out


def _net_premium_per_lot(legs: List[Dict[str, Any]]) -> float:
    """Net premium per ONE deployment lot (BUY paid as positive convention).

    Returns positive when the trader paid net debit, negative when net credit.
    Side comparisons are upper-cased — DSL enum values are 'buy'/'sell'
    lowercase but Postgres CHECK + storage use 'BUY'/'SELL'.
    """
    net = 0.0
    for L in legs:
        sign = 1.0 if str(L["side"]).upper() == "BUY" else -1.0
        net += sign * float(L["premium"]) * float(L["qty_lots"])
    return net


def _payoff_at_expiry(
    legs: List[Dict[str, Any]], spot_at_expiry: float,
) -> float:
    """Combined per-lot payoff at expiry. + flows toward the holder.

    BUY: you own the option → ITM intrinsic flows TO you (+)
    SELL: you're short → ITM intrinsic flows AWAY (-)
    """
    out = 0.0
    for L in legs:
        opt_type = str(L["option_type"]).upper()
        side_u = str(L["side"]).upper()
        is_call = opt_type == "CE"
        intrinsic = max(0.0, spot_at_expiry - L["strike"]) if is_call \
            else max(0.0, L["strike"] - spot_at_expiry)
        sign = 1.0 if side_u == "BUY" else -1.0
        out += sign * intrinsic * float(L["qty_lots"])
    return out


def _max_profit_loss(
    legs: List[Dict[str, Any]], net_premium_per_lot: float, spot: float,
) -> tuple[Optional[float], Optional[float]]:
    """Sample the expiry payoff curve at the leg strikes + outer bands
    to bracket max profit / max loss. Per-lot figures.

    Returns (max_profit, max_loss) where loss is reported positive.
    Either may be None for unlimited-risk structures (naked short).
    """
    strikes = sorted({float(L["strike"]) for L in legs})
    if not strikes:
        return None, None
    span = max(strikes) - min(strikes) + max(1.0, spot * 0.10)
    samples = [
        max(0.01, strikes[0] - span),
        *strikes,
        strikes[-1] + span,
        # Midpoints between strikes for crisper extrema
        *[(strikes[i] + strikes[i + 1]) / 2 for i in range(len(strikes) - 1)],
    ]
    # Total profit at each sampled expiry spot:
    #   profit = payoff_to_holder + entry_cashflow
    # entry_cashflow = -net_premium_per_lot (you paid net debit → cash out)
    # so profit = payoff - net_premium_per_lot
    payoffs = [
        _payoff_at_expiry(legs, s) - net_premium_per_lot
        for s in samples
    ]
    max_p = max(payoffs)
    min_p = min(payoffs)
    # Naked short detection: a SELL leg with no matching BUY of the same
    # option_type → unlimited-risk side.

    def _side(x):
        return str(x.get("side", "")).upper()

    def _opt(x):
        return str(x.get("option_type", "")).upper()
    has_unlimited_risk = any(
        _side(L) == "SELL" and not any(
            _opt(other) == _opt(L) and _side(other) == "BUY"
            for other in legs
        )
        for L in legs
    )
    return (
        round(max_p, 2),
        None if has_unlimited_risk else round(-min_p, 2),
    )


# ── public API ────────────────────────────────────────────────────────


def open_paper_option_position(
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
    source: str = "manual",
    today: Optional[date] = None,
) -> OpenResult:
    """Materialize a multi-leg position at current spot + IV estimate.

    ``lots`` here is the DEPLOYMENT size — number of units of the whole
    multi-leg structure. Each leg's qty_lots is the SHAPE ratio
    (1:1 spread, 2:1 ratio spread, etc) and stays as-is.
    """
    underlying = underlying.upper().strip()
    if not legs:
        return OpenResult(ok=False, reason="no_legs")
    if lots <= 0:
        return OpenResult(ok=False, reason="invalid_lots")
    if spot <= 0:
        return OpenResult(ok=False, reason="invalid_spot")
    if sigma <= 0:
        sigma = 0.20  # 20% default if caller didn't supply realized vol

    today = today or date.today()

    try:
        resolved = resolve_legs(legs, spot=spot, symbol=underlying,
                                today=today, sigma=sigma)
    except Exception as exc:
        return OpenResult(ok=False, reason=f"resolve_failed: {exc}")

    priced = _price_legs(resolved, spot=spot, today=today, sigma=sigma)
    net_per_lot = _net_premium_per_lot(priced)
    lot_size = _lot_size_for(underlying)
    net_premium_total = net_per_lot * lot_size * lots  # +debit / -credit
    max_p, max_l = _max_profit_loss(priced, net_per_lot, spot=spot)
    max_profit_total = round(max_p * lot_size * lots, 2) if max_p is not None else None
    max_loss_total = round(max_l * lot_size * lots, 2) if max_l is not None else None

    # Use the earliest leg expiry as the position-level expiry (calendar
    # spreads keep the long-dated one, but the front-month is what
    # gamma-decays first).
    position_expiry = min(L["expiry"] for L in priced)

    position_id = str(uuid.uuid4())
    pos_row = {
        "id": position_id,
        "user_id": user_id,
        "strategy_id": strategy_id,
        "template_slug": template_slug,
        "underlying": underlying,
        "expiry_date": position_expiry.isoformat(),
        "net_premium": round(net_premium_total, 2),
        "max_profit": max_profit_total,
        "max_loss": max_loss_total,
        "current_value": round(net_premium_total, 2),
        "unrealized_pnl": 0,
        "realized_pnl": 0,
        "status": "open",
        "metadata": {
            "spot_at_entry": round(float(spot), 2),
            "sigma_at_entry": round(float(sigma), 4),
            "lots_deployed": lots,
            "lot_size": lot_size,
        },
    }
    try:
        supabase.table("paper_option_positions").insert(pos_row).execute()
    except Exception as exc:
        logger.warning("paper_options: position insert failed: %s", exc)
        return OpenResult(ok=False, reason=f"insert_failed: {exc}")

    leg_rows = [
        {
            "id": str(uuid.uuid4()),
            "position_id": position_id,
            # DSL enum stores 'buy'/'sell' lowercase; schema CHECK expects
            # 'BUY'/'SELL' uppercase. Normalize at the persistence boundary.
            "side": str(L["side"]).upper(),
            "option_type": str(L["option_type"]).upper(),
            "strike": L["strike"],
            "expiry_date": L["expiry"].isoformat(),
            "lots": int(lots * L["qty_lots"]),  # actual deployed lots for this leg
            "lot_size": lot_size,
            "entry_price": L["premium"],
            "current_price": L["premium"],
        }
        for L in priced
    ]
    try:
        supabase.table("paper_option_legs").insert(leg_rows).execute()
    except Exception as exc:
        # Roll back the position row to avoid orphaned legless records.
        supabase.table("paper_option_positions").delete().eq("id", position_id).execute()
        return OpenResult(ok=False, reason=f"legs_insert_failed: {exc}")

    trade_id = str(uuid.uuid4())
    try:
        supabase.table("paper_option_trades").insert({
            "id": trade_id,
            "user_id": user_id,
            "position_id": position_id,
            "action": "open",
            "source": source,
            "metadata": {
                "underlying": underlying,
                "spot": round(float(spot), 2),
                "net_premium": round(net_premium_total, 2),
            },
        }).execute()
    except Exception as exc:
        logger.debug("paper_options: open trade insert failed: %s", exc)

    return OpenResult(
        ok=True, position_id=position_id, trade_id=trade_id,
        net_premium=round(net_premium_total, 2),
        max_profit=max_profit_total,
        max_loss=max_loss_total,
        legs=leg_rows,
    )


def mark_to_market(
    supabase: Any, position_row: Dict[str, Any], *,
    spot: Optional[float] = None, sigma: Optional[float] = None,
    today: Optional[date] = None,
) -> MTMResult:
    """Reprice every leg at current spot. Updates the row in place.

    Caller can pass spot/sigma if they already have them; otherwise we
    fetch a quote (BS-based mark needs a spot reference).
    """
    today = today or date.today()
    underlying = position_row["underlying"]

    if spot is None:
        # Best-effort spot fetch — the same market provider the dashboard uses.
        try:
            from ...data.market import get_market_data_provider
            quote = get_market_data_provider().get_quote(underlying)
            spot = float(
                getattr(quote, "ltp", None)
                or (quote.get("ltp") if isinstance(quote, dict) else 0)
                or 0
            )
        except Exception:
            spot = None

    if sigma is None:
        sigma = float((position_row.get("metadata") or {}).get("sigma_at_entry") or 0.20)

    legs = (
        supabase.table("paper_option_legs")
        .select("id, side, option_type, strike, expiry_date, lots, lot_size, entry_price")
        .eq("position_id", position_row["id"])
        .execute()
        .data
        or []
    )

    if not legs or not spot or spot <= 0:
        return MTMResult(
            current_value=float(position_row.get("net_premium") or 0),
            unrealized_pnl=0.0,
        )

    # PR-AX — Try the user's broker option chain first; fall back to
    # BS estimate per-leg when the chain doesn't carry that strike or
    # the user has no connected broker. The position's expiry rarely
    # spans multiple weeks, so we pull the front-month chain once.
    chain = None
    try:
        from .option_chain import get_option_chain, lookup_leg_ltp
        front_expiry = min(
            (date.fromisoformat(L["expiry_date"]) for L in legs),
            default=None,
        )
        if position_row.get("user_id") and front_expiry is not None:
            chain = get_option_chain(
                supabase,
                user_id=position_row["user_id"],
                symbol=str(underlying),
                expiry=front_expiry,
            )
    except Exception as exc:
        logger.debug("mark_to_market: chain fetch skipped: %s", exc)
        chain = None

    current_value = 0.0
    chain_hits = 0
    bs_hits = 0
    updated_legs: List[Dict[str, Any]] = []
    for L in legs:
        expiry_d = date.fromisoformat(L["expiry_date"])
        T = max((expiry_d - today).days, 0) / 365.25
        opt_type = str(L["option_type"]).upper()
        side_u = str(L["side"]).upper()

        # Prefer broker LTP from the chain when available.
        prem: Optional[float] = None
        prem_source = "bs"
        if chain:
            ltp = lookup_leg_ltp(
                chain,
                strike=float(L["strike"]),
                expiry=expiry_d,
                option_type=opt_type,
            )
            if ltp is not None and ltp > 0:
                prem = ltp
                prem_source = "chain"
                chain_hits += 1

        if prem is None:
            prem = _bs_price(
                spot, float(L["strike"]), T, RISK_FREE_RATE, sigma,
                is_call=(opt_type == "CE"),
            )
            bs_hits += 1

        # O.5 (2026-05-31) — Refresh per-leg Greeks at current spot so the
        # UI shows live delta/gamma/theta/vega instead of stale entry-time
        # values. Position-level aggregates are summed below.
        leg_greeks = None
        try:
            from .options_greeks import compute_greeks
            leg_greeks = compute_greeks(
                S=spot, K=float(L["strike"]), T=T, r=RISK_FREE_RATE, sigma=sigma,
                is_call=(opt_type == "CE"),
            )
        except Exception as gex:
            logger.debug("Greeks refresh failed for leg %s: %s", L.get("id"), gex)

        # current_value uses the SAME convention as entry net_premium:
        # positive = you'd pay debit to reopen this exact position right
        # now. BUY adds to debit, SELL subtracts.
        sign = 1.0 if side_u == "BUY" else -1.0
        qty = int(L["lots"]) * int(L["lot_size"])
        current_value += sign * prem * qty
        leg_update = {
            "id": L["id"],
            "current_price": round(prem, 4),
            "price_source": prem_source,
        }
        if leg_greeks is not None:
            # Signed Greeks (negative for short legs) so position-level
            # sums match the actual exposure the trader is carrying.
            leg_update["current_delta"] = round(sign * leg_greeks.delta * qty, 4)
            leg_update["current_gamma"] = round(sign * leg_greeks.gamma * qty, 6)
            leg_update["current_theta"] = round(sign * leg_greeks.theta * qty, 4)
            leg_update["current_vega"] = round(sign * leg_greeks.vega * qty, 4)
        updated_legs.append(leg_update)

    if chain_hits > 0 and bs_hits == 0:
        mtm_source = "chain"
    elif chain_hits > 0:
        mtm_source = "mixed"
    else:
        mtm_source = "bs"

    entry_net = float(position_row.get("net_premium") or 0)
    # P&L = current_value − entry_net.
    # For a debit position (entry_net > 0): if current_value rises above
    # entry_net, the position is worth more than you paid → profit.
    # For a credit position (entry_net < 0): if current_value rises
    # toward zero (less negative), the position is closer to worthless →
    # the seller's intent → profit. Both cases reduce to the same diff.
    unrealized = round(current_value - entry_net, 2)

    # Aggregate position-level Greeks from per-leg signed values.
    agg_delta = sum(u.get("current_delta", 0) for u in updated_legs)
    agg_gamma = sum(u.get("current_gamma", 0) for u in updated_legs)
    agg_theta = sum(u.get("current_theta", 0) for u in updated_legs)
    agg_vega = sum(u.get("current_vega", 0) for u in updated_legs)

    # Batch-update leg current prices + Greeks.
    for u in updated_legs:
        try:
            patch = {"current_price": u["current_price"]}
            for k in ("current_delta", "current_gamma", "current_theta", "current_vega"):
                if k in u:
                    patch[k] = u[k]
            supabase.table("paper_option_legs").update(patch).eq("id", u["id"]).execute()
        except Exception:
            pass

    try:
        supabase.table("paper_option_positions").update({
            "current_value": round(current_value, 2),
            "unrealized_pnl": unrealized,
            "last_marked_at": datetime.utcnow().isoformat() + "Z",
            # O.5 aggregates so the dashboard can show live Greeks per position
            "current_delta": round(agg_delta, 4),
            "current_gamma": round(agg_gamma, 6),
            "current_theta": round(agg_theta, 4),
            "current_vega": round(agg_vega, 4),
        }).eq("id", position_row["id"]).execute()
    except Exception:
        pass

    return MTMResult(
        current_value=round(current_value, 2),
        unrealized_pnl=unrealized,
        legs=updated_legs,
        source=mtm_source,
    )


def close_paper_option_position(
    *,
    supabase: Any,
    position_id: str,
    user_id: str,
    spot: Optional[float] = None,
    sigma: Optional[float] = None,
    reason: str = "manual",
    source: str = "manual",
    today: Optional[date] = None,
) -> CloseResult:
    """Close the multi-leg position by marking every leg to current
    spot price and realising the difference vs entry net premium.
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
        return CloseResult(ok=False, reason="position_not_found")
    if pos.get("status") != "open":
        return CloseResult(ok=False, reason="position_not_open")

    mtm = mark_to_market(supabase, pos, spot=spot, sigma=sigma, today=today)
    entry_net = float(pos.get("net_premium") or 0)
    # Same convention as mark_to_market.unrealized_pnl.
    realized = round(mtm.current_value - entry_net, 2)
    realized_pct = (
        round((realized / abs(entry_net)) * 100, 4) if entry_net else None
    )

    now_iso = datetime.utcnow().isoformat() + "Z"
    try:
        supabase.table("paper_option_positions").update({
            "status": "closed",
            "exit_reason": reason if reason in
            ("target", "stop_loss", "manual", "expiry", "dsl_exit", "time") else "manual",
            "realized_pnl": realized,
            "current_value": mtm.current_value,
            "closed_at": now_iso,
        }).eq("id", position_id).execute()
    except Exception as exc:
        return CloseResult(ok=False, reason=f"update_failed: {exc}")

    # Update exit_price on each leg
    legs = supabase.table("paper_option_legs").select("id, current_price").eq(
        "position_id", position_id,
    ).execute().data or []
    for L in legs:
        try:
            supabase.table("paper_option_legs").update({
                "exit_price": L.get("current_price"),
            }).eq("id", L["id"]).execute()
        except Exception:
            pass

    trade_id = str(uuid.uuid4())
    try:
        supabase.table("paper_option_trades").insert({
            "id": trade_id,
            "user_id": user_id,
            "position_id": position_id,
            "action": "close",
            "pnl": realized,
            "pnl_pct": realized_pct,
            "source": source,
            "metadata": {"reason": reason},
        }).execute()
    except Exception as exc:
        logger.debug("paper_options: close trade insert failed: %s", exc)

    return CloseResult(
        ok=True, position_id=position_id, trade_id=trade_id,
        realized_pnl=realized, realized_pnl_pct=realized_pct,
    )
