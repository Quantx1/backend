"""Portfolio context for the AI advisor (PR-BE).

Pulls the user's open positions (equity + multi-leg options) and
computes a delta exposure summary that the AI strategy advisor can use
to suggest hedges. Pure read; never writes.

Net delta convention:
  +1 per long share, -1 per short share.
  For options: option_delta × lot_size × lots × side_sign (+1 BUY, -1 SELL).

The summary text block is prompt-ready: short, numbers-first, no fluff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortfolioContext:
    """What the AI advisor needs to make a hedge-aware suggestion."""
    equity_positions: List[Dict[str, Any]] = field(default_factory=list)
    option_positions: List[Dict[str, Any]] = field(default_factory=list)
    # Delta totals in rupee-equivalent (delta × spot for context size)
    equity_delta_shares: float = 0.0
    equity_delta_inr: float = 0.0
    option_delta_shares: float = 0.0   # signed share-equivalent from options
    net_delta_inr: float = 0.0
    has_positions: bool = False
    # PR-BF.1 — Per-underlying breakdown so "hedge my RELIANCE" works:
    #   { 'RELIANCE': {'equity_delta_inr': 150000, 'option_delta_inr': 0,
    #                  'total_delta_inr': 150000, 'bias': 'LONG'},
    #     'NIFTY':    {'equity_delta_inr': 0, 'option_delta_inr': -28000, ...} }
    by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Prompt-ready one-paragraph block (or '' when no positions)
    prompt_block: str = ""


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    """First key that returns a non-None value wins."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _resolve_spot(supabase: Any, symbol: str) -> Optional[float]:
    """Best-effort spot for an underlying — for delta×spot rupee figures."""
    try:
        from ...data.market import get_market_data_provider
        q = get_market_data_provider().get_quote(symbol)
        ltp = getattr(q, "ltp", None) or (q.get("ltp") if isinstance(q, dict) else None)
        return float(ltp) if ltp else None
    except Exception:
        return None


def compute_user_book_context(
    supabase: Any,
    user_id: str,
    *,
    include_paper: bool = True,
) -> PortfolioContext:
    """Compute the user's net delta exposure for AI hedge sizing.

    ``include_paper`` controls whether paper positions are included.
    Default True — for paper-only beta users the suggestion still makes
    sense because the runner can deploy a hedge into paper too.
    """
    ctx = PortfolioContext()

    # ── Equity positions ──────────────────────────────────────────────
    equity_rows: List[Dict[str, Any]] = []
    try:
        live = (
            supabase.table("positions")
            .select("symbol, quantity, average_price, current_price, direction, execution_mode")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .limit(100)
            .execute()
            .data
            or []
        )
        equity_rows.extend(live)
    except Exception as exc:
        logger.debug("portfolio_context: live positions fetch failed: %s", exc)

    if include_paper:
        try:
            paper = (
                supabase.table("paper_positions")
                .select("symbol, qty, entry_price")
                .eq("user_id", user_id)
                .eq("status", "open")
                .limit(100)
                .execute()
                .data
                or []
            )
            # Normalise paper shape to equity shape for unified math
            for p in paper:
                equity_rows.append({
                    "symbol": p.get("symbol"),
                    "quantity": p.get("qty"),
                    "average_price": p.get("entry_price"),
                    "current_price": p.get("entry_price"),  # MTM happens elsewhere
                    "direction": "LONG",
                    "execution_mode": "paper",
                })
        except Exception as exc:
            logger.debug("portfolio_context: paper positions fetch failed: %s", exc)

    # Aggregate equity delta
    for r in equity_rows:
        qty = int(_safe_get(r, "quantity", "qty") or 0)
        if qty == 0:
            continue
        direction = str(r.get("direction") or "LONG").upper()
        signed_qty = qty if direction == "LONG" else -qty
        price = float(_safe_get(r, "current_price", "average_price") or 0)
        ctx.equity_delta_shares += signed_qty
        ctx.equity_delta_inr += signed_qty * price
        ctx.equity_positions.append({
            "symbol": r.get("symbol"),
            "quantity": signed_qty,
            "price": round(price, 2),
            "exposure_inr": round(signed_qty * price, 2),
            "mode": r.get("execution_mode") or "live",
        })

    # ── Option positions (combined multi-leg) ─────────────────────────
    try:
        option_rows = (
            supabase.table("paper_option_positions")
            .select("id, underlying, status, net_premium, unrealized_pnl, max_loss, metadata")
            .eq("user_id", user_id)
            .eq("status", "open")
            .limit(50)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.debug("portfolio_context: option positions fetch failed: %s", exc)
        option_rows = []

    if option_rows:
        # Bulk-load every leg in one query
        pos_ids = [p["id"] for p in option_rows]
        leg_rows: List[Dict[str, Any]] = []
        try:
            leg_rows = (
                supabase.table("paper_option_legs")
                .select("position_id, side, option_type, strike, expiry_date, lots, lot_size")
                .in_("position_id", pos_ids)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            logger.debug("portfolio_context: legs bulk fetch failed: %s", exc)
        legs_by_pos: Dict[str, List[Dict[str, Any]]] = {}
        for leg in leg_rows:
            legs_by_pos.setdefault(leg["position_id"], []).append(leg)

        # Greeks at each leg — reuse the same enrich helper as the chain
        from ..execution.options_greeks import compute_greeks
        for p in option_rows:
            underlying = str(p.get("underlying") or "NIFTY").upper()
            spot = _resolve_spot(supabase, underlying)
            legs = legs_by_pos.get(p["id"], [])
            if not legs or not spot:
                continue
            sigma_at_entry = float((p.get("metadata") or {}).get("sigma_at_entry") or 0.20)
            today = date.today()
            position_delta_shares = 0.0
            for L in legs:
                expiry_d = date.fromisoformat(L["expiry_date"])
                T = max((expiry_d - today).days, 0) / 365.0
                if T <= 0:
                    continue
                is_call = str(L["option_type"]).upper() == "CE"
                g = compute_greeks(
                    S=spot, K=float(L["strike"]), T=T,
                    sigma=sigma_at_entry, is_call=is_call,
                )
                side_sign = 1.0 if str(L["side"]).upper() == "BUY" else -1.0
                leg_delta_shares = (
                    g.delta * int(L["lots"]) * int(L["lot_size"]) * side_sign
                )
                position_delta_shares += leg_delta_shares
            ctx.option_delta_shares += position_delta_shares
            ctx.option_positions.append({
                "id": p["id"],
                "underlying": underlying,
                "spot": spot,
                "delta_shares": round(position_delta_shares, 2),
                "delta_inr": round(position_delta_shares * spot, 0),
                "unrealized_pnl": float(p.get("unrealized_pnl") or 0),
                "max_loss": p.get("max_loss"),
            })

    # ── Net delta + prompt block ──────────────────────────────────────
    # For options we measure delta in SHARES, but rupee exposure depends
    # on each underlying's spot. Sum per-position delta_inr (computed
    # above) instead of multiplying once.
    option_delta_inr = sum(p["delta_inr"] for p in ctx.option_positions)
    ctx.net_delta_inr = ctx.equity_delta_inr + option_delta_inr
    ctx.has_positions = bool(ctx.equity_positions or ctx.option_positions)

    # ── PR-BF.1 — Per-underlying delta breakdown ──────────────────────
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for r in ctx.equity_positions:
        sym = str(r["symbol"] or "").upper()
        if not sym:
            continue
        d = by_symbol.setdefault(sym, {
            "equity_delta_inr": 0.0,
            "option_delta_inr": 0.0,
            "total_delta_inr": 0.0,
            "equity_qty": 0,
            "options_count": 0,
        })
        d["equity_delta_inr"] += float(r["exposure_inr"])
        d["equity_qty"] += int(r["quantity"])
    for p in ctx.option_positions:
        sym = str(p["underlying"] or "").upper()
        if not sym:
            continue
        d = by_symbol.setdefault(sym, {
            "equity_delta_inr": 0.0,
            "option_delta_inr": 0.0,
            "total_delta_inr": 0.0,
            "equity_qty": 0,
            "options_count": 0,
        })
        d["option_delta_inr"] += float(p["delta_inr"])
        d["options_count"] += 1
    for sym, d in by_symbol.items():
        d["total_delta_inr"] = round(d["equity_delta_inr"] + d["option_delta_inr"], 2)
        d["equity_delta_inr"] = round(d["equity_delta_inr"], 2)
        d["option_delta_inr"] = round(d["option_delta_inr"], 2)
        d["bias"] = (
            "LONG" if d["total_delta_inr"] > 25_000
            else "SHORT" if d["total_delta_inr"] < -25_000
            else "FLAT"
        )
    ctx.by_symbol = by_symbol

    if not ctx.has_positions:
        ctx.prompt_block = "User has no open positions."
        return ctx

    lines = ["Current book:"]
    if ctx.equity_positions:
        top_eq = sorted(
            ctx.equity_positions, key=lambda r: abs(r["exposure_inr"]), reverse=True,
        )[:5]
        lines.append(
            f"  Equity: {len(ctx.equity_positions)} positions, "
            f"net delta ₹{ctx.equity_delta_inr:,.0f}. "
            f"Top: " + ", ".join(
                f"{r['symbol']} {('+' if r['quantity'] > 0 else '')}{r['quantity']} "
                f"(₹{r['exposure_inr']:,.0f})"
                for r in top_eq
            )
        )
    if ctx.option_positions:
        lines.append(
            f"  Options: {len(ctx.option_positions)} open multi-leg positions, "
            f"net delta ₹{option_delta_inr:,.0f}. "
            f"Underlyings: " + ", ".join(
                set(p["underlying"] for p in ctx.option_positions)
            )
        )
    lines.append(
        f"  NET DELTA ₹{ctx.net_delta_inr:,.0f} "
        f"({'LONG bias' if ctx.net_delta_inr > 0 else 'SHORT bias' if ctx.net_delta_inr < 0 else 'NEUTRAL'})"
    )
    if ctx.by_symbol:
        top_3 = sorted(
            ctx.by_symbol.items(),
            key=lambda kv: abs(kv[1]["total_delta_inr"]),
            reverse=True,
        )[:3]
        per_sym = ", ".join(
            f"{sym} ₹{d['total_delta_inr']:,.0f} ({d['bias']})"
            for sym, d in top_3
        )
        lines.append(f"  Top exposures: {per_sym}")
    ctx.prompt_block = "\n".join(lines)
    return ctx
