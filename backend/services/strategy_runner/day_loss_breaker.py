"""Per-strategy day-loss circuit breaker (PR-AR.1).

Tier-level limits (subscription) + global kill switch existed before
this. What was missing: a per-strategy "if today's P&L on this strategy
falls below X%, stop placing new entries and auto-pause it" — the
single most important safety net a deployed trader expects.

Why per-strategy and not per-user:
  A user might run 5 different strategies. One blowing up shouldn't
  block the others. Each strategy carries its own max_daily_loss_pct
  (defaults to PLATFORM_DEFAULT_MAX_DAY_LOSS_PCT, 3%) and is paused
  independently.

What "today's P&L on this strategy" means:
  realized   — sum(exit_price - entry_price)*qty for strategy_positions
               closed today via this strategy_id
  unrealized — sum(current_market_price - entry_price)*qty for open
               strategy_positions on this strategy_id, marked to the
               same live quotes the position sweep uses

  Together they're the day-loss figure measured against capital_deployed
  on the open + closed positions for the day.

When the breaker fires:
  - user_strategies.status flips to 'paused'
  - pause_reason = 'day_loss_breach'
  - A 'breach' signal is inserted on `signals` so the user sees it in
    the activity feed
  - Returns True so the caller (runner) skips the entry

The breaker is read-only when checking; it only writes when actually
tripping. This is called inside the entry gate in _emit_entry BEFORE
the trades-row insert / broker call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Conservative default — a beginner-friendly cap that a sensible swing
# trader wouldn't override downwards. Power users can set their own per
# strategy via the DSL (max_daily_loss_pct field).
PLATFORM_DEFAULT_MAX_DAY_LOSS_PCT = 3.0


@dataclass
class BreakerCheck:
    """Result of one breach evaluation."""
    breached: bool
    current_pnl_pct: float  # negative when losing
    threshold_pct: float    # always negative (e.g. -3.0)
    capital_deployed: float
    realized_pnl: float
    unrealized_pnl: float
    reason_text: Optional[str] = None


def evaluate_strategy_breaker(
    supabase: Any,
    user_id: str,
    strategy_id: str,
    strategy_row: Dict[str, Any],
    *,
    quote_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> BreakerCheck:
    """Compute the current day-loss state. No DB writes.

    quote_map is optional — if the caller (position sweep) already has
    fresh quotes for the open symbols, pass them in to skip a redundant
    batch fetch. Otherwise the breaker fetches per-symbol lazily; for a
    single _emit_entry check this is fine.
    """
    threshold_raw = strategy_row.get("max_daily_loss_pct")
    threshold = (
        float(threshold_raw) if threshold_raw is not None
        else PLATFORM_DEFAULT_MAX_DAY_LOSS_PCT
    )
    # Stored as a positive percent (3.0 means 3%); compared as negative.
    threshold_signed = -abs(threshold)

    today_iso = date.today().isoformat()

    # ── Realized P&L from positions closed TODAY via this strategy ──
    realized = 0.0
    capital_deployed = 0.0
    try:
        closed_today = (
            supabase.table("strategy_positions")
            .select("entry_price, exit_price, quantity, capital_deployed, last_evaluated_at")
            .eq("user_id", user_id)
            .eq("strategy_id", strategy_id)
            .eq("status", "closed")
            .gte("last_evaluated_at", today_iso)
            .limit(200)
            .execute()
            .data
            or []
        )
        for p in closed_today:
            qty = int(p.get("quantity") or 0)
            entry = float(p.get("entry_price") or 0)
            exit_p = float(p.get("exit_price") or 0)
            if qty and entry and exit_p:
                realized += (exit_p - entry) * qty
            capital_deployed += float(p.get("capital_deployed") or (entry * qty))
    except Exception as exc:
        logger.debug("breaker: closed_today lookup failed: %s", exc)

    # ── Unrealized P&L from currently-open positions on this strategy ──
    unrealized = 0.0
    try:
        open_rows = (
            supabase.table("strategy_positions")
            .select("symbol, entry_price, quantity, capital_deployed")
            .eq("user_id", user_id)
            .eq("strategy_id", strategy_id)
            .eq("status", "open")
            .limit(200)
            .execute()
            .data
            or []
        )
        if open_rows:
            need_symbols = [
                r["symbol"] for r in open_rows
                if r.get("symbol") and (not quote_map or r["symbol"] not in quote_map)
            ]
            local_quotes: Dict[str, Dict[str, Any]] = dict(quote_map or {})
            if need_symbols:
                try:
                    from ...data.market import get_market_data_provider
                    provider = get_market_data_provider()
                    raw = provider.get_quotes_batch(need_symbols[:50])
                    for sym, q in (raw or {}).items():
                        if not q:
                            continue
                        ltp = getattr(q, "ltp", None) or (
                            q.get("ltp") if isinstance(q, dict) else None
                        )
                        if ltp:
                            local_quotes[sym] = {"ltp": float(ltp)}
                except Exception as exc:
                    logger.debug("breaker: quote batch failed: %s", exc)

            for p in open_rows:
                sym = p.get("symbol")
                qty = int(p.get("quantity") or 0)
                entry = float(p.get("entry_price") or 0)
                ltp = local_quotes.get(sym, {}).get("ltp")
                if qty and entry and ltp:
                    unrealized += (float(ltp) - entry) * qty
                capital_deployed += float(p.get("capital_deployed") or (entry * qty))
    except Exception as exc:
        logger.debug("breaker: open_rows lookup failed: %s", exc)

    if capital_deployed <= 0:
        # No positions today + no open positions = nothing to risk.
        return BreakerCheck(
            breached=False, current_pnl_pct=0.0, threshold_pct=threshold_signed,
            capital_deployed=0.0, realized_pnl=0.0, unrealized_pnl=0.0,
        )

    total_pnl = realized + unrealized
    current_pct = (total_pnl / capital_deployed) * 100
    breached = current_pct <= threshold_signed
    reason_text = (
        f"day_pnl={current_pct:.2f}% breached cap of {threshold_signed:.2f}% "
        f"(realized ₹{realized:.0f} + unrealized ₹{unrealized:.0f} "
        f"on ₹{capital_deployed:.0f} deployed)"
    )
    return BreakerCheck(
        breached=breached, current_pnl_pct=round(current_pct, 2),
        threshold_pct=threshold_signed,
        capital_deployed=round(capital_deployed, 2),
        realized_pnl=round(realized, 2),
        unrealized_pnl=round(unrealized, 2),
        reason_text=reason_text,
    )


def trip_strategy(
    supabase: Any,
    user_id: str,
    strategy_id: str,
    check: BreakerCheck,
) -> None:
    """Side-effects: pause the strategy + write a breaker-trip signal.

    Idempotent — already-paused strategies stay paused; the signal
    row uses an insert (not upsert) so repeated trips create an audit
    trail of "tried to fire entry but breaker was tripped".
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        supabase.table("user_strategies").update({
            "status": "paused",
            "paused_at": now,
            "pause_reason": "day_loss_breach",
        }).eq("id", strategy_id).eq("user_id", user_id).execute()
    except Exception as exc:
        logger.warning("breaker: strategy pause update failed: %s", exc)

    try:
        supabase.table("signals").insert({
            "user_id": user_id,
            "strategy_id": strategy_id,
            "symbol": "—",
            "source": "user_strategy",
            "signal_type": "swing",
            "action": "breach",
            "entry_price": 0,
            "confidence": 1.0,
            "status": "active",
            "market_context": {
                "kind": "day_loss_breaker_trip",
                "current_pnl_pct": check.current_pnl_pct,
                "threshold_pct": check.threshold_pct,
                "realized": check.realized_pnl,
                "unrealized": check.unrealized_pnl,
                "capital_deployed": check.capital_deployed,
                "tripped_at": now,
            },
        }).execute()
    except Exception as exc:
        logger.debug("breaker: trip signal insert failed: %s", exc)

    logger.warning(
        "day_loss_breaker TRIPPED user=%s strategy=%s: %s",
        user_id, strategy_id, check.reason_text,
    )
