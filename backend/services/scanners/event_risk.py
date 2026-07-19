"""
Event-risk blackout — the "don't get killed by events" gate.

Suppresses NEW entries (opens / add-ons) in symbols with an imminent
earnings announcement, and flags index-expiry days. Deliberately
ENTRY-ONLY: it never forces an exit of an existing holding (that is a
separate exit policy) — it only stops the bot from buying INTO a known
binary event.

Deterministic, DB-first (batched single query over earnings_predictions),
honest-empty when the source is unavailable. No LLM, no per-symbol network
calls in the hot path.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Iterable, Optional, Set

logger = logging.getLogger(__name__)

# Default blackout: skip entering a name within N calendar days of earnings.
EVENT_BLACKOUT_DAYS = int(os.getenv("EVENT_RISK_BLACKOUT_DAYS", "2"))


def _sb(supabase=None):
    if supabase is not None:
        return supabase
    from ...core.database import get_supabase_admin
    return get_supabase_admin()


def symbols_in_event_window(
    symbols: Iterable[str],
    *,
    days: int = EVENT_BLACKOUT_DAYS,
    supabase=None,
    today: Optional[date] = None,
) -> Set[str]:
    """Subset of ``symbols`` with an earnings announce_date within ``days``.

    One batched query over ``earnings_predictions``. Returns an empty set on
    any failure (honest-empty — the gate fails OPEN so a data outage never
    blocks all trading; the hard SL/target rails still apply downstream).
    """
    syms = sorted({str(s).strip().upper() for s in symbols if s})
    if not syms or days <= 0:
        return set()
    start = today or date.today()
    end = start + timedelta(days=days)
    try:
        rows = (
            _sb(supabase)
            .table("earnings_predictions")
            .select("symbol, announce_date")
            .in_("symbol", syms)
            .gte("announce_date", start.isoformat())
            .lte("announce_date", end.isoformat())
            .limit(500)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("event_risk: earnings_predictions read failed: %s", exc)
        return set()
    return {str(r.get("symbol") or "").strip().upper() for r in rows if r.get("symbol")}


def is_expiry_day(d: Optional[date] = None) -> bool:
    """True on the weekly index-expiry day (Thursday, or Wednesday when
    Thursday is an NSE holiday). Heuristic — used to flag, not hard-block."""
    d = d or date.today()
    # Thursday == weekday 3. If Thursday is a holiday, expiry shifts to Wed.
    if d.weekday() == 3:
        return True
    if d.weekday() == 2:  # Wednesday — check if tomorrow (Thu) is a holiday
        try:
            from ...data.market_calendar import is_trading_day
            return not is_trading_day(d + timedelta(days=1))
        except Exception:
            return False
    return False


def filter_entry_weights(
    weights: dict,
    *,
    days: int = EVENT_BLACKOUT_DAYS,
    supabase=None,
) -> tuple[dict, Set[str]]:
    """Return (kept_weights, blacked_out_symbols) for a target-weight map.

    Used by AutoPilot to decide which target symbols are eligible to be
    OPENED/ADDED today. Symbols in the event window are dropped from the
    *eligible-to-enter* set; the caller must still allow reductions of any
    existing holding in those names (handled in the emit diff).
    """
    if not weights:
        return weights, set()
    blocked = symbols_in_event_window(weights.keys(), days=days, supabase=supabase)
    if not blocked:
        return weights, set()
    kept = {s: w for s, w in weights.items() if s.upper() not in blocked}
    return kept, blocked


__all__ = [
    "EVENT_BLACKOUT_DAYS",
    "symbols_in_event_window",
    "is_expiry_day",
    "filter_entry_weights",
]
