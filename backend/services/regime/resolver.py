"""Regime resolver — pure-Python, single source of truth.

All callers (DSL backtest, signal generator, supervisor, copilot tool)
go through these helpers so the fallback policy is identical everywhere.

Fallback chain for a single date:
  1. exact-date or same-day row in ``regime_history``
  2. last known row ≤ that date
  3. ``DEFAULT_REGIME`` (= ``"sideways"``)

The default of ``sideways`` is deliberate — it does NOT satisfy
``bull_only`` or ``bear_only`` regime_filter gates AND does NOT get
blocked by a ``sideways_only`` gate. So a strategy authored with
``regime_filter="bull_only"`` correctly stays out of the market on a
day where regime is unknown, rather than silently trading without
the gate enforced.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_REGIME: str = "sideways"
_ALLOWED_REGIMES = {"bull", "sideways", "bear"}


def _coerce_date(d: Any) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        try:
            return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(d[:10])
            except ValueError:
                return None
    return None


def _normalize_regime(value: Any) -> str:
    if not isinstance(value, str):
        return DEFAULT_REGIME
    v = value.strip().lower()
    return v if v in _ALLOWED_REGIMES else DEFAULT_REGIME


def resolve_regime_at(
    supabase: Any,
    *,
    at: Optional[date] = None,
) -> str:
    """Return the regime as-of ``at`` (or today). Pure single-value lookup.

    Best-effort — any Supabase exception falls through to ``DEFAULT_REGIME``.
    Never raises.
    """
    at = at or date.today()
    if supabase is None:
        return DEFAULT_REGIME
    try:
        rows = (
            supabase.table("regime_history")
            .select("regime, detected_at")
            .lte("detected_at", at.isoformat() + "T23:59:59")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("regime resolver query failed: %s", exc)
        return DEFAULT_REGIME
    data = rows.data or []
    if not data:
        return DEFAULT_REGIME
    return _normalize_regime(data[0].get("regime"))


def _query_regime_rows(supabase: Any, start: date, end: date) -> List[Tuple[date, str]]:
    """Fetch real ``regime_history`` rows in [month-of-start, end], sorted.
    Returns [] on any error or missing client (callers then fall back)."""
    if supabase is None:
        return []
    try:
        rows = (
            supabase.table("regime_history")
            .select("regime, detected_at")
            # One month of headroom so the boundary carry-forward uses a real
            # prior value where possible.
            .gte("detected_at", (start.replace(day=1)).isoformat())
            .lte("detected_at", end.isoformat() + "T23:59:59")
            .order("detected_at")
            .limit(2000)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("regime history bulk query failed: %s", exc)
        return []
    raw: List[Tuple[date, str]] = []
    for row in rows.data or []:
        d = _coerce_date(row.get("detected_at"))
        if d is not None:
            raw.append((d, _normalize_regime(row.get("regime"))))
    return raw


def resolve_regime_history(
    supabase: Any,
    *,
    start: date,
    end: date,
) -> Dict[date, str]:
    """Bulk regime lookup over [start, end]. Returns ``{date: regime}``
    with the carry-forward rule applied (gaps inherit the last known
    value, no missing keys).

    Used by the DSL backtest to inject regime per bar so strategies with
    ``regime_filter`` or ``engine_signal: Regime == ...`` evaluate against
    the historically-correct value at each bar instead of always-None.
    """
    if start > end:
        return {}
    return _fill_carry_forward(start, end, _query_regime_rows(supabase, start, end))


def resolve_regime_history_with_coverage(
    supabase: Any,
    *,
    start: date,
    end: date,
) -> Tuple[Dict[date, str], float]:
    """Like :func:`resolve_regime_history` but also returns *coverage* — the
    fraction of days in [start, end] that mapped to a REAL detected regime
    (a real row at/before that day) rather than the pre-history ``sideways``
    default.

    Coverage < 1.0 means part of the window predates ``regime_history`` and is
    running on the default — so a regime-gated backtest over that window is
    NOT trustworthy. The gate uses this to fail-closed (see evaluation.py).
    """
    if start > end:
        return {}, 0.0
    return _fill_carry_forward(start, end, _query_regime_rows(supabase, start, end), with_coverage=True)


def _fill_carry_forward(
    start: date,
    end: date,
    rows: List[Tuple[date, str]],
    *,
    with_coverage: bool = False,
):
    """Build a date→regime map covering every day in [start, end] using
    carry-forward (gaps inherit the previous known value). Days before any
    real row use ``DEFAULT_REGIME`` and count as NOT real for coverage.

    Returns the map; or ``(map, coverage)`` when ``with_coverage=True``.
    """
    by_date: Dict[date, str] = {d: r for d, r in rows}
    out: Dict[date, str] = {}
    cursor = DEFAULT_REGIME
    cursor_is_real = False
    # If we have prior rows, the regime at `start` carries from the last one.
    prior = [(d, r) for d, r in rows if d < start]
    if prior:
        cursor = prior[-1][1]
        cursor_is_real = True

    one_day = timedelta(days=1)
    real_days = 0
    total = 0
    current = start
    while current <= end:
        if current in by_date:
            cursor = by_date[current]
            cursor_is_real = True
        out[current] = cursor
        total += 1
        if cursor_is_real:
            real_days += 1
        current = current + one_day

    if with_coverage:
        return out, (real_days / total if total else 0.0)
    return out
