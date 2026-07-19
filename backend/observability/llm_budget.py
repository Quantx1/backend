"""Monthly LLM spend meter + hard kill-switch.

A fast in-process month-to-date counter so call-sites can refuse a PAID LLM
call the instant the monthly budget is exhausted — without a DB round-trip per
call. The counter = ``baseline`` (reconciled from ``llm_usage_events`` for the
current month, refreshed on a TTL) + ``pending`` (added in-process since the
last reconcile). The DB is the source of truth; the meter is a hot cache that
also survives restarts via the reconcile.

Free-tier models cost $0 (the price card returns 0) so they never move the
meter and are never blocked — only spend-bearing calls are gated.

NOTE: single-instance accurate. On a multi-instance deploy each process keeps
its own pending until the next reconcile, so the cap can briefly overshoot by
at most one TTL window of spend — acceptable for a solo-founder deployment and
self-corrects on reconcile.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when the monthly LLM budget is exhausted (paid calls only)."""


class UsageMeter:
    def __init__(self) -> None:
        self._baseline_micros = 0      # DB month-to-date at last reconcile
        self._pending_micros = 0       # added in-process since reconcile
        self._month_key: Optional[str] = None
        self._last_refresh = 0.0
        self._maybe_roll()

    # ── month handling ────────────────────────────────────────────────
    @staticmethod
    def _current_month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _maybe_roll(self) -> None:
        mk = self._current_month()
        if self._month_key != mk:
            self._month_key = mk
            self._baseline_micros = 0
            self._pending_micros = 0
            self._last_refresh = 0.0

    # ── counters ──────────────────────────────────────────────────────
    def record_micros(self, micros: int) -> None:
        """Add the cost of one call (micro-USD) to the in-process total."""
        self._maybe_roll()
        self._pending_micros += max(0, int(micros or 0))

    def spent_micros(self) -> int:
        self._maybe_roll()
        return self._baseline_micros + self._pending_micros

    def spent_usd(self) -> float:
        return self.spent_micros() / 1_000_000

    # ── budget checks ─────────────────────────────────────────────────
    def over_budget(self, budget_usd: float) -> bool:
        return self.spent_micros() >= budget_usd * 1_000_000

    def enforce(self, budget_usd: float) -> None:
        """Raise :class:`BudgetExceededError` if month-to-date spend ≥ budget."""
        if self.over_budget(budget_usd):
            raise BudgetExceededError(
                f"Monthly LLM budget ${budget_usd:.2f} exhausted "
                f"(spent ${self.spent_usd():.2f}). Paid model calls are paused "
                f"until next month or until the budget is raised.",
            )

    # ── DB reconcile ──────────────────────────────────────────────────
    def refresh_from_db(self, sb) -> None:
        """Reconcile the baseline from ``llm_usage_events`` for the current
        month. Best-effort — any failure keeps the prior cached value."""
        self._maybe_roll()
        if sb is None:
            return
        try:
            now = datetime.now(timezone.utc)
            month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()
            rows = (
                sb.table("llm_usage_events")
                .select("micros_usd")
                .gte("ts", month_start)
                .limit(200_000)
                .execute()
            )
            total = sum(int((r or {}).get("micros_usd") or 0) for r in (rows.data or []))
            self._baseline_micros = total
            self._pending_micros = 0
            self._last_refresh = time.monotonic()
        except Exception as exc:  # noqa: BLE001 — never let metering break a call
            logger.debug("usage meter reconcile failed: %s", exc)

    def maybe_refresh(self, sb, ttl_seconds: float = 60.0) -> None:
        """Reconcile from DB if the cache is older than ``ttl_seconds``."""
        if time.monotonic() - self._last_refresh > ttl_seconds:
            self.refresh_from_db(sb)


_meter: Optional[UsageMeter] = None


def get_meter() -> UsageMeter:
    global _meter
    if _meter is None:
        _meter = UsageMeter()
    return _meter


__all__ = ["UsageMeter", "BudgetExceededError", "get_meter"]
