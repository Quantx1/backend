"""Premium-confirmation gate — PR-DEPTH (the "falling knife" check).

Adapted from aaryansinha16/AI-trader's backend/app.py lines 1126-1167.

Before allowing a strategy entry, query the last N seconds/minutes of
tick data for the symbol. If the price slope is sharply negative,
REJECT — we'd be catching a falling knife.

Concrete example from aaryansinha's repo (commit comment 2026-04-08):
  24000PE premium fell 233.9 → 232.7 in ~25 seconds before entry.
  Without the gate: manual entry lost ₹7,031. With the gate: skipped.

For our app:
  - Options strategies: check the option contract's recent tick slope
    (when tick_data has rows for that contract)
  - Equity strategies: check the underlying stock's recent slope on
    daily/intraday bars (fallback when no tick data)
  - No data either way: fail OPEN (don't reject — gate just doesn't
    apply yet, e.g. before tick collector has been running long enough)

Fail-open semantics: missing data ≠ block. The gate only blocks when
we have strong evidence of negative slope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Default: reject if premium fell >0.8% in last 30 seconds (matches aaryansinha)
DEFAULT_SLOPE_THRESHOLD_PCT = -0.008
DEFAULT_WINDOW_SECONDS = 30
DEFAULT_MIN_TICKS = 5         # need at least 5 ticks to trust the slope


@dataclass
class PremiumGateResult:
    allowed: bool
    slope_pct: Optional[float] = None
    ticks_seen: int = 0
    block_reason: Optional[str] = None
    note: Optional[str] = None


def check_premium_slope(
    supabase: Any,
    *,
    symbol: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    slope_threshold_pct: float = DEFAULT_SLOPE_THRESHOLD_PCT,
    min_ticks: int = DEFAULT_MIN_TICKS,
    now: Optional[datetime] = None,
) -> PremiumGateResult:
    """Check if the symbol's recent tick slope is too negative to enter.

    Returns ``PremiumGateResult(allowed=False)`` only when:
      - we have ≥ min_ticks ticks in the window
      - slope_pct < slope_threshold_pct (default -0.8%)

    Otherwise allows. Never raises.
    """
    if supabase is None:
        return PremiumGateResult(allowed=True, note="no_supabase_client")

    now = now or datetime.utcnow()
    window_start = now - timedelta(seconds=window_seconds)

    try:
        rows = (
            supabase.table("tick_data")
            .select("timestamp, price")
            .eq("symbol", symbol)
            .gte("timestamp", window_start.isoformat())
            .order("timestamp")
            .limit(500)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("premium_gate query failed: %s", exc)
        return PremiumGateResult(allowed=True, note="query_failed")

    data = rows.data or []
    n = len(data)
    if n < min_ticks:
        return PremiumGateResult(
            allowed=True, ticks_seen=n,
            note=f"insufficient_ticks ({n}/{min_ticks}) — fail-open",
        )

    first_price = float(data[0].get("price") or 0)
    last_price = float(data[-1].get("price") or 0)
    if first_price <= 0 or last_price <= 0:
        return PremiumGateResult(allowed=True, ticks_seen=n,
                                 note="bad_prices — fail-open")

    slope_pct = (last_price - first_price) / first_price

    if slope_pct < slope_threshold_pct:
        return PremiumGateResult(
            allowed=False,
            slope_pct=slope_pct,
            ticks_seen=n,
            block_reason=f"premium_slope:{slope_pct:+.4f}",
            note=f"price fell {slope_pct * 100:+.2f}% in last {window_seconds}s "
            f"({first_price:.2f}→{last_price:.2f}, {n} ticks)",
        )

    return PremiumGateResult(
        allowed=True,
        slope_pct=slope_pct,
        ticks_seen=n,
        note=f"slope {slope_pct * 100:+.2f}% acceptable",
    )
