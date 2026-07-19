"""
Market overview helpers — VIX-driven risk bands + Nifty trend.

Single source of truth for market-condition classification. Two
call sites consumed this logic with different cutoffs before:

  * scheduler.market_open_check() wrote ``market_data.risk_level``
    using 3 bands (LOW/MODERATE/HIGH at >20/>25).
  * /api/market/risk computed its own risk_level using 4 bands
    (LOW/MODERATE/HIGH/EXTREME at <15/<20/<25/>=25).

Same VIX → different displayed band depending on which surface the
user looked at. The 4-band version is authoritative; ``EXTREME`` is
the only band that flips the broadcast circuit-breaker copy on
``/api/market/risk``, so the scheduler had to pick that up too.

NSE-calibrated cutoffs:
  India VIX sits at 12-15 on quiet days, 16-20 normal, 20-25 elevated,
  25+ stressed (Covid March 2020 hit ~70). The bands match the
  position-sizing playbook in the v1 Step 2 risk doc.
"""

from __future__ import annotations

from typing import TypedDict


class MarketCondition(TypedDict):
    trend: str          # BULLISH | BEARISH | SIDEWAYS
    risk_level: str     # LOW | MODERATE | HIGH | EXTREME
    recommendation: str


def classify_risk_band(vix: float) -> str:
    """Map current India VIX to a 4-band risk classifier."""
    if vix < 15:
        return "LOW"
    if vix < 20:
        return "MODERATE"
    if vix < 25:
        return "HIGH"
    return "EXTREME"


def classify_trend(nifty_change_pct: float) -> str:
    """Map daily Nifty % change to a trend tag.

    The ±0.5% deadband is intentional — anything inside it is noise
    around flat opens / mean reversion. The signal pipeline reads
    ``trend`` to weight LONG vs SHORT setups, so a tighter band would
    flip-flop on intraday consolidation.
    """
    if nifty_change_pct > 0.5:
        return "BULLISH"
    if nifty_change_pct < -0.5:
        return "BEARISH"
    return "SIDEWAYS"


def risk_recommendation(band: str) -> str:
    """User-facing copy for each risk band.

    Position-sizing language matches what the dashboard widget shows
    so the scheduler's stored ``recommendation`` reads identically to
    the on-demand ``/api/market/risk`` response.
    """
    if band == "LOW":
        return "Normal trading - full position sizes"
    if band == "MODERATE":
        return "Reduce position sizes by 25%"
    if band == "HIGH":
        return "Reduce position sizes by 50%, only high-confidence trades"
    if band == "EXTREME":
        return "Stop all new trades, consider hedging"
    return "Unknown risk level"


def determine_market_condition(vix: float, nifty_change_pct: float) -> MarketCondition:
    """Compose trend + risk_level + recommendation from raw inputs."""
    band = classify_risk_band(vix)
    return {
        "trend": classify_trend(nifty_change_pct),
        "risk_level": band,
        "recommendation": risk_recommendation(band),
    }
