"""Stagnation-aware trailing-stop boost — PR-DEPTH.

Adapted from aaryansinha16/AI-trader's _ratchet_trailing logic (2026-04-16).

Pattern: when the peak premium hasn't advanced in N bars AND we have
meaningful profit (≥15%), tighten the SL aggressively. Rationale: options
bleed theta each minute. If the trade isn't advancing, it's losing to
time decay. Equity strategies have a similar logic — if a position
stops making new highs, the lower-tail risk grows.

This is a SMALL addition layered on top of the standard retention tiers.
Pure function — caller maintains state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StagnationTrailingState:
    """Caller maintains this — passes it into update_stagnation_trailing each tick."""
    peak_price: float
    peak_bar_idx: int                       # bar index when peak was last set
    current_sl: float


def update_stagnation_trailing(
    state: StagnationTrailingState,
    *,
    entry_price: float,
    current_bar_idx: int,
    base_retention: float,
) -> float:
    """Return updated retention% to apply (NOT a new SL; caller computes SL).

    Boost rules:
      0-5 bars flat        → no boost (retention = base)
      5-10 bars flat       → +6% retention (gentle tighten)
      10-20 bars flat      → +12% retention (tighten hard)
      20+ bars flat        → +20% retention (near-peak lock)

    Only applies when:
      - peak gain ≥ 15% (meaningful profit to protect)
      - we have a peak_bar_idx (caller tracks it)

    Returns the effective retention (capped at 0.90).
    """
    if entry_price <= 0:
        return base_retention

    gain_pct = (state.peak_price - entry_price) / entry_price
    if gain_pct < 0.15:
        return base_retention

    bars_since_peak = current_bar_idx - state.peak_bar_idx
    if bars_since_peak < 5:
        return base_retention

    if bars_since_peak >= 20:
        boost = 0.20
    elif bars_since_peak >= 10:
        boost = 0.12
    else:
        boost = 0.06

    boosted = base_retention + boost
    # Cap at 0.90 to avoid setting SL effectively at the peak
    return min(0.90, boosted)


def compute_stagnation_aware_sl(
    *,
    entry_price: float,
    peak_price: float,
    peak_bar_idx: int,
    current_bar_idx: int,
    current_sl: float,
    base_retention: float = 0.50,
) -> float:
    """Convenience: compute the new SL after applying stagnation-aware
    retention. Returns max(current_sl, new_sl) — never moves SL down."""
    if entry_price <= 0 or peak_price <= entry_price:
        return current_sl

    state = StagnationTrailingState(
        peak_price=peak_price,
        peak_bar_idx=peak_bar_idx,
        current_sl=current_sl,
    )
    retention = update_stagnation_trailing(
        state,
        entry_price=entry_price,
        current_bar_idx=current_bar_idx,
        base_retention=base_retention,
    )
    gain_from_entry = peak_price - entry_price
    new_sl = entry_price + retention * gain_from_entry
    return max(current_sl, new_sl)
