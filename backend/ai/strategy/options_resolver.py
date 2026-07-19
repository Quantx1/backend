"""Resolve symbolic LegSpecs into concrete (strike, expiry) tuples — PR-J3.

The DSL stores legs declaratively (e.g. ``ATM+2``, ``current_week``) so a
template works at any spot level and any date. At entry time we need to
materialize each LegSpec into a concrete strike price + expiry date so
the BS pricer and the broker order builder can use them.

This module is pure-Python — no DB, no network, no clock except for an
optional ``today`` injection point used by tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

from .dsl import (
    ExpiryAnchor,
    LegSpec,
    OptionSide,
    OptionType,
    StrikeAnchor,
)


# Strike grid per NSE underlier — matches backend/ai/fo/strategies.py.
STRIKE_INTERVAL: Dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
    "SENSEX": 100,
}

# Weekly expiry weekday per symbol. NSE moved most weeklies to Thursday;
# BankNifty historically Wednesday. Matches our existing F&O engine.
_WEEKLY_DAY: Dict[str, int] = {
    "NIFTY": 3,  # Thursday
    "FINNIFTY": 3,
    "SENSEX": 3,
    "BANKNIFTY": 2,  # Wednesday
    "MIDCPNIFTY": 2,
}


@dataclass
class ResolvedLeg:
    """A LegSpec after strike/expiry resolution — what gets priced + ordered."""
    side: OptionSide
    option_type: OptionType
    strike: float
    expiry: date
    qty_lots: int
    # Carry the source LegSpec for debugging / audit trails
    source: LegSpec


def _atm_strike(spot: float, interval: int) -> float:
    """Round spot to the nearest strike interval."""
    return round(spot / interval) * interval


def _next_weekly_expiry(symbol: str, today: date) -> date:
    """Next weekly expiry day strictly after ``today``."""
    target_weekday = _WEEKLY_DAY.get(symbol.upper(), 3)
    cur = today + timedelta(days=1)
    while cur.weekday() != target_weekday:
        cur += timedelta(days=1)
    return cur


def _last_weekly_of_month(symbol: str, year: int, month: int) -> date:
    """Last weekly expiry of a given month — proxy for the monthly expiry."""
    target_weekday = _WEEKLY_DAY.get(symbol.upper(), 3)
    # Walk back from the 1st of next month
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    while end.weekday() != target_weekday:
        end -= timedelta(days=1)
    return end


def resolve_expiry(anchor: ExpiryAnchor, symbol: str, today: Optional[date] = None) -> date:
    """Resolve a symbolic expiry anchor into a concrete date.

    All dates returned are strictly in the future relative to ``today``.
    """
    today = today or date.today()
    symbol = symbol.upper()

    if anchor == ExpiryAnchor.CURRENT_WEEK:
        return _next_weekly_expiry(symbol, today)

    if anchor == ExpiryAnchor.NEXT_WEEK:
        first = _next_weekly_expiry(symbol, today)
        return _next_weekly_expiry(symbol, first)

    if anchor == ExpiryAnchor.CURRENT_MONTH:
        cm = _last_weekly_of_month(symbol, today.year, today.month)
        if cm <= today:
            # Already past — roll to next month
            ny, nm = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
            return _last_weekly_of_month(symbol, ny, nm)
        return cm

    if anchor == ExpiryAnchor.NEXT_MONTH:
        ny, nm = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return _last_weekly_of_month(symbol, ny, nm)

    raise ValueError(f"unhandled expiry anchor: {anchor}")


def resolve_strike(
    anchor: StrikeAnchor,
    offset: float,
    *,
    spot: float,
    symbol: str,
    sigma: Optional[float] = None,
    T_years: Optional[float] = None,
    option_type: Optional[OptionType] = None,
) -> float:
    """Resolve a symbolic strike anchor + offset into an absolute strike.

    For OTM_DELTA we need σ + T to invert the BS-delta → strike map;
    if not provided we fall back to a flat ATM±N approximation using
    a heuristic (5 delta ≈ 0.5σ√T moneyness).
    """
    interval = STRIKE_INTERVAL.get(symbol.upper(), 50)
    atm = _atm_strike(spot, interval)

    if anchor == StrikeAnchor.ATM:
        return atm
    if anchor == StrikeAnchor.ATM_PLUS_N:
        return atm + int(offset) * interval
    if anchor == StrikeAnchor.ATM_MINUS_N:
        return atm - int(offset) * interval
    if anchor == StrikeAnchor.PCT_OFFSET:
        raw = spot * (1.0 + offset / 100.0)
        return round(raw / interval) * interval
    if anchor == StrikeAnchor.OTM_DELTA:
        # Invert Black-Scholes delta to find the strike with |Δ| ≈ offset.
        # Under risk-neutral lognormal: |Δ_call(K)| = N(d1).
        # We need N(d1) ≈ offset → d1 = Φ⁻¹(offset).
        # Then K = S · exp(-d1 σ √T + 0.5 σ² T).
        if sigma is None or T_years is None or sigma <= 0 or T_years <= 0:
            # Fallback: treat offset as moneyness fraction
            raw = spot * (1.0 - offset)  # OTM call strike is above for buy puts; both sides handled at leg level
            return round(raw / interval) * interval
        d1 = _norm_ppf(offset)
        # Call OTM is K > S; we'll let the caller side+option_type decide direction.
        # Use the magnitude: K = S * exp(-d1 σ √T + 0.5 σ² T)
        K = spot * math.exp(-d1 * sigma * math.sqrt(T_years) + 0.5 * sigma * sigma * T_years)
        # For PE the symmetric strike below spot:
        if option_type == OptionType.PE:
            K = spot * spot / K  # reflect through ATM (log-normal symmetry approx)
        return round(K / interval) * interval

    raise ValueError(f"unhandled strike anchor: {anchor}")


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF — Beasley-Springer-Moro approximation.

    Vendored to avoid scipy dependency. Accurate to ~1e-7 for p in (0, 1).
    """
    # Beasley-Springer-Moro coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d_ = [7.784695709041462e-03, 3.224671290700398e-01,
          2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1)


def resolve_legs(
    legs: List[LegSpec],
    *,
    spot: float,
    symbol: str,
    today: Optional[date] = None,
    sigma: Optional[float] = None,
) -> List[ResolvedLeg]:
    """Resolve every LegSpec to a concrete ResolvedLeg using the same
    ``today`` for all of them (so a single position has a coherent expiry).
    """
    today = today or date.today()
    out: List[ResolvedLeg] = []
    for leg in legs:
        expiry = resolve_expiry(leg.expiry, symbol, today=today)
        T_years = max((expiry - today).days, 1) / 365.25
        strike = resolve_strike(
            leg.strike_anchor,
            leg.strike_offset,
            spot=spot,
            symbol=symbol,
            sigma=sigma,
            T_years=T_years,
            option_type=leg.option_type,
        )
        out.append(ResolvedLeg(
            side=leg.side,
            option_type=leg.option_type,
            strike=float(strike),
            expiry=expiry,
            qty_lots=leg.qty_lots,
            source=leg,
        ))
    return out


__all__ = [
    "ResolvedLeg",
    "STRIKE_INTERVAL",
    "resolve_legs",
    "resolve_strike",
    "resolve_expiry",
]
