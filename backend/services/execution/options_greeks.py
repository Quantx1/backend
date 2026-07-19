"""Implied volatility + Greeks computation for the option chain (PR-AY).

Brokers don't return IV directly (Kite chain ships ``iv=0``).
Computing it client-side adds the missing data point traders need:
  - IV per strike (vol smile / skew visible)
  - Delta / Gamma / Theta / Vega per strike for risk display

Solver:
  Brent's method for IV — robust to flat-bottom regions where price
  is intrinsic-only. Newton-Raphson is faster but needs a derivative
  bracket; we stay with bisection-style for simplicity since the
  chain has at most ~80 strikes (sub-100ms full chain).

All Greeks come from analytic BS formulas (no scipy dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Match the constants in options_backtest.py
RISK_FREE_RATE = 0.065


def _ncdf(x: float) -> float:
    """Standard normal CDF — math.erf-based, no scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(
    S: float, K: float, T: float, r: float, sigma: float, *, is_call: bool,
) -> float:
    """Black-Scholes mid price. Returns intrinsic at expiry."""
    if T <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float   # per CALENDAR day (already divided by 365)
    vega: float    # per 1 vol-point (so 0.01 sigma move = vega rupees)
    iv: float      # annualised implied vol


def implied_volatility(
    market_price: float,
    *,
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    is_call: bool,
    lo: float = 0.001,
    hi: float = 5.0,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> Optional[float]:
    """Solve for σ such that BS(S, K, T, r, σ) == market_price.

    Brent-style bisection on a bracketed [lo, hi]. Returns None when:
      - price < intrinsic (arbitrage / stale quote)
      - price > S (impossible call) or > K*e^-rT (impossible put)
      - T == 0 (no time value to imply)
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    upper_bound = S if is_call else K * math.exp(-r * T)
    if market_price < intrinsic - tol or market_price > upper_bound + tol:
        return None

    def price_at(sigma: float) -> float:
        return _bs_price(S, K, T, r, sigma, is_call=is_call) - market_price

    f_lo = price_at(lo)
    f_hi = price_at(hi)
    if f_lo > 0 and f_hi > 0:
        return None  # market price below floor — flat OTM rate region
    if f_lo < 0 and f_hi < 0:
        return None  # market price above ceiling — shouldn't happen

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = price_at(mid)
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


def compute_greeks(
    *,
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    sigma: float,
    is_call: bool,
) -> Greeks:
    """Analytic Black-Scholes Greeks for one option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return Greeks(
            delta=1.0 if (is_call and S > K) else
            -1.0 if (not is_call and S < K) else 0.0,
            gamma=0.0, theta=0.0, vega=0.0, iv=sigma,
        )
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    delta = _ncdf(d1) if is_call else _ncdf(d1) - 1.0
    gamma = _npdf(d1) / (S * sigma * sqrt_T)
    # Theta in rupees per year → divide by 365 for per-calendar-day
    theta_annual = (
        - (S * _npdf(d1) * sigma) / (2 * sqrt_T)
        - (r * K * math.exp(-r * T) * (_ncdf(d2) if is_call else -_ncdf(-d2)))
    )
    if not is_call:
        # Correct sign for puts: +rK e^-rT N(-d2)
        theta_annual = (
            - (S * _npdf(d1) * sigma) / (2 * sqrt_T)
            + (r * K * math.exp(-r * T) * _ncdf(-d2))
        )
    theta_per_day = theta_annual / 365.0
    # Vega in rupees per 100% sigma → /100 for per 1 vol-point (1%).
    vega_per_point = S * _npdf(d1) * sqrt_T / 100.0

    return Greeks(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta_per_day, 4),
        vega=round(vega_per_point, 4),
        iv=round(sigma, 4),
    )


def enrich_chain_row(
    *,
    spot: float,
    strike: float,
    expiry_days: int,
    ltp: float,
    option_type: str,
    r: float = RISK_FREE_RATE,
) -> Optional[Greeks]:
    """Compute IV + Greeks for one chain row. Returns None when IV
    can't be solved (intrinsic-only, expired, malformed)."""
    if expiry_days <= 0 or ltp <= 0 or spot <= 0 or strike <= 0:
        return None
    T = expiry_days / 365.0
    is_call = option_type.upper() == "CE"
    iv = implied_volatility(ltp, S=spot, K=strike, T=T, r=r, is_call=is_call)
    if iv is None or iv <= 0:
        return None
    return compute_greeks(S=spot, K=strike, T=T, r=r, sigma=iv, is_call=is_call)
