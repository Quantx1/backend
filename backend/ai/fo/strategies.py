"""
F&O strategy recommender — F6 Elite.

Picks a weekly options strategy for index underliers (NIFTY,
BANKNIFTY, FINNIFTY) from two deterministic inputs:

    - VIX TFT forecast direction (rising / falling / stable)
    - HMM market regime                (bull / sideways / bear)

Maps each (regime × VIX direction) cell to 1-2 canonical strategies,
then prices every leg with Black-Scholes for theoretical premium +
Greeks and computes max-profit, max-loss, breakevens, and probability
of profit.

Kept deliberately small — no RL yet. The Options-PPO training is
deferred F6 work; this module is the rule layer that ships today.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from ...trading.fo.engine import BlackScholes, NSE_LOT_SIZES, OptionType


# Strike grid per underlying.
STRIKE_INTERVAL: Dict[str, int] = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
    "SENSEX": 100,
}


@dataclass
class StrategyLeg:
    action: str              # "BUY" or "SELL"
    option_type: str         # "CE" or "PE"
    strike: float
    expiry: str              # ISO date
    premium: float           # theoretical BS per unit
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float


@dataclass
class StrategyProposal:
    symbol: str
    strategy: str            # e.g. "iron_condor"
    name: str                # display name
    regime: str
    vix_direction: str
    vix_level: Optional[float]
    view: str                # one-line thesis
    legs: List[StrategyLeg] = field(default_factory=list)
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakevens: List[float] = field(default_factory=list)
    net_premium: float = 0.0     # + received, − paid per unit
    credit_debit: str = "debit"  # "credit" | "debit"
    lot_size: int = 1
    probability_of_profit: Optional[float] = None
    expiry: str = ""
    strike_interval: int = 50


# ---------------------------------------------------------------- helpers


def _years_to_expiry(expiry: date, today: Optional[date] = None) -> float:
    today = today or date.today()
    days = max((expiry - today).days, 1)
    return days / 365.25


def _atm_strike(spot: float, interval: int) -> float:
    return round(spot / interval) * interval


def _price_leg(
    *,
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r: float,
    option_type: OptionType,
    action: str,
    expiry: date,
) -> StrategyLeg:
    if option_type == OptionType.CALL:
        prem = BlackScholes.call_price(spot, strike, T, r, sigma)
    else:
        prem = BlackScholes.put_price(spot, strike, T, r, sigma)
    d = BlackScholes.delta(spot, strike, T, r, sigma, option_type)
    g = BlackScholes.gamma(spot, strike, T, r, sigma)
    t = BlackScholes.theta(spot, strike, T, r, sigma, option_type)
    v = BlackScholes.vega(spot, strike, T, r, sigma)
    return StrategyLeg(
        action=action,
        option_type=option_type.value,
        strike=float(strike),
        expiry=expiry.isoformat(),
        premium=round(prem, 2),
        delta=round(d, 3),
        gamma=round(g, 5),
        theta=round(t, 3),
        vega=round(v, 3),
        iv=round(sigma, 3),
    )


def _next_weekly_expiry(symbol: str, today: Optional[date] = None) -> date:
    """Thursday for NIFTY/FINNIFTY/SENSEX, Wednesday for BANKNIFTY.
    Stays simple — matches FOTradingEngine convention."""
    today = today or date.today()
    day_of_week = {
        "NIFTY": 3,  # Thu
        "FINNIFTY": 3,
        "SENSEX": 3,
        "BANKNIFTY": 2,  # Wed
        "MIDCPNIFTY": 2,
    }.get(symbol.upper(), 3)
    current = today
    while True:
        current += timedelta(days=1)
        if current.weekday() == day_of_week:
            return current


def _pop(breakevens: List[float], spot: float, sigma: float, T: float) -> float:
    """Rough probability of profit under log-normal terminal distribution.
    For 2-breakeven strategies (condor / strangle) integrates between.
    For 1-breakeven returns the spot-relative side-of-strike probability.
    """
    if sigma <= 0 or T <= 0 or not breakevens:
        return 0.5
    # log-normal terminal: ln(S_T / S_0) ~ N(-0.5 σ²T, σ²T) under risk-neutral
    mu = -0.5 * sigma * sigma * T
    sd = sigma * math.sqrt(T)

    def cdf(x: float) -> float:
        z = (math.log(x / spot) - mu) / sd
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    if len(breakevens) == 1:
        return round(1.0 - cdf(breakevens[0]), 3)
    lo, hi = sorted(breakevens[:2])
    # credit strategies (condor/strangle short) → S_T ∈ [lo, hi] is profit zone
    return round(max(0.0, min(1.0, cdf(hi) - cdf(lo))), 3)


# ---------------------------------------------------------------- strategies


def _bull_call_spread(*, symbol, spot, sigma, T, r, interval, expiry) -> StrategyProposal:
    atm = _atm_strike(spot, interval)
    long_k = atm
    short_k = atm + 2 * interval
    l1 = _price_leg(spot=spot, strike=long_k, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="BUY", expiry=expiry)
    l2 = _price_leg(spot=spot, strike=short_k, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="SELL", expiry=expiry)
    net = l2.premium - l1.premium                          # usually negative (debit)
    max_profit = (short_k - long_k) + net
    max_loss = -net                                      # premium paid
    be = long_k - net
    return StrategyProposal(
        symbol=symbol, strategy="bull_call_spread", name="Bull Call Spread",
        regime="bull", vix_direction="any", vix_level=None,
        view="Moderately bullish — capped upside, defined risk",
        legs=[l1, l2], max_profit=round(max_profit, 2), max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)], net_premium=round(net, 2),
        credit_debit="debit", lot_size=NSE_LOT_SIZES.get(symbol, 1),
        probability_of_profit=_pop([be], spot, sigma, T),
        expiry=expiry.isoformat(), strike_interval=interval,
    )


def _bear_put_spread(*, symbol, spot, sigma, T, r, interval, expiry) -> StrategyProposal:
    atm = _atm_strike(spot, interval)
    long_k = atm
    short_k = atm - 2 * interval
    l1 = _price_leg(spot=spot, strike=long_k, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="BUY", expiry=expiry)
    l2 = _price_leg(spot=spot, strike=short_k, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="SELL", expiry=expiry)
    net = l2.premium - l1.premium
    max_profit = (long_k - short_k) + net
    max_loss = -net
    be = long_k + net
    return StrategyProposal(
        symbol=symbol, strategy="bear_put_spread", name="Bear Put Spread",
        regime="bear", vix_direction="any", vix_level=None,
        view="Moderately bearish — capped downside target",
        legs=[l1, l2], max_profit=round(max_profit, 2), max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)], net_premium=round(net, 2),
        credit_debit="debit", lot_size=NSE_LOT_SIZES.get(symbol, 1),
        probability_of_profit=_pop([be], spot, sigma, T),
        expiry=expiry.isoformat(), strike_interval=interval,
    )


def _iron_condor(*, symbol, spot, sigma, T, r, interval, expiry) -> StrategyProposal:
    atm = _atm_strike(spot, interval)
    p_sell = atm - 2 * interval
    p_buy = atm - 4 * interval
    c_sell = atm + 2 * interval
    c_buy = atm + 4 * interval
    sp = _price_leg(spot=spot, strike=p_sell, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="SELL", expiry=expiry)
    bp = _price_leg(spot=spot, strike=p_buy, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="BUY", expiry=expiry)
    sc = _price_leg(spot=spot, strike=c_sell, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="SELL", expiry=expiry)
    bc = _price_leg(spot=spot, strike=c_buy, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="BUY", expiry=expiry)
    credit = sp.premium + sc.premium - bp.premium - bc.premium
    max_profit = credit
    max_loss = (c_buy - c_sell) - credit
    be_lo = p_sell - credit
    be_hi = c_sell + credit
    return StrategyProposal(
        symbol=symbol, strategy="iron_condor", name="Iron Condor",
        regime="sideways", vix_direction="falling", vix_level=None,
        view="Range-bound thesis — sell premium on both wings",
        legs=[sp, bp, sc, bc], max_profit=round(max_profit, 2),
        max_loss=round(max_loss, 2), breakevens=[round(be_lo, 2), round(be_hi, 2)],
        net_premium=round(credit, 2), credit_debit="credit",
        lot_size=NSE_LOT_SIZES.get(symbol, 1),
        probability_of_profit=_pop([be_lo, be_hi], spot, sigma, T),
        expiry=expiry.isoformat(), strike_interval=interval,
    )


def _long_straddle(*, symbol, spot, sigma, T, r, interval, expiry) -> StrategyProposal:
    atm = _atm_strike(spot, interval)
    lc = _price_leg(spot=spot, strike=atm, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="BUY", expiry=expiry)
    lp = _price_leg(spot=spot, strike=atm, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="BUY", expiry=expiry)
    debit = lc.premium + lp.premium
    be_lo = atm - debit
    be_hi = atm + debit
    return StrategyProposal(
        symbol=symbol, strategy="long_straddle", name="Long Straddle",
        regime="any", vix_direction="rising", vix_level=None,
        view="Big move coming either way — long volatility",
        legs=[lc, lp], max_profit=None, max_loss=round(debit, 2),
        breakevens=[round(be_lo, 2), round(be_hi, 2)],
        net_premium=round(-debit, 2), credit_debit="debit",
        lot_size=NSE_LOT_SIZES.get(symbol, 1),
        probability_of_profit=round(1.0 - _pop([be_lo, be_hi], spot, sigma, T), 3),
        expiry=expiry.isoformat(), strike_interval=interval,
    )


def _short_strangle(*, symbol, spot, sigma, T, r, interval, expiry) -> StrategyProposal:
    atm = _atm_strike(spot, interval)
    c_k = atm + 2 * interval
    p_k = atm - 2 * interval
    sc = _price_leg(spot=spot, strike=c_k, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="SELL", expiry=expiry)
    sp = _price_leg(spot=spot, strike=p_k, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="SELL", expiry=expiry)
    credit = sc.premium + sp.premium
    be_lo = p_k - credit
    be_hi = c_k + credit
    return StrategyProposal(
        symbol=symbol, strategy="short_strangle", name="Short Strangle",
        regime="sideways", vix_direction="falling", vix_level=None,
        view="IV crush — collect premium, undefined risk (margin-heavy)",
        legs=[sc, sp], max_profit=round(credit, 2), max_loss=None,
        breakevens=[round(be_lo, 2), round(be_hi, 2)],
        net_premium=round(credit, 2), credit_debit="credit",
        lot_size=NSE_LOT_SIZES.get(symbol, 1),
        probability_of_profit=_pop([be_lo, be_hi], spot, sigma, T),
        expiry=expiry.isoformat(), strike_interval=interval,
    )


def _iron_butterfly(*, symbol, spot, sigma, T, r, interval, expiry) -> StrategyProposal:
    atm = _atm_strike(spot, interval)
    sp = _price_leg(spot=spot, strike=atm, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="SELL", expiry=expiry)
    sc = _price_leg(spot=spot, strike=atm, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="SELL", expiry=expiry)
    bp = _price_leg(spot=spot, strike=atm - 2 * interval, T=T, sigma=sigma, r=r,
                    option_type=OptionType.PUT, action="BUY", expiry=expiry)
    bc = _price_leg(spot=spot, strike=atm + 2 * interval, T=T, sigma=sigma, r=r,
                    option_type=OptionType.CALL, action="BUY", expiry=expiry)
    credit = sp.premium + sc.premium - bp.premium - bc.premium
    max_profit = credit
    max_loss = (2 * interval) - credit
    be_lo = atm - credit
    be_hi = atm + credit
    return StrategyProposal(
        symbol=symbol, strategy="iron_butterfly", name="Iron Butterfly",
        regime="sideways", vix_direction="falling", vix_level=None,
        view="Pin-risk at ATM — maximum credit at expiry pin",
        legs=[sp, sc, bp, bc], max_profit=round(max_profit, 2),
        max_loss=round(max_loss, 2), breakevens=[round(be_lo, 2), round(be_hi, 2)],
        net_premium=round(credit, 2), credit_debit="credit",
        lot_size=NSE_LOT_SIZES.get(symbol, 1),
        probability_of_profit=_pop([be_lo, be_hi], spot, sigma, T),
        expiry=expiry.isoformat(), strike_interval=interval,
    )


# ---------------------------------------------------------------- selection


# (regime, vix_direction) → list of strategy builders, ranked best-first.
_MATRIX: Dict[tuple, List] = {
    ("bull", "falling"): [_bull_call_spread, _short_strangle],
    ("bull", "stable"): [_bull_call_spread, _iron_condor],
    ("bull", "rising"): [_bull_call_spread, _long_straddle],
    ("sideways", "falling"): [_iron_condor, _iron_butterfly],
    ("sideways", "stable"): [_iron_condor, _short_strangle],
    ("sideways", "rising"): [_long_straddle, _iron_condor],
    ("bear", "falling"): [_bear_put_spread, _iron_condor],
    ("bear", "stable"): [_bear_put_spread, _iron_butterfly],
    ("bear", "rising"): [_long_straddle, _bear_put_spread],
}


def recommend_strategies(
    *,
    symbol: str,
    spot: float,
    vix: float,
    vix_direction: str,
    regime: str,
    risk_free_rate: float = 0.07,
    expiry: Optional[date] = None,
    today: Optional[date] = None,
) -> List[StrategyProposal]:
    """Return a ranked list of 1-2 F&O strategies for the symbol.

    Parameters
    ----------
    vix : current India VIX value (0..100)
    vix_direction : 'rising' | 'falling' | 'stable' — from TFT forecast
    regime : 'bull' | 'sideways' | 'bear' — from HMM
    expiry : overrides weekly next-Thursday/Wednesday pick
    """
    symbol_u = symbol.upper()
    interval = STRIKE_INTERVAL.get(symbol_u, 50)
    expiry = expiry or _next_weekly_expiry(symbol_u, today)
    T = _years_to_expiry(expiry, today)
    # India VIX is annualised % IV — convert to decimal.
    sigma = max(0.05, (vix or 15.0) / 100.0)

    regime = (regime or "sideways").lower()
    vix_direction = (vix_direction or "stable").lower()
    builders = _MATRIX.get((regime, vix_direction), [_iron_condor])

    out: List[StrategyProposal] = []
    for b in builders:
        prop = b(
            symbol=symbol_u, spot=spot, sigma=sigma, T=T,
            r=risk_free_rate, interval=interval, expiry=expiry,
        )
        prop.regime = regime
        prop.vix_direction = vix_direction
        prop.vix_level = round(float(vix), 2) if vix is not None else None
        out.append(prop)
    return out


def price_strategy(
    strategy: str,
    *,
    symbol: str,
    spot: float,
    vix: float,
    expiry: Optional[date] = None,
    risk_free_rate: float = 0.07,
) -> Optional[StrategyProposal]:
    """Price a single named strategy on demand (no regime/VIX dependency)."""
    builders = {
        "iron_condor": _iron_condor,
        "bull_call_spread": _bull_call_spread,
        "bear_put_spread": _bear_put_spread,
        "long_straddle": _long_straddle,
        "short_strangle": _short_strangle,
        "iron_butterfly": _iron_butterfly,
    }
    b = builders.get(strategy)
    if b is None:
        return None
    symbol_u = symbol.upper()
    interval = STRIKE_INTERVAL.get(symbol_u, 50)
    exp = expiry or _next_weekly_expiry(symbol_u)
    T = _years_to_expiry(exp)
    sigma = max(0.05, (vix or 15.0) / 100.0)
    prop = b(
        symbol=symbol_u, spot=spot, sigma=sigma, T=T,
        r=risk_free_rate, interval=interval, expiry=exp,
    )
    prop.vix_level = round(float(vix), 2) if vix is not None else None
    return prop
