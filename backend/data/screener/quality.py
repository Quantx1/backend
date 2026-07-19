"""Per-match quality scoring + ATR-derived levels for every scanner hit.

Adds three things to a scanner result row that traders actually use:
  1. ``quality_score`` (0-100) — composite of trend + momentum + volume
     + volatility + regime alignment for THIS specific match. Lets the
     UI rank obvious-best matches above noisy ones inside the same
     scanner output.
  2. ``entry``/``stop``/``target1``/``rr`` — ATR-derived trade levels
     so the trader doesn't have to compute risk by hand.
  3. ``triggers`` — short human-readable list of which conditions fired
     ("close > EMA200", "volume 2.4× SMA20", "RSI(14)=58 building").

Applied in ``LiveScreenerEngine._format_for_frontend`` so every legacy
scanner output gets the same enrichment — no per-scanner work needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ── Composite quality score ──────────────────────────────────────────


def compute_quality_score(row: pd.Series) -> float:
    """Composite 0-100 quality score for one scanner match.

    Components (weights sum to 100):
       Trend alignment           20  — close vs EMA21/SMA50/EMA200
       Momentum                  20  — RSI sweet spot + MACD agreement
       Volume confirmation       20  — volume_ratio + volume_pocket flag
       Volatility setup          15  — ATR percentile (low ATR + volume = ready)
       Recent strength            15  — distance from 52w high (closer = stronger)
       Risk/range health          10  — ATR/close % (not stretched)

    Returns a single float — higher is better. Capped to [0, 100].
    """
    score = 0.0

    close = _f(row.get("close"))
    if close <= 0:
        return 0.0

    # 1. TREND (20 pts) — above key MAs
    ema_21 = _f(row.get("ema_21"))
    sma_50 = _f(row.get("sma_50"))
    ema_200 = _f(row.get("ema_200"))
    sma_200 = _f(row.get("sma_200")) or ema_200
    if ema_21 and close > ema_21:
        score += 5
    if sma_50 and close > sma_50:
        score += 5
    if sma_200 and close > sma_200:
        score += 5
    if ema_21 and sma_50 and ema_21 > sma_50:    # stacked
        score += 5

    # 2. MOMENTUM (20 pts)
    rsi = _f(row.get("rsi_14"))
    macd = _f(row.get("macd"))
    macd_signal = _f(row.get("macd_signal"))
    macd_hist = _f(row.get("macd_hist"))
    # RSI sweet spot 45-65 (building momentum) > 65 (overbought)
    if 45 <= rsi <= 65:
        score += 8
    elif 30 <= rsi < 45 or 65 < rsi <= 75:
        score += 5
    if macd > macd_signal:
        score += 6
    if macd_hist > 0:
        score += 6

    # 3. VOLUME (20 pts)
    vol_ratio = _f(row.get("volume_ratio"), 1.0)
    if vol_ratio >= 2.0:
        score += 20
    elif vol_ratio >= 1.5:
        score += 15
    elif vol_ratio >= 1.2:
        score += 10
    elif vol_ratio >= 1.0:
        score += 5

    # 4. VOLATILITY setup (15 pts) — low ATR % = coiled
    atr = _f(row.get("atr_14"))
    if atr and close > 0:
        atr_pct = atr / close
        if 0.005 <= atr_pct <= 0.020:
            score += 15
        elif 0.020 < atr_pct <= 0.035:
            score += 10
        elif atr_pct < 0.005:
            score += 5     # too quiet

    # 5. RECENT STRENGTH (15 pts) — close to 52w high
    hi_52w = _f(row.get("high_52w"))
    if hi_52w and close > 0:
        dist = (close / hi_52w) * 100
        if dist >= 95:
            score += 15
        elif dist >= 85:
            score += 10
        elif dist >= 70:
            score += 5

    # 6. RANGE HEALTH (10 pts) — not stretched 3 ATR above MA21
    if atr and ema_21 and close - ema_21 < 3 * atr:
        score += 10

    # ADX trend bonus / regime alignment — ADX 25+ adds 5 (but caps at 100)
    adx = _f(row.get("adx"))
    if adx >= 25:
        score = min(100.0, score + 5)

    return round(min(100.0, max(0.0, score)), 1)


# ── ATR-derived levels ───────────────────────────────────────────────


def suggest_levels(row: pd.Series,
                   atr_mult_stop: float = 1.8,
                   atr_mult_target: float = 3.5) -> Dict[str, Any]:
    """Entry / stop / target derived from close + 14-bar ATR.

    Slightly tighter stop than the textbook 2 ATR — Indian intraday
    spreads + STT make 2 ATR stops trigger on noise. 1.8 × ATR balances
    survivability with reasonable R:R for swing.
    """
    close = _f(row.get("close"))
    atr = _f(row.get("atr_14"))
    if close <= 0 or atr <= 0:
        return {}

    entry = round(close, 2)
    stop = round(close - atr_mult_stop * atr, 2)
    target1 = round(close + atr_mult_target * atr, 2)
    target2 = round(close + 1.5 * atr_mult_target * atr, 2)
    risk = abs(entry - stop)
    reward = abs(target1 - entry)
    rr = round(reward / risk, 2) if risk > 1e-6 else 0.0

    return {
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr": rr,
        "stop_basis": f"-{atr_mult_stop}×ATR",
        "target_basis": f"+{atr_mult_target}×ATR",
    }


# ── Human-readable trigger list ──────────────────────────────────────


def compute_triggers(row: pd.Series) -> List[str]:
    """Up-to-5 short reasons why this row is interesting. Used as a
    tooltip + as the 'why this matched' summary on the scanner card."""
    out: List[str] = []
    close = _f(row.get("close"))
    rsi = _f(row.get("rsi_14"))
    vol_ratio = _f(row.get("volume_ratio"), 1.0)
    _f(row.get("ema_21"))
    sma_50 = _f(row.get("sma_50"))
    ema_200 = _f(row.get("ema_200"))
    macd = _f(row.get("macd"))
    macd_signal = _f(row.get("macd_signal"))
    adx = _f(row.get("adx"))
    hi_52w = _f(row.get("high_52w"))
    atr = _f(row.get("atr_14"))

    # 1. Trend chain
    if close and ema_200 and close > ema_200:
        out.append("Above EMA200")
    if close and sma_50 and close > sma_50:
        out.append("Above SMA50")

    # 2. Volume
    if vol_ratio >= 2.0:
        out.append(f"Volume {vol_ratio:.1f}× avg")
    elif vol_ratio >= 1.5:
        out.append(f"Vol confirm {vol_ratio:.1f}×")

    # 3. Momentum
    if macd > macd_signal:
        out.append("MACD bullish cross")
    if 45 <= rsi <= 65:
        out.append(f"RSI {rsi:.0f} (building)")
    elif rsi < 30:
        out.append(f"RSI {rsi:.0f} oversold")

    # 4. 52w proximity
    if hi_52w and close > 0:
        dist = (close / hi_52w) * 100
        if dist >= 95:
            out.append("Near 52w high")
        elif dist >= 85:
            out.append(f"{dist:.0f}% of 52w high")

    # 5. Trend strength
    if adx >= 25:
        out.append(f"ADX {adx:.0f} (trending)")

    # 6. Tight range = coiled
    if atr and close > 0 and atr / close < 0.015:
        out.append("Tight range (coiled)")

    return out[:5]


def _f(v, default: float = 0.0) -> float:
    """Safe float coercion — handles None/NaN/string."""
    try:
        if v is None:
            return default
        f = float(v)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# ── Enrich one scanner result dict ───────────────────────────────────


def enrich_row(result: Dict[str, Any], row: pd.Series) -> Dict[str, Any]:
    """Add quality_score + levels + triggers in place. Used by
    ``LiveScreenerEngine._format_for_frontend`` on every result row."""
    result["quality_score"] = compute_quality_score(row)
    result["triggers"] = compute_triggers(row)
    levels = suggest_levels(row)
    if levels:
        # Don't overwrite if the scanner already set its own (e.g. patterns)
        result.setdefault("entry", levels["entry"])
        result.setdefault("stop_loss", levels["stop"])
        result.setdefault("target", levels["target1"])
        result.setdefault("target_2", levels.get("target2"))
        result.setdefault("rr", levels["rr"])
        result.setdefault("stop_basis", levels["stop_basis"])
        result.setdefault("target_basis", levels["target_basis"])
    return result
