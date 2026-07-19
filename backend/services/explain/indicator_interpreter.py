"""AI Indicator Interpreter (#3) — indicator values -> plain English.

Deterministic per-indicator reads (RSI / MACD / ADX+DI / trend vs 200-DMA /
volume) — 0 tokens, can't hallucinate — plus an overall bias and an optional
grounded one-line synthesis. `interpret` + `bias` are pure (tested);
`interpret_symbol` computes the indicators via the shared indicator stack.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_BULL = {"bullish", "oversold"}
_BEAR = {"bearish", "overbought"}


def interpret(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-indicator plain-English reads from a metrics dict."""
    out: List[Dict[str, Any]] = []

    rsi = m.get("rsi_14")
    if rsi is not None:
        if rsi >= 70:
            sig, txt = "overbought", "above 70 — strong momentum, but short-term pullbacks become more likely."
        elif rsi <= 30:
            sig, txt = "oversold", "below 30 — oversold; a bounce becomes more likely."
        elif rsi >= 55:
            sig, txt = "bullish", "in the 55-70 zone — healthy bullish momentum."
        elif rsi <= 45:
            sig, txt = "bearish", "below 45 — momentum is weak."
        else:
            sig, txt = "neutral", "near 50 — momentum is neutral."
        out.append({"indicator": "RSI(14)", "value": round(rsi, 1), "signal": sig, "read": f"RSI is {txt}"})

    hist = m.get("macd_hist")
    if hist is not None:
        sig = "bullish" if hist > 0 else "bearish"
        side = "above" if hist > 0 else "below"
        out.append({"indicator": "MACD", "value": round(hist, 2), "signal": sig,
                    "read": f"MACD is {side} its signal line — {sig} momentum."})

    adx = m.get("adx")
    if adx is not None:
        if adx >= 25:
            up = (m.get("di_plus") or 0) >= (m.get("di_minus") or 0)
            sig = "bullish" if up else "bearish"
            out.append({"indicator": "ADX", "value": round(adx, 1), "signal": sig,
                        "read": f"ADX {adx:.0f} — a strong {'up' if up else 'down'}trend is in force."})
        else:
            out.append({"indicator": "ADX", "value": round(adx, 1), "signal": "neutral",
                        "read": f"ADX {adx:.0f} — trend is weak; range-bound conditions."})

    price = m.get("close")
    e200 = m.get("ema_200") or m.get("sma_200")
    if price and e200:
        up = price > e200
        out.append({"indicator": "Trend", "value": None, "signal": "bullish" if up else "bearish",
                    "read": f"Price is {'above' if up else 'below'} the 200-DMA — primary trend is {'up' if up else 'down'}."})

    vr = m.get("volume_ratio")
    if vr is not None and vr >= 1.5:
        out.append({"indicator": "Volume", "value": round(vr, 1), "signal": "high",
                    "read": f"Volume is {vr:.1f}× its 20-day average — elevated participation."})
    return out


def bias(notes: List[Dict[str, Any]]) -> str:
    b = sum(1 for n in notes if n["signal"] in _BULL)
    s = sum(1 for n in notes if n["signal"] in _BEAR)
    if b > s:
        return "bullish"
    if s > b:
        return "bearish"
    return "mixed"


def interpret_symbol(symbol: str, *, use_llm: bool = False, user_id: Optional[str] = None) -> Dict[str, Any]:
    sym = symbol.strip().upper()
    notes: List[Dict[str, Any]] = []
    try:
        from ...data.market import get_market_data_provider
        from ml.features.indicators import compute_all_indicators
        df = get_market_data_provider().get_historical(sym, period="6mo", interval="1d")
        if df is not None and len(df) >= 30:
            df.columns = [c.lower() for c in df.columns]
            ind = compute_all_indicators(df)
            last = ind.iloc[-1]
            m = {k: (float(last[k]) if k in ind.columns and last[k] == last[k] else None)
                 for k in ("rsi_14", "macd_hist", "adx", "di_plus", "di_minus",
                           "ema_200", "sma_200", "close", "volume_ratio")}
            notes = interpret(m)
    except Exception as e:
        logger.debug("interpret_symbol failed for %s: %s", sym, e)

    b = bias(notes)
    narrative = None
    if use_llm and notes:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            {"symbol": sym, "bias": b, "indicators": notes},
            f"In one sentence, what's the combined technical picture for {sym}?",
            cache_key=f"interp:{sym}:{date.today().isoformat()}", user_id=user_id)
    return {"symbol": sym, "bias": b, "indicators": notes, "narrative": narrative}
