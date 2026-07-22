"""Technical Panel — the full per-symbol technicals system (2026-07-21).

One deterministic endpoint that gives an AI trader everything the old
KeyLevels card didn't: the complete oscillator suite with per-indicator
reads, every moving average with its price-vs-MA vote, classic floor
pivots + CPR, KDE-clustered swing support/resistance with touch counts,
Fibonacci retracement, 52-week anchors, ATR volatility, candlestick
patterns on the last bar — and a **technical sentiment** gauge computed
by tallying the votes (bullish / bearish / neutral language only, never
buy/sell).

All of it reuses the battle-tested ml.features.indicators module (the
same math the scanners and strategies run on) over settled EOD candles —
SEBI-safe published data, cached per symbol per day.

`sentiment_read` layers the AI: technical sentiment + news mood + market
regime fused into one grounded, day-cached narrative.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# symbol -> (iso_day, payload). EOD inputs — one compute per symbol per day.
_CACHE: Dict[str, Tuple[str, Dict[str, Any]]] = {}


def _fv(row: pd.Series, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _vote(label: str) -> str:
    return label  # 'bullish' | 'bearish' | 'neutral' — kept for readability


def _osc_rows(idf: pd.DataFrame, last: pd.Series) -> List[Dict[str, Any]]:
    """Oscillator suite: value + vote + one-line read per indicator.
    Extremes vote contrarian (oversold → bullish); trend tools vote with
    the trend — the standard technical-rating convention."""
    import ta as _ta

    high, low, close = idf["high"], idf["low"], idf["close"]
    vol = idf["volume"].astype(float)
    out: List[Dict[str, Any]] = []

    def add(key: str, label: str, value: Optional[float], vote: str, read: str, digits: int = 1):
        if value is None or pd.isna(value):
            return
        out.append({
            "key": key, "label": label, "value": round(float(value), digits),
            "vote": vote, "read": read,
        })

    rsi = _fv(last, "rsi_14")
    if rsi is not None:
        v = "bullish" if rsi < 30 else "bearish" if rsi > 70 else "neutral"
        read = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "mid-range"
        add("rsi", "RSI (14)", rsi, v, read)

    macd, sig = _fv(last, "macd"), _fv(last, "macd_signal")
    if macd is not None and sig is not None:
        v = "bullish" if macd > sig else "bearish"
        add("macd", "MACD (12,26,9)", macd, v, "above signal" if macd > sig else "below signal", 2)

    adx, dip, dim = _fv(last, "adx"), _fv(last, "di_plus"), _fv(last, "di_minus")
    if adx is not None:
        if adx < 20 or dip is None or dim is None:
            add("adx", "ADX (14)", adx, "neutral", "weak trend")
        else:
            v = "bullish" if dip > dim else "bearish"
            add("adx", "ADX (14)", adx, v, f"trending · DI{'+' if dip > dim else '−'} leads")

    try:
        k = _ta.momentum.stoch(high, low, close, window=14, smooth_window=3).iloc[-1]
        d = _ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3).iloc[-1]
        if not pd.isna(k):
            v = "bullish" if k < 20 else "bearish" if k > 80 else "neutral"
            add("stoch", "Stochastic %K (14,3)", float(k), v,
                "oversold" if k < 20 else "overbought" if k > 80 else f"%D {float(d):.1f}")
    except Exception:
        pass

    try:
        cci = _ta.trend.cci(high, low, close, window=20).iloc[-1]
        if not pd.isna(cci):
            v = "bullish" if cci < -100 else "bearish" if cci > 100 else "neutral"
            add("cci", "CCI (20)", float(cci), v,
                "oversold" if cci < -100 else "overbought" if cci > 100 else "mid-range", 0)
    except Exception:
        pass

    try:
        wr = _ta.momentum.williams_r(high, low, close, lbp=14).iloc[-1]
        if not pd.isna(wr):
            v = "bullish" if wr < -80 else "bearish" if wr > -20 else "neutral"
            add("willr", "Williams %R (14)", float(wr), v,
                "oversold" if wr < -80 else "overbought" if wr > -20 else "mid-range", 0)
    except Exception:
        pass

    try:
        mfi = _ta.volume.money_flow_index(high, low, close, vol, window=14).iloc[-1]
        if not pd.isna(mfi):
            v = "bullish" if mfi < 20 else "bearish" if mfi > 80 else "neutral"
            add("mfi", "MFI (14)", float(mfi), v,
                "money-flow oversold" if mfi < 20 else "money-flow overbought" if mfi > 80 else "balanced flow", 0)
    except Exception:
        pass

    try:
        roc = _ta.momentum.roc(close, window=12).iloc[-1]
        if not pd.isna(roc):
            add("roc", "ROC (12)", float(roc), "bullish" if roc > 0 else "bearish",
                "momentum up" if roc > 0 else "momentum down")
    except Exception:
        pass

    st_dir = last.get("supertrend_direction")
    st_val = _fv(last, "supertrend")
    if st_val is not None and st_dir is not None and not pd.isna(st_dir):
        v = "bullish" if int(st_dir) == 1 else "bearish"
        add("supertrend", "SuperTrend (10,2)", st_val, v, "price above stop" if v == "bullish" else "price below stop")

    psar_bull = last.get("psar_bullish")
    psar = _fv(last, "psar")
    if psar is not None and psar_bull is not None and not pd.isna(psar_bull):
        v = "bullish" if bool(psar_bull) else "bearish"
        add("psar", "Parabolic SAR", psar, v, "SAR below price" if v == "bullish" else "SAR above price")

    return out


_MA_KEYS: List[Tuple[str, str]] = [
    ("ema_10", "EMA 10"), ("ema_21", "EMA 21"), ("sma_20", "SMA 20"),
    ("sma_50", "SMA 50"), ("sma_150", "SMA 150"), ("sma_200", "SMA 200"),
    ("ema_200", "EMA 200"),
]


def _ma_rows(last: pd.Series, close: float) -> List[Dict[str, Any]]:
    out = []
    for key, label in _MA_KEYS:
        v = _fv(last, key)
        if v is None or v <= 0:
            continue
        out.append({
            "key": key, "label": label, "value": round(v, 2),
            "vote": "bullish" if close > v else "bearish",
            "dist_pct": round((close - v) / v * 100, 2),
        })
    return out


def _tally(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    bull = sum(1 for r in rows if r["vote"] == "bullish")
    bear = sum(1 for r in rows if r["vote"] == "bearish")
    neut = sum(1 for r in rows if r["vote"] == "neutral")
    cast = bull + bear
    if cast == 0:
        label = "neutral"
    else:
        share = bull / cast
        label = (
            "strong bullish" if share >= 0.72 and cast >= 5
            else "bullish" if share >= 0.58
            else "strong bearish" if share <= 0.28 and cast >= 5
            else "bearish" if share <= 0.42
            else "neutral"
        )
    return {"bullish": bull, "bearish": bear, "neutral": neut, "label": label}


def _classic_pivots(idf: pd.DataFrame) -> Dict[str, float]:
    """Floor pivots off the last completed session (today's bar is the last
    row of EOD data, so 'prior session' = the last row itself after close)."""
    h = float(idf["high"].iloc[-1])
    lo = float(idf["low"].iloc[-1])
    c = float(idf["close"].iloc[-1])
    p = (h + lo + c) / 3
    return {
        "p": round(p, 2),
        "r1": round(2 * p - lo, 2), "s1": round(2 * p - h, 2),
        "r2": round(p + (h - lo), 2), "s2": round(p - (h - lo), 2),
        "r3": round(h + 2 * (p - lo), 2), "s3": round(lo - 2 * (h - p), 2),
    }


def technical_panel(symbol: str) -> Dict[str, Any]:
    """The full technical read. Deterministic, EOD, day-cached."""
    sym = symbol.strip().upper()
    today = date.today().isoformat()
    hit = _CACHE.get(sym)
    if hit and hit[0] == today:
        return hit[1]

    from ...data.market import get_market_data_provider
    df = get_market_data_provider().get_historical(sym, period="2y", interval="1d")
    if df is None or df.empty or len(df) < 60:
        return {
            "symbol": sym, "available": False,
            "note": f"Not enough price history to compute the technical panel for {sym}.",
        }
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    from ml.features.indicators import (
        compute_all_indicators,
        detect_fibonacci_levels,
        detect_support_resistance_with_touches,
    )

    idf = compute_all_indicators(df)
    last = idf.iloc[-1]
    close = float(last["close"])

    oscillators = _osc_rows(idf, last)
    mas = _ma_rows(last, close)
    osc_summary = _tally(oscillators)
    ma_summary = _tally(mas)
    overall = _tally(oscillators + mas)

    # KDE-clustered swing S/R over ~6 months, split around price, nearest first.
    supports: List[Dict[str, Any]] = []
    resistances: List[Dict[str, Any]] = []
    try:
        sup, res = detect_support_resistance_with_touches(idf, lookback=120)
        for price, touches in sup + res:
            row = {
                "price": round(float(price), 2), "touches": int(touches),
                "dist_pct": round((float(price) - close) / close * 100, 2),
            }
            (supports if price < close else resistances).append(row)
        supports.sort(key=lambda r: -r["price"])       # nearest below first
        resistances.sort(key=lambda r: r["price"])     # nearest above first
        supports, resistances = supports[:4], resistances[:4]
    except Exception as e:
        logger.debug("technical_panel S/R failed for %s: %s", sym, e)

    fib = None
    try:
        f = detect_fibonacci_levels(idf, lookback=120)
        if f:
            fib = {
                "trend": f["trend"],
                "swing_high": round(float(f["swing_high"]), 2),
                "swing_low": round(float(f["swing_low"]), 2),
                "levels": {str(k): round(float(v), 2) for k, v in f["levels"].items()},
            }
    except Exception as e:
        logger.debug("technical_panel fib failed for %s: %s", sym, e)

    atr = _fv(last, "atr_14")
    patterns = sorted(
        c.replace("candle_", "").replace("_", " ")
        for c in idf.columns
        if c.startswith("candle_") and bool(last.get(c))
    )

    as_of = str(idf.index[-1])[:10] if hasattr(idf.index, "__getitem__") else today
    payload: Dict[str, Any] = {
        "symbol": sym,
        "available": True,
        "as_of": as_of,
        "price": round(close, 2),
        "summary": {"oscillators": osc_summary, "moving_averages": ma_summary, "overall": overall},
        "oscillators": oscillators,
        "moving_averages": mas,
        "pivots": _classic_pivots(idf),
        "cpr": {
            "tc": _fv(last, "cpr_tc"), "p": _fv(last, "cpr_p"), "bc": _fv(last, "cpr_bc"),
            "narrow": bool(last.get("cpr_narrow")) if last.get("cpr_narrow") is not None and not pd.isna(last.get("cpr_narrow")) else None,
        },
        "supports": supports,
        "resistances": resistances,
        "fibonacci": fib,
        "week52": {"high": _fv(last, "high_52w"), "low": _fv(last, "low_52w")},
        "atr": {"value": round(atr, 2) if atr else None, "pct": round(atr / close * 100, 2) if atr else None},
        "candle_patterns": patterns,
        "golden_cross": bool(last.get("golden_cross")) if last.get("golden_cross") is not None and not pd.isna(last.get("golden_cross")) else None,
        "note": "EOD · derived from settled closes · analysis, not investment advice",
    }
    _CACHE[sym] = (today, payload)
    return payload


# ── Sentiment read: technical + news + market, with optional AI narrative ──

_SENT_SYSTEM = (
    "You are an Indian-equities desk analyst. Fuse the three sentiment layers "
    "in the facts JSON — the stock's TECHNICAL sentiment (indicator/MA votes), "
    "its NEWS mood, and the MARKET backdrop (regime, VIX, breadth) — into one "
    "read for a swing trader. Say where the layers agree, where they conflict, "
    "and which layer deserves the most weight right now and why. Use ONLY the "
    "exact numbers in the facts. Plain prose, 3-5 sentences, no emoji, no "
    "markdown. Describe direction as bullish/bearish/neutral — never tell the "
    "reader to buy or sell. Engines are Alpha, Mood, or Regime only."
)


def sentiment_read(symbol: str, *, use_llm: bool = False, user_id: Optional[str] = None) -> Dict[str, Any]:
    """{technical, news, market, narrative} — deterministic layers always;
    grounded narrative (day-cached) only when use_llm."""
    sym = symbol.strip().upper()
    facts: Dict[str, Any] = {"symbol": sym, "as_of": date.today().isoformat()}

    try:
        tp = technical_panel(sym)
        if tp.get("available"):
            facts["technical"] = {
                "summary": tp["summary"]["overall"],
                "oscillators": tp["summary"]["oscillators"],
                "moving_averages": tp["summary"]["moving_averages"],
                "adx_note": next((o["read"] for o in tp["oscillators"] if o["key"] == "adx"), None),
            }
    except Exception as e:
        logger.debug("sentiment_read technical failed %s: %s", sym, e)

    try:
        from ...core.database import get_supabase_admin
        rows = (
            get_supabase_admin().table("news_sentiment")
            .select("symbol,mean_score,headline_count,trade_date")
            .eq("symbol", sym).order("trade_date", desc=True).limit(1).execute()
        ).data or []
        if rows:
            r = rows[0]
            score = r.get("mean_score")
            if score is not None:
                facts["news"] = {
                    "mood_score": round(float(score), 3),
                    "label": "bullish" if float(score) > 0.15 else "bearish" if float(score) < -0.15 else "neutral",
                    "headline_count": r.get("headline_count"),
                    "as_of": r.get("trade_date"),
                }
    except Exception as e:
        logger.debug("sentiment_read news failed %s: %s", sym, e)

    try:
        from ..regime.refresh import current_regime
        reg = current_regime()
        if reg:
            facts["market"] = {
                "regime": reg.get("regime"),
                "confidence": reg.get("confidence"),
            }
    except Exception as e:
        logger.debug("sentiment_read regime failed %s: %s", sym, e)

    narrative: Optional[str] = None
    if use_llm and (facts.get("technical") or facts.get("news")):
        try:
            from ...ai.agents.grounded import grounded_reason
            narrative = grounded_reason(
                facts,
                f"Fuse the technical, news, and market sentiment layers for {sym} "
                "into one read: agreement, conflict, and which layer to weight.",
                cache_key=f"sentread:v1:{sym}:{date.today().isoformat()}",
                system=_SENT_SYSTEM, user_id=user_id)
        except Exception as e:
            logger.debug("sentiment_read narrative failed %s: %s", sym, e)

    return {"symbol": sym, **facts, "narrative": narrative}
