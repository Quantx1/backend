"""Result formatting + signal classification for the screener.

Pure functions: row in, dict/string out. The two ``format_*`` builders
take ``stock_info`` as an explicit dependency (the NSE_STOCK_INFO table
lives next to the engine) so this module has no upward imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

# Legacy pattern labels (_CONSOLIDATION_LABELS / _REVERSAL_LABELS) removed
# 2026-05-31 along with the chart-pattern scanner IDs they served.


# ────────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────────


def classify_trend(row) -> str:
    """Classify trend from real indicators."""
    close = float(row.get('close', 0))
    sma_20 = float(row.get('sma_20', 0))
    sma_50 = float(row.get('sma_50', 0))
    adx = float(row.get('adx', 0))

    if sma_20 > 0 and sma_50 > 0 and close > sma_20 > sma_50:
        return "Strong Up" if adx > 25 else "Up"
    if sma_20 > 0 and close > sma_20:
        return "Up"
    if sma_20 > 0 and sma_50 > 0 and close < sma_20 < sma_50:
        return "Strong Down" if adx > 25 else "Down"
    if sma_20 > 0 and close < sma_20:
        return "Down"
    return "Sideways"


def classify_ma_signal(row) -> str:
    """Classify moving-average alignment."""
    close = float(row.get('close', 0))
    ema_21 = float(row.get('ema_21', 0))
    sma_50 = float(row.get('sma_50', 0))
    sma_200 = float(row.get('sma_200', 0))

    if sma_50 > 0 and sma_200 > 0 and sma_50 > sma_200 and close > sma_50:
        return "Golden Cross"
    if ema_21 > 0 and sma_50 > 0 and ema_21 > sma_50:
        return "Bull Cross"
    if sma_200 > 0 and close > sma_200:
        return "Above 200 SMA"
    if sma_50 > 0 and close > sma_50:
        return "Above 50 SMA"
    if ema_21 > 0 and close > ema_21:
        return "Above 20 EMA"
    return "Below MAs"


def detect_pattern_label(row, scanner_id: int) -> str:
    """Pattern label from indicator flags, with scanner-id fallback."""
    if row.get('candle_engulfing_bull'):
        return "Bullish Engulfing"
    if row.get('candle_engulfing_bear'):
        return "Bearish Engulfing"
    if row.get('candle_hammer'):
        return "Hammer"
    if row.get('candle_morning_star'):
        return "Morning Star"
    if row.get('candle_doji'):
        return "Doji"
    if row.get('nr4'):
        return "NR4"
    if row.get('nr7'):
        return "NR7"
    if row.get('inside_bar'):
        return "Inside Bar"
    if row.get('ttm_squeeze'):
        return "TTM Squeeze"

    scanner_patterns = {
        1: "Consolidation Breakout", 4: "Volume Breakout", 5: "52W High",
        6: "10D High", 7: "52W Low", 8: "Volume Surge", 9: "RSI Oversold",
        10: "RSI Overbought", 14: "VCP", 17: "Momentum", 26: "MACD Cross",
        30: "Momentum Burst", 31: "Trend Template", 32: "SuperTrend Bullish",
        33: "Pivot Breakout",
    }
    return scanner_patterns.get(scanner_id, "")


def generate_signal(row, scanner_id: int) -> str:
    """Generate a Buy/Sell/Hold label from the row + scanner type."""
    change = float(row.get('change_pct', 0))
    rsi = float(row.get('rsi_14', 50))

    # Bearish scanners
    if scanner_id in (3, 13, 27):
        return "Sell" if change < -3 else "Weak"

    # RSI-based
    if scanner_id == 9:    # Oversold
        return "Strong Buy" if rsi < 25 else "Buy"
    if scanner_id == 10:   # Overbought
        return "Take Profit" if rsi > 80 else "Hold"

    # Bullish scanners
    if scanner_id in (1, 2, 4, 5, 12, 14, 17, 30, 31):
        if change > 3 and rsi > 60:
            return "Strong Buy"
        if change > 1:
            return "Buy"
        return "Hold"

    # Default
    if change > 2:
        return "Buy"
    if change < -2:
        return "Sell"
    return "Hold"


# ────────────────────────────────────────────────────────────────────
# Row formatting
# ────────────────────────────────────────────────────────────────────


def format_stock_result(
    row: Mapping[str, Any],
    scanner_id: int,
    stock_info: Mapping[str, Mapping[str, str]],
    pattern_override: Optional[str] = None,
) -> Dict:
    """Format a single stock summary row for the frontend JSON contract.

    PR-S9 (2026-05-31): every result now ships with a 0-100 quality
    score, ATR-derived suggested entry/stop/target/RR, and a short
    triggers list ("Above EMA200", "Vol 2.4× avg", etc.) so the trader
    can rank cards inside the same scanner output and act in one click.
    """
    from .quality import enrich_row

    symbol = row['symbol'] if isinstance(row, dict) else row.get('symbol', '')
    info = stock_info.get(symbol, {})
    close = float(row.get('close', 0))
    atr = float(row.get('atr_14', close * 0.02))

    result = {
        "symbol": symbol,
        "name": info.get("name", symbol),
        "sector": info.get("sector", ""),
        "ltp": round(close, 2),
        "change_pct": round(float(row.get('change_pct', 0)), 2),
        "volume": f"{float(row.get('volume_ratio', 1.0)):.1f}x",
        "volume_ratio": round(float(row.get('volume_ratio', 1.0)), 2),
        "volume_raw": int(float(row.get('volume', 0))),
        "rsi": round(float(row.get('rsi_14', 50))),
        "trend": classify_trend(row),
        "pattern": pattern_override or detect_pattern_label(row, scanner_id),
        "signal": generate_signal(row, scanner_id),
        "ma_signal": classify_ma_signal(row),
        "breakout_level": round(float(row.get('bb_upper', close * 1.05)), 2),
        "support_level": round(float(row.get('bb_lower', close * 0.95)), 2),
        "target_1": round(close + atr * 1.5, 2),
        "target_2": round(close + atr * 3.0, 2),
        "stop_loss": round(close - atr * 1.5, 2),
    }
    # PR-S9 enrichment — quality score + ATR levels + triggers
    return enrich_row(result, row if hasattr(row, "get") else pd.Series(row))


def format_for_frontend(
    df: pd.DataFrame,
    scanner_id: int,
    stock_info: Mapping[str, Mapping[str, str]],
) -> List[Dict]:
    """Convert filtered DataFrame to frontend-expected JSON list (max 50)."""
    if df.empty:
        return []
    return [
        format_stock_result(row, scanner_id, stock_info)
        for _, row in df.head(50).iterrows()
    ]
