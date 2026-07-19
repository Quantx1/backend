"""
Pure scanner filter functions — Quant X Screener filter library.

Each filter is a stateless ``(df: pd.DataFrame) -> pd.DataFrame`` that
selects rows from the pre-computed summary DataFrame (one row per
symbol, all indicators populated). Dispatched via ``SCANNER_FILTERS``
(scanner_id → callable). Used by ``LiveScreenerEngine.run_scanner()``.

Extracted from ``live_screener_engine.py`` so the engine class stays
focused on orchestration (universe load, indicator compute, dispatch,
caching) instead of mixing 40+ filter bodies inline.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def _match_symbols(df: pd.DataFrame, symbols: Iterable) -> pd.DataFrame:
    """Select rows of the summary frame whose symbol is in ``symbols``.

    The live scan path passes ``pd.DataFrame(summary_rows)`` — symbol is a
    COLUMN with a RangeIndex (engine.py); some callers pass a symbol-indexed
    frame. Matching only ``df.index`` (as a few older scanners did) silently
    returned empty in the live path. This handles both shapes.
    """
    symset = set(symbols)
    if df.empty or not symset:
        return df.iloc[0:0]
    if "symbol" in df.columns:
        return df[df["symbol"].isin(symset)]
    return df[df.index.isin(symset)]


# =============================================================================
# SCANNER FILTER FUNCTIONS
# =============================================================================

def _filter_full_screening(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 0: Full screening - all stocks with decent volume"""
    return df[df['volume_ratio'] > 0.5].sort_values('change_pct', ascending=False)


def _filter_breakout_consolidation(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 1: Breakout from consolidation - low ATR + volume surge"""
    atr_pct = df['atr_14'] / df['close']
    return df[
        (atr_pct < 0.025) &  # Low volatility (consolidation)
        (df['volume_ratio'] > 1.5) &  # Volume surge
        (df['close'] > df['sma_20']) &  # Above 20 SMA
        (df['change_pct'] > 0.5)  # Positive day
    ].sort_values('volume_ratio', ascending=False)


def _filter_top_gainers(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 2: Top gainers (>2%)"""
    return df[df['change_pct'] > 2.0].sort_values('change_pct', ascending=False)


def _filter_top_losers(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 3: Top losers (<-2%)"""
    return df[df['change_pct'] < -2.0].sort_values('change_pct', ascending=True)


def _filter_volume_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 4: Volume breakout"""
    return df[
        (df['volume_ratio'] > 2.0) &
        (df['change_pct'] > 1.0)
    ].sort_values('volume_ratio', ascending=False)


def _filter_52w_high(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 5: At/near 52-week high"""
    return df[df['close'] >= df['high_52w'] * 0.98].sort_values('change_pct', ascending=False)


def _filter_10d_high(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 6: At 10-day high"""
    return df[df['close'] >= df['high_10d'] * 0.99].sort_values('change_pct', ascending=False)


def _filter_52w_low(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 7: Near 52-week low (reversal potential)"""
    return df[df['close'] <= df['low_52w'] * 1.05].sort_values('change_pct', ascending=False)


def _filter_volume_surge(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 8: Volume > 2.5x average"""
    return df[df['volume_ratio'] > 2.5].sort_values('volume_ratio', ascending=False)


def _filter_rsi_oversold(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 9: RSI < 30"""
    return df[df['rsi_14'] < 30].sort_values('rsi_14', ascending=True)


def _filter_rsi_overbought(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 10: RSI > 70"""
    return df[df['rsi_14'] > 70].sort_values('rsi_14', ascending=False)


def _filter_ma_crossover(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 11: Price near/crossing 20 EMA"""
    pct_from_ema = abs(df['close'] - df['ema_21']) / df['close']
    return df[
        (df['close'] > df['ema_21']) &
        (pct_from_ema < 0.02)  # Within 2% of EMA
    ].sort_values('change_pct', ascending=False)


def _filter_bullish_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 12: Bullish engulfing pattern"""
    col = 'candle_engulfing_bull'
    if col not in df.columns:
        return pd.DataFrame()
    return df[df[col]].sort_values('change_pct', ascending=False)


def _filter_bearish_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 13: Bearish engulfing pattern"""
    col = 'candle_engulfing_bear'
    if col not in df.columns:
        return pd.DataFrame()
    return df[df[col]].sort_values('change_pct', ascending=True)


def _filter_vcp(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 14: VCP (Volatility Contraction Pattern)"""
    atr_pct = df['atr_14'] / df['close']
    return df[
        (atr_pct < 0.02) &  # Low volatility
        (df['volume_ratio'] < 0.8) &  # Declining volume
        (df['close'] > df['sma_50']) &  # Above 50 SMA
        (abs(df['close'] - df['sma_20']) / df['close'] < 0.03)  # Tight consolidation
    ].sort_values('close', ascending=False)


def _filter_bull_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 15: 20 EMA crossing above 50 SMA"""
    ema_above = df['ema_21'] > df['sma_50']
    close_cross = abs(df['ema_21'] - df['sma_50']) / df['sma_50'] < 0.01
    return df[ema_above & close_cross].sort_values('change_pct', ascending=False)


def _filter_bull_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 17: Strong bullish momentum"""
    return df[
        (df['rsi_14'] > 55) &
        (df['macd'] > df['macd_signal']) &
        (df['close'] > df['ema_21']) &
        (df['adx'] > 20) &
        (df['change_pct'] > 0)
    ].sort_values('rsi_14', ascending=False)


def _filter_atr_trailing(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 18: Price above ATR trailing stop"""
    if 'atr_trailing_stop' not in df.columns:
        return pd.DataFrame()
    return df[
        (df['close'] > df['atr_trailing_stop']) &
        (df['close'] > df['sma_50']) &
        (df['change_pct'] > 0)
    ].sort_values('change_pct', ascending=False)


def _filter_psar_reversal(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 19: Parabolic SAR bullish reversal"""
    if 'psar_bullish' not in df.columns:
        return pd.DataFrame()
    return df[df['psar_bullish']].sort_values('change_pct', ascending=False)


def _filter_nr4(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 21: NR4 pattern"""
    if 'nr4' not in df.columns:
        return pd.DataFrame()
    return df[df['nr4']].sort_values('volume_ratio', ascending=False)


def _filter_nr7(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 22: NR7 pattern"""
    if 'nr7' not in df.columns:
        return pd.DataFrame()
    return df[df['nr7']].sort_values('volume_ratio', ascending=False)


def _filter_macd_crossover(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 26: MACD bullish crossover"""
    return df[
        (df['macd'] > df['macd_signal']) &
        (df['macd_hist'] > 0) &
        (df['macd_hist'] < abs(df['macd_signal']) * 0.15)  # Recent cross
    ].sort_values('macd_hist', ascending=True)


def _filter_macd_bearish(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 27: MACD bearish crossover"""
    return df[
        (df['macd'] < df['macd_signal']) &
        (df['macd_hist'] < 0)
    ].sort_values('macd_hist', ascending=True)


def _filter_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 28: Inside bar pattern"""
    if 'inside_bar' not in df.columns:
        return pd.DataFrame()
    return df[df['inside_bar']].sort_values('volume_ratio', ascending=False)


def _filter_ttm_squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 29: TTM Squeeze (BB inside KC)"""
    if 'ttm_squeeze' not in df.columns:
        return pd.DataFrame()
    return df[df['ttm_squeeze']].sort_values('volume_ratio', ascending=False)


def _filter_momentum_burst(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 30: Sudden momentum increase"""
    return df[
        (df['change_pct'] > 3.0) &
        (df['volume_ratio'] > 2.0) &
        (df['rsi_14'] > 50)
    ].sort_values('change_pct', ascending=False)


def _filter_trend_template(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 31: Minervini Trend Template"""
    within_25_of_high = df['close'] >= df['high_52w'] * 0.75
    above_25_of_low = df['close'] >= df['low_52w'] * 1.25
    return df[
        (df['close'] > df['sma_50']) &
        (df['sma_50'] > df['sma_200']) &
        (df['close'] > df['sma_200']) &
        within_25_of_high &
        above_25_of_low &
        (df['rsi_14'] > 50)
    ].sort_values('change_pct', ascending=False)


def _filter_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 32: SuperTrend bullish signal"""
    if 'supertrend_direction' not in df.columns:
        return pd.DataFrame()
    return df[df['supertrend_direction'] == 1].sort_values('change_pct', ascending=False)


def _filter_pivot_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 33: Breaking above pivot R1"""
    if 'pivot_r1' not in df.columns:
        return pd.DataFrame()
    return df[df['close'] > df['pivot_r1']].sort_values('change_pct', ascending=False)


def _filter_high_tight_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 48: High & Tight Flag — stock up 50%+ in 2 months with tight recent consolidation."""
    if df.empty or 'sma_50' not in df.columns:
        return pd.DataFrame()
    # Stock must be well above 50-day MA (strong prior run)
    strong_run = df[df['close'] > df['sma_50'] * 1.3].copy()
    if strong_run.empty:
        return pd.DataFrame()
    # Tight consolidation: low ATR relative to price (< 2% of close)
    if 'atr_14' in strong_run.columns:
        strong_run['atr_pct'] = strong_run['atr_14'] / strong_run['close'] * 100
        tight = strong_run[strong_run['atr_pct'] < 2.5]
        return tight.sort_values('change_pct', ascending=False).head(20)
    return strong_run.sort_values('change_pct', ascending=False).head(20)


# ---------------------------------------------------------------------------
# Scanners 34-42 — real NSE institutional data
# Uses NSEDataService for delivery %, FII/DII, bulk deals, F&O OI
# ---------------------------------------------------------------------------

def _filter_high_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 34: High delivery % — real NSE data."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        delivery_df = nse.get_delivery_data()
        if delivery_df.empty:
            return pd.DataFrame()
        # High delivery % (> 50% = accumulation, low intraday churn).
        high_del = delivery_df[delivery_df['delivery_pct'] > 50].copy()
        if high_del.empty:
            return pd.DataFrame()
        del_map = high_del.set_index('symbol')['delivery_pct'].to_dict()
        # Enrich with the summary df's columns when the symbol is in our
        # universe; fall back to the raw delivery rows otherwise.
        merged = _match_symbols(df, del_map.keys()).copy()
        if not merged.empty:
            key = merged['symbol'] if 'symbol' in merged.columns else merged.index.to_series()
            merged['delivery_pct'] = key.map(del_map).to_numpy()
            return merged.sort_values('delivery_pct', ascending=False)
        return high_del.sort_values('delivery_pct', ascending=False).head(30)
    except Exception as e:
        logger.debug(f"Delivery scanner fallback: {e}")
        return pd.DataFrame()


def _filter_bulk_deals(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 35: Bulk/block deals — real NSE data."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        deals = nse.get_bulk_deals()
        if not deals:
            return pd.DataFrame()
        deal_symbols = {d['symbol'] for d in deals if d.get('symbol')}
        matched = _match_symbols(df, deal_symbols)
        if not matched.empty:
            return matched.sort_values('volume_ratio', ascending=False)
        return pd.DataFrame(deals).head(30)
    except Exception as e:
        logger.debug(f"Bulk deals scanner fallback: {e}")
        return pd.DataFrame()


def _filter_fii_net_buyers(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 36: FII net buyers — real NSE FII/DII data + volume/momentum filter.
    When FII net positive, show F&O stocks with bullish momentum."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        fii_dii = nse.get_fii_dii_activity()

        if fii_dii.get("fii_net", 0) <= 0:
            # FII selling today — show empty (no FII buying signal)
            return pd.DataFrame()

        # FII buying: show F&O stocks with volume surge + bullish structure
        if df.empty:
            return pd.DataFrame()
        return df[
            (df['volume_ratio'] > 1.5) &
            (df['change_pct'] > 0.5) &
            (df['close'] > df['sma_50']) &
            (df['rsi_14'] > 50)
        ].sort_values('volume_ratio', ascending=False).head(30)
    except Exception as e:
        logger.debug(f"FII scanner fallback: {e}")
        return pd.DataFrame()


def _filter_dii_net_buyers(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 37: DII net buyers — real NSE data + accumulation filter."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        fii_dii = nse.get_fii_dii_activity()

        if fii_dii.get("dii_net", 0) <= 0:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()
        return df[
            (df['volume_ratio'] > 1.2) &
            (df['close'] > df['sma_200']) &
            (df['change_pct'] > 0)
        ].sort_values('volume_ratio', ascending=False).head(30)
    except Exception as e:
        logger.debug(f"DII scanner fallback: {e}")
        return pd.DataFrame()


def _filter_institutional_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 38: Combined FII+DII positive — real NSE data."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        fii_dii = nse.get_fii_dii_activity()

        fii_net = fii_dii.get("fii_net", 0)
        dii_net = fii_dii.get("dii_net", 0)

        if fii_net + dii_net <= 0:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()
        return df[
            (df['volume_ratio'] > 2.0) &
            (df['change_pct'] > 0.5) &
            (df['close'] > df['sma_50'])
        ].sort_values('volume_ratio', ascending=False).head(30)
    except Exception as e:
        logger.debug(f"Institutional combined scanner fallback: {e}")
        return pd.DataFrame()


def _filter_oi_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 39: OI analysis — real NSE F&O OI spurt data."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        oi_data = nse.get_participant_oi()
        spurts = oi_data.get("data", [])

        if not spurts:
            return pd.DataFrame()

        # Sort by absolute OI change %
        spurts_sorted = sorted(spurts, key=lambda x: abs(x.get("oi_change_pct", 0)), reverse=True)
        oi_symbols = [s['symbol'] for s in spurts_sorted[:30] if abs(s.get("oi_change_pct", 0)) > 5]

        matched = _match_symbols(df, oi_symbols)
        if not matched.empty:
            return matched
        return pd.DataFrame(spurts_sorted[:30])
    except Exception as e:
        logger.debug(f"OI scanner fallback: {e}")
        return pd.DataFrame()


def _filter_long_buildup(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 40: Long buildup — real NSE OI data (price up + OI up)."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        oi_data = nse.get_participant_oi()
        spurts = oi_data.get("data", [])

        if not spurts:
            return pd.DataFrame()

        # Long buildup: price up + OI increase
        long_symbols = [
            s['symbol'] for s in spurts
            if s.get("change_pct", 0) > 0.5 and s.get("oi_change_pct", 0) > 5
        ]

        matched = _match_symbols(df, long_symbols)
        if not matched.empty:
            return matched.sort_values('change_pct', ascending=False)
        return pd.DataFrame()
    except Exception as e:
        logger.debug(f"Long buildup scanner fallback: {e}")
        return pd.DataFrame()


def _filter_short_buildup(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 41: Short buildup — real NSE OI data (price down + OI up)."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        oi_data = nse.get_participant_oi()
        spurts = oi_data.get("data", [])

        if not spurts:
            return pd.DataFrame()

        # Short buildup: price down + OI increase
        short_symbols = [
            s['symbol'] for s in spurts
            if s.get("change_pct", 0) < -0.5 and s.get("oi_change_pct", 0) > 5
        ]

        matched = _match_symbols(df, short_symbols)
        if not matched.empty:
            return matched.sort_values('change_pct', ascending=True)
        return pd.DataFrame()
    except Exception as e:
        logger.debug(f"Short buildup scanner fallback: {e}")
        return pd.DataFrame()


def _filter_short_covering(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 42: Short covering — real NSE OI data (price up + OI down)."""
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        oi_data = nse.get_participant_oi()
        spurts = oi_data.get("data", [])

        if not spurts:
            return pd.DataFrame()

        # Short covering: price up + OI decrease
        cover_symbols = [
            s['symbol'] for s in spurts
            if s.get("change_pct", 0) > 1.0 and s.get("oi_change_pct", 0) < -5
        ]

        matched = _match_symbols(df, cover_symbols)
        if not matched.empty:
            return matched.sort_values('change_pct', ascending=False)
        return pd.DataFrame()
    except Exception as e:
        logger.debug(f"Short covering scanner fallback: {e}")
        return pd.DataFrame()


def _filter_ipo_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 16: IPO breakout — stocks near 52W high with momentum."""
    if df.empty:
        return pd.DataFrame()
    return df[
        (df['close'] > df['high_52w'] * 0.95) &
        (df['volume_ratio'] > 2.0) &
        (df['change_pct'] > 1.0) &
        (df['rsi_14'] > 60)
    ].sort_values('change_pct', ascending=False)


# =============================================================================
# PR-S9 — CUTTING-EDGE NEW SCANNERS (52..70)
# 10 high-impact setups missing from the legacy 50: power setup, squeeze
# release, multi-MA stack, pre-breakout coils, fresh trend starts, etc.
# Each is filterable in <50ms against the cached summary_df.
# =============================================================================


def _filter_power_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 52: Power Setup — 4-of-4 confirmation.

    The "any day you'd take this trade" composite: above EMA200 (trend) +
    RSI 50-70 (momentum building, not overbought) + volume confirm +
    ADX rising. Catches the high-probability swing entries.
    """
    return df[
        (df["close"] > df["ema_200"]) &
        (df["close"] > df["sma_50"]) &
        (df["rsi_14"].between(50, 70)) &
        (df["macd"] > df["macd_signal"]) &
        (df["volume_ratio"] > 1.3) &
        (df["adx"] > 22)
    ].sort_values(["volume_ratio", "rsi_14"], ascending=False)


def _filter_squeeze_release(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 53: Squeeze Release — TTM was squeezing, now breaking out.

    Looks for stocks that were inside-Keltner (low volatility) but TODAY
    have a volume + price expansion. The signature setup for a multi-day
    move out of consolidation.
    """
    if "ttm_squeeze" not in df.columns:
        return pd.DataFrame()
    # Was squeezing recently (low ATR%) + today expansion
    atr_pct = df["atr_14"] / df["close"]
    return df[
        (atr_pct < 0.025) &
        (df["volume_ratio"] > 1.8) &
        (df["change_pct"].abs() > 1.0) &
        (df["close"] > df["sma_50"])
    ].sort_values("volume_ratio", ascending=False)


def _filter_ma_stack_bullish(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 54: MA Stack Bullish — price > EMA21 > SMA50 > SMA200.

    Stocks in textbook uptrends. Order matters: when MAs are stacked in
    descending order with price on top, every pullback to a MA tends to
    hold. The cleanest swing-trade universe.
    """
    return df[
        (df["close"] > df["ema_21"]) &
        (df["ema_21"] > df["sma_50"]) &
        (df["sma_50"] > df["sma_200"]) &
        (df["change_pct"] > -1.0)   # not in active selloff
    ].sort_values("rsi_14", ascending=False)


def _filter_pre_breakout_coil(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 55: Pre-Breakout Coil — NR7 + above EMA21 + low volume.

    The setup BEFORE the breakout: tight range, holding key support,
    quiet volume = supply absorbed, breakout imminent. Trader who catches
    this gets the breakout candle, not the late-chase.
    """
    if "nr7" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["nr7"]) &
        (df["close"] > df["ema_21"]) &
        (df["volume_ratio"] < 1.0) &
        (df["close"] >= df["high_52w"] * 0.85)    # near highs
    ].sort_values("close", ascending=False)


def _filter_fresh_trend_start(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 56: Fresh Trend Start — ADX rising from below 20.

    The "first big bar" of a new trend. ADX cycling up from quiet zone
    (<20) with price above EMA50 means a regime change in momentum that
    typically has weeks of runway, not days.
    """
    return df[
        (df["adx"] > 20) &
        (df["adx"] < 30) &           # still early
        (df["close"] > df["ema_21"]) &
        (df["close"] > df["sma_50"]) &
        (df["change_pct"] > 1.5) &
        (df["volume_ratio"] > 1.2)
    ].sort_values("adx", ascending=True)   # earliest first


def _filter_oversold_bounce_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 57: Oversold Bounce Setup — RSI<35 but above SMA200.

    RSI extreme reads as panic noise UNLESS the stock is in a structural
    uptrend (close > SMA200). This combo is mean-reversion in a healthy
    trend — the highest win-rate reversal setup we screen for.
    """
    return df[
        (df["rsi_14"] < 35) &
        (df["close"] > df["sma_200"]) &
        (df["volume_ratio"] > 1.5)     # exhaustion vol = capitulation
    ].sort_values("rsi_14", ascending=True)


def _filter_breakout_with_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 58: Breakout w/ Volume — 52w high + vol 2×.

    New 52-week high paired with volume confirmation. Classic Stan Weinstein
    Stage-2 entry. Distinct from "near 52w" — this fires on the actual
    break, with volume vouching for it.
    """
    return df[
        (df["close"] >= df["high_52w"] * 0.99) &
        (df["volume_ratio"] > 2.0) &
        (df["change_pct"] > 1.5)
    ].sort_values("volume_ratio", ascending=False)


def _filter_pullback_to_ema21(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 59: Pullback to EMA21 — uptrend dip near a key MA.

    The buy-the-dip-not-the-rip filter. Stock is in an uptrend
    (above SMA50) but TODAY is at/near EMA21 (the most-watched short-MA
    in swing trading). Classic pullback continuation entry.
    """
    pct_from_ema = (df["close"] - df["ema_21"]) / df["close"]
    return df[
        (df["close"] > df["sma_50"]) &
        (df["sma_50"] > df["sma_200"]) &
        (pct_from_ema.between(-0.02, 0.01)) &     # within ±2/+1% of EMA21
        (df["rsi_14"].between(40, 55))            # mild pullback RSI
    ].sort_values("change_pct", ascending=True)


def _filter_bollinger_squeeze_release(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 60: Bollinger Squeeze Release — BB width was tight, expanding.

    Variant of squeeze-release using BB instead of TTM. Adds the
    directional bias (price crossed UPPER band) for clear bullish
    breakouts. Pair with volume gate so it's not a fakeout.
    """
    bb_width = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    return df[
        (bb_width > 0) &
        (df["close"] > df["bb_upper"] * 0.99) &      # touching/crossing upper band
        (df["volume_ratio"] > 1.5) &
        (df["close"] > df["sma_50"]) &
        (bb_width < 0.10)                            # range was reasonable
    ].sort_values("volume_ratio", ascending=False)


def _filter_relative_strength_leader(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 61: RS Leader — outperforming the market AND own sector.

    Pure RS: stock up while market is flat/down, OR up significantly more
    than market average. Using change_pct > 2× the universe median as a
    quick proxy when sector-RS data isn't available.

    Conservative gates: must also be above SMA50 (no junk bounces).
    """
    if df.empty:
        return df
    median_change = df["change_pct"].median()
    threshold = max(2.0, median_change + 2.0)
    return df[
        (df["change_pct"] > threshold) &
        (df["close"] > df["sma_50"]) &
        (df["rsi_14"] > 50) &
        (df["volume_ratio"] > 1.2)
    ].sort_values("change_pct", ascending=False)


# =============================================================================
# PR-S17 — BEARISH PARITY SCANNERS (62..71)
# Mirror the top-10 bullish setups so short-side traders aren't second-class
# citizens. Each one inverts the bullish thesis: instead of "above EMA200
# + RSI 50-70 + MACD up" we look for "below EMA200 + RSI 30-50 + MACD down".
# =============================================================================


def _filter_power_setup_short(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 62: Power Setup Short — bear-side 4-of-4 confirmation.

    Mirror of #52: below EMA200 (downtrend) + RSI 30-50 (still has room to
    fall, not yet oversold) + MACD bearish + volume confirm + ADX rising.
    The "any day you'd take this short" composite.
    """
    return df[
        (df["close"] < df["ema_200"]) &
        (df["close"] < df["sma_50"]) &
        (df["rsi_14"].between(30, 50)) &
        (df["macd"] < df["macd_signal"]) &
        (df["volume_ratio"] > 1.3) &
        (df["adx"] > 22)
    ].sort_values(["volume_ratio", "rsi_14"], ascending=[False, True])


def _filter_ma_stack_bearish(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 63: MA Stack Bearish — price < EMA21 < SMA50 < SMA200.

    Mirror of #54: textbook downtrend stacking. When MAs descend with
    price below them, every rally to a MA tends to get sold. The cleanest
    short-trade universe.
    """
    return df[
        (df["close"] < df["ema_21"]) &
        (df["ema_21"] < df["sma_50"]) &
        (df["sma_50"] < df["sma_200"]) &
        (df["change_pct"] < 1.0)   # not in active short squeeze
    ].sort_values("rsi_14", ascending=True)


def _filter_fresh_downtrend_start(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 64: Fresh Downtrend Start — ADX rising from quiet zone.

    Mirror of #56: "first big bar" of a new downtrend. ADX cycling up
    (<30, still early) + price below EMA50 + meaningful red bar with
    volume = regime change. Weeks of runway on the short side.
    """
    return df[
        (df["adx"] > 20) &
        (df["adx"] < 30) &
        (df["close"] < df["ema_21"]) &
        (df["close"] < df["sma_50"]) &
        (df["change_pct"] < -1.5) &
        (df["volume_ratio"] > 1.2)
    ].sort_values("adx", ascending=True)


def _filter_overbought_rejection_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 65: Overbought Rejection — RSI>65 but below SMA200.

    Mirror of #57: RSI extreme in a structural DOWNTREND. Counter-trend
    rallies that hit RSI 65+ while still below the long-term MA are the
    highest-probability fade setups for short entries.
    """
    return df[
        (df["rsi_14"] > 65) &
        (df["close"] < df["sma_200"]) &
        (df["volume_ratio"] > 1.5)
    ].sort_values("rsi_14", ascending=False)


def _filter_breakdown_with_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 66: Breakdown w/ Volume — 52w low + vol 2×.

    Mirror of #58: new 52-week low confirmed by volume. Classic Stan
    Weinstein Stage-4 entry — the fresh-breakdown short, with volume
    vouching for distribution (not capitulation low).
    """
    return df[
        (df["close"] <= df["low_52w"] * 1.01) &
        (df["volume_ratio"] > 2.0) &
        (df["change_pct"] < -1.5)
    ].sort_values("volume_ratio", ascending=False)


def _filter_rally_to_ema21_short(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 67: Rally to EMA21 — downtrend bounce into resistance.

    Mirror of #59: stock in a confirmed downtrend that rallied today
    into its EMA21 (a textbook short-the-rip level). RSI 45-60 means
    the bounce isn't exhausted yet — there's still air for the fade.
    """
    pct_from_ema = (df["ema_21"] - df["close"]) / df["close"]
    return df[
        (df["close"] < df["sma_50"]) &
        (df["sma_50"] < df["sma_200"]) &
        (pct_from_ema.between(-0.01, 0.02)) &
        (df["rsi_14"].between(45, 60))
    ].sort_values("change_pct", ascending=False)


def _filter_bollinger_squeeze_release_short(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 68: BB Squeeze Release Short — tight range broke DOWN.

    Mirror of #60: BB width was compressed, today price tagged or
    crossed the LOWER band on volume while below SMA50. The bearish
    counterpart to the long squeeze-release breakout.
    """
    bb_width = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    return df[
        (bb_width > 0) &
        (df["close"] < df["bb_lower"] * 1.01) &
        (df["volume_ratio"] > 1.5) &
        (df["close"] < df["sma_50"]) &
        (bb_width < 0.10)
    ].sort_values("volume_ratio", ascending=False)


def _filter_relative_strength_laggard(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 69: RS Laggard — underperforming the market AND own sector.

    Mirror of #61: stock down significantly more than market average,
    AND below SMA50. Pair-trade material: short the laggard against
    a long in the leader (#61) for a clean RS spread.
    """
    if df.empty:
        return df
    median_change = df["change_pct"].median()
    threshold = min(-2.0, median_change - 2.0)
    return df[
        (df["change_pct"] < threshold) &
        (df["close"] < df["sma_50"]) &
        (df["rsi_14"] < 50) &
        (df["volume_ratio"] > 1.2)
    ].sort_values("change_pct", ascending=True)


def _filter_bear_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 70: Bear Momentum — strong sustained downside.

    Mirror of #17 (Bull Momentum). Down >3% today, RSI<35, MACD<signal,
    volume confirmed. The "running short" filter — entry on retracement
    to EMA21 is the textbook play.
    """
    return df[
        (df["change_pct"] < -3.0) &
        (df["rsi_14"] < 35) &
        (df["macd"] < df["macd_signal"]) &
        (df["volume_ratio"] > 1.5) &
        (df["close"] < df["ema_21"])
    ].sort_values("change_pct", ascending=True)


def _filter_momentum_crash(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 71: Momentum Crash — sudden bearish acceleration.

    Mirror of #30 (Momentum Burst). Today's down move is 2.5×+ the 20-day
    average bar size, with breaking volume. Catches gap-down + sell-off
    days — the kind that mark the start of a multi-day slide.
    """
    if "atr_14" not in df.columns:
        return pd.DataFrame()
    bar_size_pct = (df["close"] - df["close"].shift(1)).abs() / df["close"]
    atr_pct = df["atr_14"] / df["close"]
    return df[
        (df["change_pct"] < -2.0) &
        (bar_size_pct > atr_pct * 1.5) &
        (df["volume_ratio"] > 2.0) &
        (df["close"] < df["ema_21"])
    ].sort_values("change_pct", ascending=True)


# =============================================================================
# PR-S18 — INSTITUTIONAL-GRADE SETUP SCANNERS (72..86)
# Each setup verified against a primary author source (book/blog/paper).
# 10 swing (72-81) + 5 positional (82-86). Bullish unless otherwise tagged.
#
# Where the full multi-bar formula needs history we don't carry in
# summary_df, we use a "best-effort approximation" — the indicator
# columns added in ml/features/indicators.py:_compute_screener_indicators
# precompute the relevant signal so each filter remains a single-row check.
# =============================================================================


def _filter_pocket_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 72: Pocket Pivot (Kacher/Morales).

    Source: Trade Like an O'Neil Disciple (Wiley 2010, Ch. 4); ChartMill spec.
    Rule: close >= EMA(10) AND close > EMA(50) AND today's volume > MAX of any
    down-close volume in last 10 days. Optional: not extended >5% above 10-EMA.
    """
    if "ema_10" not in df.columns:
        return pd.DataFrame()
    not_extended = (df["close"] - df["ema_10"]) / df["ema_10"] <= 0.05
    return df[
        (df["close"] >= df["ema_10"]) &
        (df["close"] > df["sma_50"]) &
        (df["pocket_pivot_volume"]) &
        (df["sma_50"] > df["sma_200"]) &
        not_extended
    ].sort_values("volume_ratio", ascending=False)


def _filter_wyckoff_spring(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 73: Wyckoff Spring (Phase C).

    Source: StockCharts Wyckoff Method; Wyckoff Analytics. Today's low
    penetrated 60-day support by ≤0.5% but close reclaimed back above it
    on lower-than-avg volume = classic spring shakeout.
    """
    return df[
        (df["low"] < df["low_52w"] * 1.005) &     # penetrated near 60d/52w low
        (df["low"] < df["sma_50"] * 0.98) &        # below 50-MA briefly
        (df["close"] > df["low"] * 1.005) &        # reclaimed back up
        (df["close"] > df["low_52w"]) &
        (df["volume_ratio"].between(0.6, 1.4))     # not panic, not explosive
    ].sort_values("change_pct", ascending=False)


def _filter_episodic_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 74: Episodic Pivot (Pradeep Bonde / Qullamaggie).

    Source: Stockbee blog; Qullamaggie 'How to Master Episodic Pivots'.
    Gap ≥8% on day-of with volume ≥2× SMA(50) and new 60-day high.
    NOTE: catalyst confirmation (news/earnings) must be done downstream.
    """
    if "gap_pct" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["gap_pct"] >= 8.0) &
        (df["volume_ratio"] >= 2.0) &
        (df["close"] > df["high_52w"] * 0.99) &     # near/at 52w highs (proxy for 60d high)
        (df["close"] > df["open"])                  # closed green, not gap-fade
    ].sort_values("gap_pct", ascending=False)


def _filter_holy_grail(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 75: Holy Grail (Linda Raschke).

    Source: Connors & Raschke, Street Smarts (1995). ADX ≥30 (strong trend) +
    today's bar touched EMA21 (pullback in trend) + RSI 40-55 (mild pullback).
    Buy on break above today's high.
    """
    return df[
        (df["adx"] >= 30) &
        (df["low"] <= df["ema_21"]) & (df["high"] >= df["ema_21"]) &
        (df["close"] > df["sma_50"]) &
        (df["sma_50"] > df["sma_200"]) &
        (df["rsi_14"].between(40, 55))
    ].sort_values("adx", ascending=False)


def _filter_coiled_spring(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 76: Coiled Spring — BB-width compressed + NR7 + near highs.

    Source: Crabel, Day Trading with Short-Term Price Patterns (1990); StockCharts NR7.
    BB-width 60-bar percentile rank <25 + NR7 + close within 5% of 52w high +
    above all MAs = setup before a multi-week expansion.
    """
    if "bb_width_pct_rank_60" not in df.columns or "nr7" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["bb_width_pct_rank_60"] < 0.25) &
        (df["nr7"]) &
        (df["close"] >= df["high_52w"] * 0.95) &
        (df["close"] > df["ema_21"]) &
        (df["close"] > df["sma_50"])
    ].sort_values("bb_width_pct_rank_60", ascending=True)


def _filter_inside_after_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 77: Inside Day After Trend Day (Crabel).

    Source: Crabel (1990), 'Inside Days' chapter. Yesterday was a trend day
    (range ≥1.5×ATR20, body ≥70% of range, closed at extreme). Today is an
    inside bar. Continuation trade in prior trend direction.
    """
    if "prior_trend_day_up" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["prior_trend_day_up"]) &
        (df["inside_bar"])
    ].sort_values("change_pct", ascending=False)


def _filter_three_soldiers_after_pullback(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 78: Three White Soldiers After Pullback (Nison + Minervini overlay).

    Source: Nison, Japanese Candlestick Charting Techniques (2001); Minervini,
    Trade Like a Stock Market Wizard (2013). Three green closes + RSI <70 +
    today's volume > yesterday > SMA(20) + uptrend intact.
    """
    if "three_green" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["three_green"]) &
        (df["rsi_14"] < 70) &
        (df["close"] > df["sma_50"]) &
        (df["volume_ratio"] > 1.2) &
        (df["close"] > df["ema_21"])
    ].sort_values("change_pct", ascending=False)


def _filter_dragonfly_at_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 79: Dragonfly Doji at Key MA.

    Source: Nison (2001); Bulkowski thepatternsite.com; Quantified Strategies.
    Dragonfly doji touching EMA21 or SMA50 from above, in an uptrend
    (close > SMA200), with RSI <45 (pullback context). Awaits confirmation.
    """
    if "dragonfly_doji" not in df.columns:
        return pd.DataFrame()
    touched_ema21 = (df["low"] <= df["ema_21"]) & (df["high"] >= df["ema_21"])
    touched_sma50 = (df["low"] <= df["sma_50"]) & (df["high"] >= df["sma_50"])
    return df[
        (df["dragonfly_doji"]) &
        (touched_ema21 | touched_sma50) &
        (df["close"] > df["sma_200"]) &
        (df["rsi_14"] < 45)
    ].sort_values("rsi_14", ascending=True)


def _filter_gap_fill_reversal(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 80: Gap Fill Reversal (Exhaustion Gap Fade).

    Source: Murphy, Technical Analysis of the Financial Markets (1999), Ch.4;
    Trade That Swing SPY study (gap-fill base rates). Big up-gap in an
    extended uptrend that closed BELOW open with volume = potential exhaustion.
    BEARISH bias — short the failure.
    """
    if "gap_pct" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["gap_pct"] >= 3.0) &
        (df["close"] < df["open"]) &
        (df["volume_ratio"] > 1.5) &
        (df["close"] >= df["high_52w"] * 0.92) &     # was already extended
        (df["rsi_14"] > 65)                          # overbought
    ].sort_values("gap_pct", ascending=False)


def _filter_weekly_pivot_reclaim(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 81: Weekly Pivot Reclaim (Ochoa/Varsity CPR).

    Source: Zerodha Varsity 'Central Pivot Range' chapter; Ochoa,
    Secrets of a Pivot Boss (2010). Prior close was below CPR_BC, today
    closes above CPR_TC on volume. Narrow CPR (≤0.5% wide) = explosive bias.
    """
    if "cpr_tc" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["prev_close"] < df["cpr_bc"]) &
        (df["close"] > df["cpr_tc"]) &
        (df["volume_ratio"] > 1.2) &
        (df["close"] > df["sma_50"])
    ].sort_values("volume_ratio", ascending=False)


# ---------- POSITIONAL (82-86) ----------


def _filter_stage_2_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 82: Stage 2 Acceleration (Weinstein).

    Source: Stan Weinstein, Secrets for Profiting in Bull and Bear Markets
    (McGraw-Hill 1988). Close > 150D SMA (~30W) AND 150D SMA rising AND
    new 52w high AND volume ≥1.5× SMA50. Fresh stage-2 breakout.
    """
    if "sma_150" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["close"] > df["sma_150"]) &
        (df["sma_150_rising"]) &
        (df["close"] >= df["high_52w"] * 0.99) &
        (df["volume_ratio"] >= 1.5) &
        (df["close"] > df["sma_50"])
    ].sort_values("volume_ratio", ascending=False)


def _filter_canslim_base_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 83: CAN SLIM Base Breakout (William O'Neil).

    Source: O'Neil, How to Make Money in Stocks 4e (2009). 60-bar base with
    ≤15% depth (flat base) + breakout to new 60-bar high + volume ≥1.5×
    SMA(50). Best-effort: full CAN SLIM requires fundamentals (EPS, ROE)
    which the screener doesn't carry — gate via base + breakout + volume only.
    """
    if "base_height_pct_60" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["base_height_pct_60"] <= 0.15) &           # flat base
        (df["close"] >= df["high_52w"] * 0.99) &       # new highs
        (df["volume_ratio"] >= 1.5) &
        (df["close"] > df["sma_50"]) &
        (df["sma_50"] > df["sma_200"])
    ].sort_values("volume_ratio", ascending=False)


def _filter_cup_handle_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 84: Cup-and-Handle Volume Pocket (O'Neil).

    Source: O'Neil, How to Make Money in Stocks (2009) Ch.2; StockCharts.
    Full cup-handle detection lives in the v2 Pattern Scanner (ml/features/
    patterns.py). This summary-row variant looks for the *volume signature*:
    base depth 12-33% + low volume in last 5 bars + breakout w/ vol surge.
    """
    if "base_height_pct_60" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["base_height_pct_60"].between(0.12, 0.33)) &
        (df["close"] >= df["high_52w"] * 0.98) &
        (df["volume_ratio"] >= 1.5) &
        (df["close"] > df["open"]) &
        (df["sma_50"] > df["sma_200"])
    ].sort_values("base_height_pct_60", ascending=True)


def _filter_pead_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 85: Post-Earnings-Drift Proxy (Bernard & Thomas 1989).

    Source: Bernard & Thomas, Journal of Accounting Research 1989; Caltech
    PEAD survey. Without an earnings consensus feed we use the price-based
    surprise proxy noted in the literature: |return| > 5% with vol > 3× avg.
    Catches earnings-driven jumps that exhibit drift over the next 60 days.
    """
    return df[
        (df["change_pct"].abs() > 5.0) &
        (df["volume_ratio"] > 3.0) &
        (df["close"] > df["sma_50"])              # drift only works in uptrend
    ].sort_values("change_pct", ascending=False)


def _filter_long_unwinding(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 87: Long Unwinding — price ↓ + OI ↓ (trend-exhaustion).

    Source: Zerodha Varsity Open Interest chapter. Longs covering positions
    is *less sustainable* than fresh short buildup — momentum often fades.
    Surface as 'trend-exhaustion warning', not an entry signal.

    Joins NSE participant OI data (via nse_data) with the per-symbol
    summary_df so we only flag F&O-eligible symbols actually trading down.
    """
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        oi_data = nse.get_oi_spurts()
        spurts = oi_data.get("data", [])
        if not spurts:
            return pd.DataFrame()

        # Long unwinding signature: OI dropped >5% with price down
        unwinding_syms = [
            s["symbol"] for s in spurts
            if s.get("oi_change_pct", 0) < -5.0
        ]
        if not unwinding_syms or df.empty:
            return pd.DataFrame()

        if df.index.name == "symbol":
            sel = df[df.index.isin(unwinding_syms)]
        else:
            sel = df[df["symbol"].isin(unwinding_syms)]
        # Require actual price-down today to confirm the long unwind
        sel = sel[sel["change_pct"] < -0.5]
        return sel.sort_values("change_pct", ascending=True)
    except Exception as e:
        logger.debug("Long unwinding scanner: %s", e)
        return pd.DataFrame()


def _filter_oi_spike(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 88: OI Spike — single-symbol OI change ≥ 20%.

    Source: StockGro / Jainam 'OI Spurts' classification — a 20%+ single-
    session OI jump on a single F&O symbol is institutional footprint.
    Direction (long vs short positioning) inferred from same-day price.

    Without OI provider data this returns empty (per no-fallbacks lock).
    """
    try:
        from .nse_data import get_nse_data
        nse = get_nse_data()
        oi_data = nse.get_oi_spurts()
        spurts = oi_data.get("data", [])
        if not spurts:
            return pd.DataFrame()

        spike_syms = [s["symbol"] for s in spurts if abs(s.get("oi_change_pct", 0)) >= 20.0]
        if not spike_syms or df.empty:
            return pd.DataFrame()

        if df.index.name == "symbol":
            sel = df[df.index.isin(spike_syms)]
        else:
            sel = df[df["symbol"].isin(spike_syms)]
        return sel.sort_values("volume_ratio", ascending=False)
    except Exception as e:
        logger.debug("OI spike scanner: %s", e)
        return pd.DataFrame()


def _filter_channel_mid_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """Scanner 86: Ascending Channel Mid-Reversion (Murphy/Pring).

    Source: Murphy, Technical Analysis of the Financial Markets (1999) Ch.4;
    Pring, Technical Analysis Explained 5e (2014) Ch.5-6. 60-bar regression
    slope is positive (ascending channel), price is near the lower edge,
    RSI <40 = mean-revert long entry inside the channel.
    """
    if "channel_slope_60" not in df.columns:
        return pd.DataFrame()
    return df[
        (df["channel_slope_60"] > 0) &              # ascending channel
        (df["rsi_14"] < 40) &                       # near bottom of channel
        (df["close"] > df["sma_200"]) &             # long-term uptrend intact
        (df["close"] > df["low_10d"])               # didn't break lower
    ].sort_values("rsi_14", ascending=True)


# Map scanner ID to filter function
SCANNER_FILTERS: Dict[int, Callable] = {
    0: _filter_full_screening,
    1: _filter_breakout_consolidation,
    2: _filter_top_gainers,
    3: _filter_top_losers,
    4: _filter_volume_breakout,
    5: _filter_52w_high,
    6: _filter_10d_high,
    7: _filter_52w_low,
    8: _filter_volume_surge,
    9: _filter_rsi_oversold,
    10: _filter_rsi_overbought,
    11: _filter_ma_crossover,
    12: _filter_bullish_engulfing,
    13: _filter_bearish_engulfing,
    14: _filter_vcp,
    15: _filter_bull_cross,
    16: _filter_ipo_breakout,
    17: _filter_bull_momentum,
    18: _filter_atr_trailing,
    19: _filter_psar_reversal,
    # 20 (ORB) removed 2026-05-31 — needs intraday data we don't ingest yet
    21: _filter_nr4,
    22: _filter_nr7,
    # 23-25 (Cup&Handle / Double Bottom / Inv H&S) removed 2026-05-31 —
    # served by v2 Pattern Scanner (/api/screener/patterns/v2/scan)
    26: _filter_macd_crossover,
    27: _filter_macd_bearish,
    28: _filter_inside_bar,
    29: _filter_ttm_squeeze,
    30: _filter_momentum_burst,
    31: _filter_trend_template,
    32: _filter_supertrend,
    33: _filter_pivot_breakout,
    34: _filter_high_delivery,              # Delivery % — real NSE data
    35: _filter_bulk_deals,                 # Bulk deals — real NSE data
    36: _filter_fii_net_buyers,             # FII buying — real NSE data
    37: _filter_dii_net_buyers,             # DII buying — real NSE data
    38: _filter_institutional_combined,     # Combined institutional — real NSE data
    39: _filter_oi_analysis,                # OI analysis — real NSE data
    40: _filter_long_buildup,              # F&O Long buildup — real NSE data
    41: _filter_short_buildup,             # F&O Short buildup — real NSE data
    42: _filter_short_covering,            # F&O Short covering — real NSE data
    # IDs 43-47, 49-51 (chart-pattern scanners) removed 2026-05-31 — replaced
    # by v2 Pattern Scanner. 48 (High & Tight Flag) stays as a pure indicator.
    48: _filter_high_tight_flag,
    # PR-S9 — 10 cutting-edge scanners (high-conviction trader setups)
    52: _filter_power_setup,
    53: _filter_squeeze_release,
    54: _filter_ma_stack_bullish,
    55: _filter_pre_breakout_coil,
    56: _filter_fresh_trend_start,
    57: _filter_oversold_bounce_setup,
    58: _filter_breakout_with_volume,
    59: _filter_pullback_to_ema21,
    60: _filter_bollinger_squeeze_release,
    61: _filter_relative_strength_leader,
    # PR-S17 — 10 bearish-counterpart scanners (short-side parity)
    62: _filter_power_setup_short,
    63: _filter_ma_stack_bearish,
    64: _filter_fresh_downtrend_start,
    65: _filter_overbought_rejection_setup,
    66: _filter_breakdown_with_volume,
    67: _filter_rally_to_ema21_short,
    68: _filter_bollinger_squeeze_release_short,
    69: _filter_relative_strength_laggard,
    70: _filter_bear_momentum,
    71: _filter_momentum_crash,
    # PR-S18 — 10 institutional swing setups (verified algorithms)
    72: _filter_pocket_pivot,
    73: _filter_wyckoff_spring,
    74: _filter_episodic_pivot,
    75: _filter_holy_grail,
    76: _filter_coiled_spring,
    77: _filter_inside_after_trend,
    78: _filter_three_soldiers_after_pullback,
    79: _filter_dragonfly_at_ma,
    80: _filter_gap_fill_reversal,
    81: _filter_weekly_pivot_reclaim,
    # PR-S18 — 5 positional setups
    82: _filter_stage_2_acceleration,
    83: _filter_canslim_base_breakout,
    84: _filter_cup_handle_volume,
    85: _filter_pead_proxy,
    86: _filter_channel_mid_reversion,
    # PR-S20 — per-stock F&O scanners (require NSE participant OI data)
    87: _filter_long_unwinding,
    88: _filter_oi_spike,
}

# Pattern-scanner dispatch maps (PATTERN_SCANNERS / _CONSOLIDATION_MAP /
# _CONSOLIDATION_LABELS / _REVERSAL_MAP / _REVERSAL_LABELS) removed
# 2026-05-31 along with the legacy chart-pattern scanner IDs. The v2
# Pattern Scanner (services.chart_patterns) supersedes them entirely.
