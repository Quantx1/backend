"""Data-source helpers for the screener.

Pure functions that turn raw OHLCV + indicator dataframes into screener
summary rows. No I/O — the broker fetch lives in ``engine._fetch_via_kite``
because it needs the engine's logger/state. We just consume what it
hands us.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def _safe_bool(val, default: bool = False) -> bool:
    try:
        if pd.isna(val):
            return default
        return bool(val)
    except (TypeError, ValueError):
        return default


def extract_summary_row(
    symbol: str, df: pd.DataFrame, last: pd.Series,
) -> Optional[Dict]:
    """Extract a summary row from the last bar of computed indicators.

    Returns ``None`` if the row can't be built (zero close, NaNs, etc.).
    The schema here is the contract with every scanner filter — adding
    a column requires updating both producer and consumer.
    """
    try:
        close = float(last['close'])
        if close <= 0 or pd.isna(close):
            return None

        prev_close = float(last.get('prev_close', close))
        if pd.isna(prev_close) or prev_close <= 0:
            prev_close = close

        change_pct = float(last.get('change_pct', 0))
        if pd.isna(change_pct):
            change_pct = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0

        return {
            'symbol': symbol,
            'close': close,
            'open': _safe_float(last.get('open', close)),
            'high': _safe_float(last.get('high', close)),
            'low': _safe_float(last.get('low', close)),
            'volume': _safe_float(last.get('volume', 0)),
            'prev_close': prev_close,
            'change_pct': round(change_pct, 2),
            # Indicators
            'rsi_14': _safe_float(last.get('rsi_14'), 50),
            'macd': _safe_float(last.get('macd')),
            'macd_signal': _safe_float(last.get('macd_signal')),
            'macd_hist': _safe_float(last.get('macd_hist')),
            'ema_9': _safe_float(last.get('ema_9')),
            'ema_21': _safe_float(last.get('ema_21')),
            'ema_200': _safe_float(last.get('ema_200')),
            'sma_20': _safe_float(last.get('sma_20')),
            'sma_50': _safe_float(last.get('sma_50')),
            'sma_200': _safe_float(last.get('sma_200')),
            'adx': _safe_float(last.get('adx')),
            'atr_14': _safe_float(last.get('atr_14')),
            'bb_upper': _safe_float(last.get('bb_upper')),
            'bb_lower': _safe_float(last.get('bb_lower')),
            'volume_ratio': _safe_float(last.get('volume_ratio'), 1.0),
            'golden_cross': _safe_bool(last.get('golden_cross')),
            # Screener indicators
            'high_52w': _safe_float(last.get('high_52w', close)),
            'low_52w': _safe_float(last.get('low_52w', close)),
            'high_10d': _safe_float(last.get('high_10d', close)),
            'low_10d': _safe_float(last.get('low_10d', close)),
            'daily_range': _safe_float(last.get('daily_range')),
            'nr4': _safe_bool(last.get('nr4')),
            'nr7': _safe_bool(last.get('nr7')),
            'inside_bar': _safe_bool(last.get('inside_bar')),
            'pivot_r1': _safe_float(last.get('pivot_r1')),
            'pivot_s1': _safe_float(last.get('pivot_s1')),
            'supertrend_direction': _safe_float(last.get('supertrend_direction'), 0),
            'psar_bullish': _safe_bool(last.get('psar_bullish')),
            'ttm_squeeze': _safe_bool(last.get('ttm_squeeze')),
            'atr_trailing_stop': _safe_float(last.get('atr_trailing_stop')),
            # Candlestick patterns
            'candle_engulfing_bull': _safe_bool(last.get('candle_engulfing_bull')),
            'candle_engulfing_bear': _safe_bool(last.get('candle_engulfing_bear')),
            'candle_hammer': _safe_bool(last.get('candle_hammer')),
            'candle_morning_star': _safe_bool(last.get('candle_morning_star')),
            'candle_doji': _safe_bool(last.get('candle_doji')),
            # PR-S18 institutional setup columns (scanners 72-86). These are
            # precomputed in indicators._compute_screener_indicators but were
            # dropped here, so every column-gated filter (`if "ema_10" not in
            # df.columns: return empty`) silently returned nothing. Forward them
            # so the 12 institutional scanners actually run. NaN-safe defaults
            # for the two percentile/ratio columns are set so an insufficient-
            # history row (NaN) can't false-positive the "compression"/"flat
            # base" filters (both test `< threshold`); the rest default low so
            # their `>= threshold` / `== True` guards exclude NaN rows.
            'ema_10': _safe_float(last.get('ema_10')),
            'sma_150': _safe_float(last.get('sma_150')),
            'sma_150_rising': _safe_bool(last.get('sma_150_rising')),
            'pocket_pivot_volume': _safe_bool(last.get('pocket_pivot_volume')),
            'gap_pct': _safe_float(last.get('gap_pct')),
            'bb_width_pct_rank_60': _safe_float(last.get('bb_width_pct_rank_60'), 1.0),
            'prior_trend_day_up': _safe_bool(last.get('prior_trend_day_up')),
            'base_height_pct_60': _safe_float(last.get('base_height_pct_60'), 1.0),
            'channel_slope_60': _safe_float(last.get('channel_slope_60')),
            'three_green': _safe_bool(last.get('three_green')),
            'dragonfly_doji': _safe_bool(last.get('dragonfly_doji')),
            'cpr_bc': _safe_float(last.get('cpr_bc')),
            'cpr_tc': _safe_float(last.get('cpr_tc')),
        }
    except Exception as e:
        logger.debug(f"Summary extraction failed for {symbol}: {e}")
        return None
