"""
PRD-aligned feature engineering for Quant X.
Computes base technical features (LightGBM) and proxy features (TFT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import (
    OnBalanceVolumeIndicator,
    VolumeWeightedAveragePrice,
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    ForceIndexIndicator,
)


@dataclass
class FeatureConfig:
    base_window: int = 20
    rsi_window: int = 14
    atr_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vwap_window: int = 14


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def slope_fn(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        idx = np.arange(len(x))
        try:
            slope = np.polyfit(idx, x, 1)[0]
            return float(slope)
        except Exception:
            return 0.0
    return series.rolling(window).apply(lambda x: slope_fn(np.array(x)), raw=False)


def _consecutive_streak(values: pd.Series, mode: str) -> pd.Series:
    streak = []
    count = 0
    prev = None
    for v in values:
        if prev is None:
            count = 0
        else:
            if mode == "up":
                count = count + 1 if v > prev else 0
            else:
                count = count + 1 if v < prev else 0
        streak.append(count)
        prev = v
    return pd.Series(streak, index=values.index)


def compute_features(
    df: pd.DataFrame,
    benchmark_close: Optional[pd.Series] = None,
    cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Compute PRD-aligned features for LightGBM + TFT.
    Expects columns: open, high, low, close, volume (lowercase).
    """
    data = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    # Base indicators
    data["ema_9"] = EMAIndicator(data["close"], window=9).ema_indicator()
    data["ema_20"] = EMAIndicator(data["close"], window=20).ema_indicator()
    data["ema_21"] = EMAIndicator(data["close"], window=21).ema_indicator()
    data["ema_50"] = EMAIndicator(data["close"], window=50).ema_indicator()

    rsi = RSIIndicator(data["close"], window=cfg.rsi_window)
    data["rsi_14"] = rsi.rsi()

    macd = MACD(
        close=data["close"],
        window_slow=cfg.macd_slow,
        window_fast=cfg.macd_fast,
        window_sign=cfg.macd_signal,
    )
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_hist"] = macd.macd_diff()

    bb = BollingerBands(close=data["close"], window=20, window_dev=2)
    data["bb_upper"] = bb.bollinger_hband()
    data["bb_lower"] = bb.bollinger_lband()
    data["bb_mid"] = bb.bollinger_mavg()
    data["bb_percent"] = bb.bollinger_pband()
    data["bb_bandwidth"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"]

    atr = AverageTrueRange(
        high=data["high"],
        low=data["low"],
        close=data["close"],
        window=cfg.atr_window,
    )
    data["atr_14"] = atr.average_true_range()

    data["volume_sma_20"] = data["volume"].rolling(20).mean()
    data["volume_ratio"] = data["volume"] / data["volume_sma_20"]

    obv = OnBalanceVolumeIndicator(data["close"], data["volume"])
    data["obv"] = obv.on_balance_volume()

    vwap = VolumeWeightedAveragePrice(
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
        window=cfg.vwap_window,
    )
    data["vwap"] = vwap.volume_weighted_average_price()
    data["vwap_diff"] = data["close"] - data["vwap"]

    # Candle anatomy
    rng = (data["high"] - data["low"]).replace(0, np.nan)
    body = (data["close"] - data["open"]).abs()
    upper_wick = data["high"] - data[["open", "close"]].max(axis=1)
    lower_wick = data[["open", "close"]].min(axis=1) - data["low"]

    data["body_pct"] = (body / rng).fillna(0)
    data["wick_pct"] = ((upper_wick + lower_wick) / rng).fillna(0)

    # Macro pattern proxies (10)
    rolling_high_10 = data["close"].rolling(10).max()
    rolling_low_10 = data["close"].rolling(10).min()
    data["rolling_high_dist_10"] = (rolling_high_10 - data["close"]) / data["close"]
    data["rolling_low_dist_10"] = (data["close"] - rolling_low_10) / data["close"]
    data["slope_10"] = _rolling_slope(data["close"], 10)
    data["slope_change"] = data["slope_10"] - data["slope_10"].shift(5)

    rolling_high_20 = data["close"].rolling(20).max()
    rolling_low_20 = data["close"].rolling(20).min()
    touch_low = (data["close"] <= rolling_low_20 * 1.01).astype(int)
    touch_high = (data["close"] >= rolling_high_20 * 0.99).astype(int)
    data["support_touch_count"] = touch_low.rolling(20).sum()
    data["resistance_touch_count"] = touch_high.rolling(20).sum()
    data["channel_width"] = (rolling_high_20 - rolling_low_20) / rolling_low_20.replace(0, np.nan)
    data["price_in_channel"] = (data["close"] - rolling_low_20) / (rolling_high_20 - rolling_low_20)
    data["consecutive_higher_highs"] = _consecutive_streak(data["high"], mode="up")
    data["consecutive_lower_lows"] = _consecutive_streak(data["low"], mode="down")

    # Micro pattern proxies (8)
    data["upper_wick_ratio"] = (upper_wick / rng).fillna(0)
    data["lower_wick_ratio"] = (lower_wick / rng).fillna(0)
    data["body_direction"] = (data["close"] > data["open"]).astype(int)
    data["engulfing_score"] = (
        (body > body.shift(1)) & (data["body_direction"] != data["body_direction"].shift(1))
    ).astype(int)
    data["pin_bar_score"] = (
        (data[["upper_wick_ratio", "lower_wick_ratio"]].max(axis=1) > 0.6)
        & (data["body_pct"] < 0.3)
    ).astype(int)
    data["gap_pct"] = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)
    data["body_size_rank"] = body.rolling(20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )
    data["doji_score"] = (body < (rng * 0.1)).astype(int)

    # Regime features (8)
    adx = ADXIndicator(high=data["high"], low=data["low"], close=data["close"], window=14)
    data["adx"] = adx.adx()
    data["di_plus"] = adx.adx_pos()
    data["di_minus"] = adx.adx_neg()
    data["realized_vol"] = data["close"].pct_change().rolling(10).std()
    data["atr_ratio"] = data["atr_14"] / data["atr_14"].rolling(20).mean()
    data["bb_bandwidth"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"]

    if benchmark_close is not None:
        bench_ret = benchmark_close.pct_change()
        data["nifty_correlation_20"] = data["close"].pct_change().rolling(20).corr(bench_ret)
    else:
        data["nifty_correlation_20"] = 0

    # Regime label: 0=consolidating, 1=up, 2=down, 3=highvol
    def _regime(row) -> int:
        if row["adx"] < 20 and row["atr_ratio"] < 0.9:
            return 0
        if row["adx"] > 25 and row["di_plus"] > row["di_minus"]:
            return 1
        if row["adx"] > 25 and row["di_minus"] > row["di_plus"]:
            return 2
        return 3

    data["regime_label"] = data.apply(_regime, axis=1)

    # Volume features (8)
    data["vol_sma_ratio"] = data["volume_ratio"]
    data["obv_slope_10"] = _rolling_slope(data["obv"], 10)
    adi = AccDistIndexIndicator(
        high=data["high"], low=data["low"], close=data["close"], volume=data["volume"]
    )
    data["ad_line"] = adi.acc_dist_index()
    cmf = ChaikinMoneyFlowIndicator(
        high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], window=20
    )
    data["cmf_20"] = cmf.chaikin_money_flow()
    fi = ForceIndexIndicator(close=data["close"], volume=data["volume"], window=2)
    data["force_index"] = fi.force_index()
    data["vol_trend_days"] = _consecutive_streak(data["volume"], mode="up")
    data["above_vwap_vol_ratio"] = (data["close"] > data["vwap"]).astype(int)
    price_slope_5 = _rolling_slope(data["close"], 5)
    obv_slope_5 = _rolling_slope(data["obv"], 5)
    data["obv_divergence"] = (np.sign(price_slope_5) != np.sign(obv_slope_5)).astype(int)

    # Extra context
    data["previous_day_return"] = data["close"].pct_change()
    rolling_high_52 = data["close"].rolling(252).max()
    rolling_low_52 = data["close"].rolling(252).min()
    data["dist_52w_high"] = (rolling_high_52 - data["close"]) / data["close"]
    data["dist_52w_low"] = (data["close"] - rolling_low_52) / data["close"]

    return data


def build_feature_row(df: pd.DataFrame) -> Dict:
    """Extract the latest feature row as a dict."""
    if df.empty:
        return {}
    row = df.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0)
    return row.to_dict()


def split_feature_sets(row: Dict) -> Tuple[Dict, Dict]:
    """LEGACY shim — kept for callers that haven't migrated yet.

    Returns (lgbm_legacy_15, tft_row). DO NOT USE for the v2 LGBM gate;
    its feature schema is the 30-key set defined in
    ml.features.lgbm_v2.FEATURE_ORDER. For LGBM inference, call
    compute_lgbm_v2_features() below instead — it returns a dict with
    exactly the keys LGBMGate.predict() requires.
    """
    lgbm_keys = [
        "close",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "bb_percent",
        "ema_20",
        "ema_50",
        "atr_14",
        "volume_ratio",
        "obv",
        "vwap_diff",
        "body_pct",
        "wick_pct",
    ]
    lgbm = {k: row.get(k, 0) for k in lgbm_keys}
    tft = row
    return lgbm, tft


def compute_lgbm_v2_features(
    symbol: str,
    ohlcv_df: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
    *,
    flow_features_df: Optional[pd.DataFrame] = None,
    sentiment_features_df: Optional[pd.DataFrame] = None,
    fundamentals_features: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Live-inference wrapper for the v2 LGBM gate.

    Produces a dict with EXACTLY the 30 keys in
    ml.features.lgbm_v2.FEATURE_ORDER (+ Kronos block if enabled).
    Feeds LGBMGate.predict() at signal-generation time.

    Args:
        symbol: NSE symbol (no .NS suffix).
        ohlcv_df: daily OHLCV history. Yfinance shape — Close, High,
            Low, Volume capitalized.
        as_of_date: bar-close date to score. Defaults to the last
            available bar.
        flow_features_df: optional FII/DII flow features (zero-filled
            if absent).
        sentiment_features_df: optional per-symbol sentiment features
            (zero-filled if absent).
        fundamentals_features: optional fundamentals dict (zero-filled
            if absent).

    Returns:
        Dict[str, float] keyed exactly by FEATURE_ORDER.
    """
    # Local import — keeps the live service path light when this isn't called.
    from ml.features.lgbm_v2 import compute_inference_features  # noqa: PLC0415

    if as_of_date is None:
        if ohlcv_df.empty:
            raise ValueError("compute_lgbm_v2_features: ohlcv_df is empty")
        as_of_date = ohlcv_df.index[-1]
    if not isinstance(as_of_date, pd.Timestamp):
        as_of_date = pd.Timestamp(as_of_date)

    return compute_inference_features(
        symbol=symbol,
        ohlcv_df=ohlcv_df,
        as_of_date=as_of_date,
        flow_features_df=flow_features_df,
        sentiment_features_df=sentiment_features_df,
        fundamentals_features=fundamentals_features,
    )
