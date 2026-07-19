"""Feature extraction for outcome models — PR-DEPTH.

When a strategy fires entry, we need to capture the MARKET STATE at that
moment as a feature vector. Later, when the trade closes, we'll write
{features_at_entry, won} to strategy_outcomes. The trainer reads those
rows and learns: "given features X, does this strategy win?"

Features extracted from OHLCV bars (calls reuse our existing indicator
computations from ai/strategy/indicators.py):
  rsi14, ema8, ema21, ema50, adx, atr, vwap, bbands position,
  macd_hist, volume_ratio, regime, vix, day_of_week, hour, session_pct
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# These are the keys we emit. Trainers read them back from JSONB at training time.
OUTCOME_FEATURE_KEYS = (
    "close", "rsi14", "ema8", "ema21", "ema50",
    "adx", "atr", "macd_hist",
    "bbands_position",        # (close - bbands_lower) / (bbands_upper - bbands_lower)
    "volume_ratio",           # current vol / SMA20 vol
    "close_minus_vwap_pct",
    "regime", "vix",
    "day_of_week", "hour",
)


def build_outcome_features(
    bars: pd.DataFrame,
    *,
    regime: Optional[str] = None,
    vix: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a feature vector from the LAST bar of OHLCV. Returns a dict
    safe to JSONB-serialise (no numpy types)."""
    if bars is None or bars.empty:
        return {}

    try:
        from ..strategy.indicators import compute_indicator
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    # Full feature set — borrowed from aaryansinha16's 50-feature contract
    # plus our additions. NaN-safe: feature simply absent if computation
    # returns NaN. Tree models handle missing values natively.
    feature_indicators = (
        # Existing
        "close", "rsi14", "ema8", "ema21", "ema50",
        "adx", "atr", "macd_hist",
        # PR-FEATURES: momentum additions
        "rsi7", "rsi9", "stoch_rsi_k", "stoch_rsi_d",
        "williams_r", "mfi", "cci",
        "roc_10", "roc_20",
        # PR-FEATURES: trend additions
        "di_plus", "di_minus", "supertrend", "psar",
        "ema13", "ema100", "ema200",
        "sma20", "sma50", "sma200",
        "macd", "macd_signal",
        # PR-FEATURES: volatility additions
        "volatility_20", "volatility_60", "volatility_regime",
        "bbands_upper", "bbands_middle", "bbands_lower",
        # PR-FEATURES: volume additions
        "obv", "obv_slope", "volume_ratio", "volume_delta_20",
        "vwap", "vwap_distance_pct",
        # PR-FEATURES: session features (mostly 0/375 on daily bars but
        # meaningful for intraday strategies)
        "minutes_since_open", "session_progress",
        "is_first_hour", "is_last_hour",
    )
    for key in feature_indicators:
        try:
            v = compute_indicator(key, bars)
            if v is not None and not (isinstance(v, float) and (v != v)):  # not NaN
                out[key] = float(v)
        except Exception:
            pass

    # bbands_position — where in the band the current close sits
    try:
        bb_up = compute_indicator("bbands_upper", bars)
        bb_lo = compute_indicator("bbands_lower", bars)
        last_close = float(bars["close"].iloc[-1])
        if bb_up and bb_lo and bb_up > bb_lo:
            out["bbands_position"] = float((last_close - bb_lo) / (bb_up - bb_lo))
    except Exception:
        pass

    # volume_ratio
    try:
        if "volume" in bars.columns and len(bars) >= 20:
            current_vol = float(bars["volume"].iloc[-1])
            avg_vol = float(bars["volume"].tail(20).mean())
            if avg_vol > 0:
                out["volume_ratio"] = current_vol / avg_vol
    except Exception:
        pass

    # close_minus_vwap_pct
    try:
        vwap = compute_indicator("vwap", bars)
        last_close = float(bars["close"].iloc[-1])
        if vwap and vwap > 0:
            out["close_minus_vwap_pct"] = float((last_close - vwap) / vwap * 100)
    except Exception:
        pass

    if regime:
        # Encode regime as one-of-N for tree models
        out["regime_bull"] = 1.0 if regime == "bull" else 0.0
        out["regime_sideways"] = 1.0 if regime == "sideways" else 0.0
        out["regime_bear"] = 1.0 if regime == "bear" else 0.0

    if vix is not None:
        out["vix"] = float(vix)

    # Calendar features
    try:
        ts = bars.index[-1]
        if hasattr(ts, "weekday"):
            out["day_of_week"] = int(ts.weekday())
            out["hour"] = int(ts.hour) if hasattr(ts, "hour") else 9

            # Session % through trading day (NSE: 9:15-15:30 = 375 min)
            try:
                t = ts.time() if hasattr(ts, "time") else None
                if t:
                    mins_from_open = (t.hour - 9) * 60 + (t.minute - 15)
                    out["session_pct"] = max(0.0, min(1.0, mins_from_open / 375.0))
            except Exception:
                pass
    except Exception:
        pass

    return out
