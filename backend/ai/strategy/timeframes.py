"""Timeframe → data-fetch + annualization config.

The strategy timeframe is user/LLM-decided (the DSL ``timeframe`` field), so
the backtest must adapt to *any* declared timeframe rather than a fixed menu:
which provider interval to request, how much history, how to annualize the
Sharpe ratio, and whether to resample (4h is built from 1h because no provider
serves it natively).

NSE cash session = 09:15–15:30 = 6h15m = 375 min/day; ~252 trading days/yr
(252 kept to match the legacy daily Sharpe annualization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .dsl import Timeframe

_TRADING_DAYS = 252
_SESSION_MIN = 375  # NSE cash-session minutes


@dataclass(frozen=True)
class TFConfig:
    """Everything the backtest needs to handle one timeframe."""

    timeframe: str            # canonical DSL value, e.g. "5m"
    fetch_interval: str       # provider interval to request ("5m"; "1h" for 4h)
    fetch_period: str         # provider history window ("60d" / "2y" / "max")
    bars_per_year: float      # Sharpe annualization factor (periods/year)
    resample_to: Optional[str] = None   # pandas offset if we must resample (4h←1h)
    intraday: bool = False


def _per_day(minutes: int) -> float:
    return _SESSION_MIN / minutes


# yfinance history limits drive fetch_period: 1m≈7d, 5–30m≈60d, 1h≈730d, 1d=max.
_CONFIG: Dict[str, TFConfig] = {
    "1m": TFConfig("1m", "1m", "7d", _per_day(1) * _TRADING_DAYS, None, True),
    "5m": TFConfig("5m", "5m", "60d", _per_day(5) * _TRADING_DAYS, None, True),
    "15m": TFConfig("15m", "15m", "60d", _per_day(15) * _TRADING_DAYS, None, True),
    "30m": TFConfig("30m", "30m", "60d", _per_day(30) * _TRADING_DAYS, None, True),
    "1h": TFConfig("1h", "1h", "2y", _per_day(60) * _TRADING_DAYS, None, True),
    # 4h: no provider serves it → fetch 1h and resample.
    "4h": TFConfig("4h", "1h", "2y", _per_day(240) * _TRADING_DAYS, "4h", False),
    "1d": TFConfig("1d", "1d", "max", float(_TRADING_DAYS), None, False),
}


def tf_config(timeframe) -> TFConfig:
    """Resolve a Timeframe enum / string to its TFConfig. Unknown → daily."""
    val = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
    return _CONFIG.get(val, _CONFIG["1d"])


def annualization_periods(timeframe) -> float:
    """Periods-per-year for Sharpe annualization at this timeframe."""
    return tf_config(timeframe).bars_per_year


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate a lower-interval OHLCV frame up to ``rule`` (e.g. "4h").

    Standard OHLCV aggregation: open=first, high=max, low=min, close=last,
    volume=sum. Empty buckets (off-session gaps) are dropped.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = {c.lower(): c for c in df.columns}
    work = df.rename(columns={cols[k]: k for k in agg if k in cols})
    out = work.resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


__all__ = ["TFConfig", "tf_config", "annualization_periods", "resample_ohlcv"]
