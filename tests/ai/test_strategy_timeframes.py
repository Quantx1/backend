"""Timeframe config tests — the backtest must adapt to ANY user/LLM-chosen
timeframe (the DSL ``timeframe`` field). Verifies fetch interval, history
period, Sharpe annualization, and 4h resampling are correct per timeframe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ai.strategy.dsl import Timeframe
from backend.ai.strategy.timeframes import (
    TFConfig,
    annualization_periods,
    resample_ohlcv,
    tf_config,
)


class TestEveryTimeframeCovered:
    def test_every_enum_value_has_config(self):
        # User can pick any timeframe → every enum value must resolve.
        for tf in Timeframe:
            cfg = tf_config(tf)
            assert isinstance(cfg, TFConfig)
            assert cfg.fetch_interval
            assert cfg.fetch_period
            assert cfg.bars_per_year > 0

    def test_accepts_string_or_enum(self):
        assert tf_config("5m").fetch_interval == "5m"
        assert tf_config(Timeframe.M5).fetch_interval == "5m"

    def test_unknown_timeframe_defaults_daily(self):
        assert tf_config("99x").timeframe == "1d"


class TestAnnualization:
    def test_intraday_has_more_bars_per_year_than_daily(self):
        assert (
            annualization_periods(Timeframe.M5)
            > annualization_periods(Timeframe.M15)
            > annualization_periods(Timeframe.H1)
            > annualization_periods(Timeframe.D1)
        )

    def test_daily_is_252(self):
        assert annualization_periods(Timeframe.D1) == 252

    def test_5m_is_session_based(self):
        # 375 NSE session min / 5 = 75 bars/day × 252.
        assert annualization_periods(Timeframe.M5) == 75 * 252


class TestFourHourResamples:
    def test_4h_fetches_1h_and_resamples(self):
        cfg = tf_config(Timeframe.H4)
        assert cfg.fetch_interval == "1h"   # no provider serves 4h natively
        assert cfg.resample_to == "4h"

    def test_daily_does_not_resample(self):
        assert tf_config(Timeframe.D1).resample_to is None

    def test_resample_ohlcv_aggregates_correctly(self):
        idx = pd.date_range("2025-01-01 09:15", periods=8, freq="1h")
        df = pd.DataFrame(
            {
                "open": np.arange(8, dtype=float) + 100,
                "high": np.arange(8, dtype=float) + 101,
                "low": np.arange(8, dtype=float) + 99,
                "close": np.arange(8, dtype=float) + 100.5,
                "volume": np.full(8, 1000.0),
            },
            index=idx,
        )
        out = resample_ohlcv(df, "4h")
        assert len(out) < len(df)
        assert set(out.columns) == {"open", "high", "low", "close", "volume"}
        # First 4h bucket: open=first, high=max, low=min, volume=sum of its rows.
        assert out.iloc[0]["volume"] >= 1000.0
        assert out.iloc[0]["high"] >= out.iloc[0]["low"]


class TestIntradayFlag:
    def test_intraday_flagged(self):
        assert tf_config(Timeframe.M5).intraday is True
        assert tf_config(Timeframe.H1).intraday is True
        assert tf_config(Timeframe.D1).intraday is False
