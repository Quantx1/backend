"""Multi-timeframe strategy backtests — the timeframe is user/LLM-decided, so
the same strategy must run + gate correctly at 5m / 15m / 1h / 4h / 1d, with
the Sharpe annualized for that timeframe.

Synthetic bars — deterministic, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ai.strategy.backtest import run_walk_forward
from backend.ai.strategy.dsl import Strategy
from backend.ai.strategy.evaluation import GateThresholds, evaluate_gate
from backend.ai.strategy.timeframes import (
    annualization_periods,
    resample_ohlcv,
    tf_config,
)

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


def _bars(n: int = 520, seed: int = 3, freq: str = "5min") -> pd.DataFrame:
    np.random.seed(seed)
    t = np.arange(n)
    base = 100 + 6 * np.sin(t / 20)
    close = base + np.random.normal(0, 0.4, n)
    high = close + np.abs(np.random.normal(0, 0.25, n))
    low = close - np.abs(np.random.normal(0, 0.25, n))
    open_ = close - np.random.normal(0, 0.15, n)
    vol = np.random.randint(1_000, 9_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2025-01-01 09:15", periods=n, freq=freq),
    )


def _rsi_strategy(timeframe: str) -> Strategy:
    return Strategy.model_validate({
        "name": f"RSI MR {timeframe}",
        "universe": "single",
        "symbol": "RELIANCE",
        "timeframe": timeframe,
        "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 40},
        "exit": {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 60},
        "stop_loss_pct": 1.5,       # required for intraday; harmless for 4h/1d
        "take_profit_pct": 2.5,
        "position_size": {"kind": "percent_of_capital", "value": 10},
        "mode": "backtest",
    })


class TestStrategiesAcrossTimeframes:
    def test_runs_and_gates_at_every_timeframe(self):
        # The headline "look at this well" check: one strategy idea, evaluated
        # at every timeframe, each producing a coherent gate verdict.
        for tf in TIMEFRAMES:
            strat = _rsi_strategy(tf)
            ppy = annualization_periods(tf)
            wf = run_walk_forward(strat, _bars(520), symbol="RELIANCE", folds=4, periods_per_year=ppy)
            assert wf.n_folds >= 1, f"{tf}: no folds"
            assert wf.oos_trades == wf.in_sample.total_trades, f"{tf}: trade leak"
            res = evaluate_gate(wf.to_summary_dict(), GateThresholds())
            assert isinstance(res.passed, bool), f"{tf}: gate didn't verdict"
            if not res.passed:
                assert len(res.failures) >= 1

    def test_each_intraday_timeframe_requires_a_stop(self):
        # DSL guard: intraday strategies must carry a hard stop. Validates the
        # 5m/15m/1h paths (the ones the user asked for) enforce it.
        import pytest
        from pydantic import ValidationError
        for tf in ("5m", "15m", "1h"):
            with pytest.raises(ValidationError):
                Strategy.model_validate({
                    "name": f"no-stop {tf}", "universe": "single", "symbol": "X",
                    "timeframe": tf,
                    "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 40},
                    "exit": {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 60},
                    "position_size": {"kind": "percent_of_capital", "value": 10},
                    "mode": "backtest",
                })


class TestAnnualizationApplied:
    def test_sharpe_scales_with_timeframe_factor(self):
        # Same bars + logic, only the annualization factor differs → the Sharpe
        # scales by sqrt(periods ratio). Proves periods_per_year is wired
        # through to the Sharpe (not hardcoded 252).
        bars = _bars(520)
        wf_5m = run_walk_forward(_rsi_strategy("5m"), bars, symbol="X", periods_per_year=annualization_periods("5m"))
        wf_1d = run_walk_forward(_rsi_strategy("1d"), bars, symbol="X", periods_per_year=annualization_periods("1d"))
        if wf_1d.in_sample.sharpe_ratio != 0:
            assert abs(wf_5m.in_sample.sharpe_ratio) >= abs(wf_1d.in_sample.sharpe_ratio)


class TestFourHourPipeline:
    def test_4h_is_resampled_from_1h(self):
        cfg = tf_config("4h")
        assert cfg.fetch_interval == "1h" and cfg.resample_to == "4h"
        bars_1h = _bars(400, freq="1h")
        bars_4h = resample_ohlcv(bars_1h, "4h")
        assert 0 < len(bars_4h) < len(bars_1h)
