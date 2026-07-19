"""Walk-forward / out-of-sample backtest tests.

The walk-forward harness runs the single in-sample backtest once, then
segments its trades + equity curve into K contiguous time windows and
reports per-window + aggregate OOS metrics. This is the data the
promotion gate scores (see test_strategy_gate.py).

Synthetic OHLCV — no network, deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ai.strategy.backtest import (
    DSLBacktestResult,
    WalkForwardResult,
    run_walk_forward,
)
from backend.ai.strategy.dsl import Strategy
from backend.ai.strategy.evaluation import GateThresholds, evaluate_gate


def _make_bars(n: int = 400, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    t = np.arange(n)
    base = 100 + 8 * np.sin(t / 14)
    noise = np.random.normal(0, 0.5, n)
    close = base + noise
    high = close + np.abs(np.random.normal(0, 0.3, n))
    low = close - np.abs(np.random.normal(0, 0.3, n))
    open_ = close - np.random.normal(0, 0.2, n)
    vol = np.random.randint(100_000, 500_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2024-01-01", periods=n),
    )


def _rsi_strategy() -> Strategy:
    return Strategy.model_validate({
        "name": "RSI Mean Reversion",
        "universe": "nifty50",
        "timeframe": "1d",
        "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 40},
        "exit": {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 60},
        "stop_loss_pct": 3.0,
        "take_profit_pct": 5.0,
        "position_size": {"kind": "percent_of_capital", "value": 10},
        "mode": "backtest",
    })


class TestWalkForwardShape:
    def test_produces_folds_and_oos_block(self):
        wf = run_walk_forward(_rsi_strategy(), _make_bars(400), symbol="TEST", folds=4)
        assert isinstance(wf, WalkForwardResult)
        assert wf.n_folds == 4
        assert len(wf.folds) == wf.n_folds
        assert isinstance(wf.in_sample, DSLBacktestResult)

    def test_summary_carries_gate_readable_oos_block(self):
        wf = run_walk_forward(_rsi_strategy(), _make_bars(400), symbol="TEST")
        s = wf.to_summary_dict()
        assert "out_of_sample" in s
        oos = s["out_of_sample"]
        for key in (
            "n_folds", "oos_trades", "oos_folds_profitable", "oos_consistency",
            "oos_mean_sharpe", "oos_worst_drawdown_pct",
            "holdout_return_pct", "holdout_sharpe", "holdout_trades",
        ):
            assert key in oos, f"missing {key}"
        assert 0.0 <= oos["oos_consistency"] <= 1.0

    def test_oos_trades_equal_in_sample_trades(self):
        # Every trade falls in exactly one fold → no double-count, no drop.
        wf = run_walk_forward(_rsi_strategy(), _make_bars(400), symbol="TEST")
        assert wf.oos_trades == wf.in_sample.total_trades
        assert wf.oos_trades == sum(f.trades for f in wf.folds)

    def test_consistency_is_profitable_folds_over_total(self):
        wf = run_walk_forward(_rsi_strategy(), _make_bars(400), symbol="TEST")
        expected = wf.oos_folds_profitable / wf.n_folds
        assert abs(wf.oos_consistency - expected) < 1e-9


class TestWalkForwardDegrades:
    def test_short_history_reduces_folds(self):
        # eval region ~60 bars (260 - MIN_LOOKBACK 200) → fewer folds.
        wf = run_walk_forward(_rsi_strategy(), _make_bars(260), symbol="TEST", folds=4)
        assert 1 <= wf.n_folds <= 4


class TestWalkForwardFeedsGate:
    def test_real_output_runs_through_gate_without_error(self):
        wf = run_walk_forward(_rsi_strategy(), _make_bars(400), symbol="TEST")
        res = evaluate_gate(wf.to_summary_dict(), GateThresholds())
        assert isinstance(res.passed, bool)
        # On synthetic mean-reverting data the gate may pass or fail, but it
        # must produce a coherent verdict with reasons when it fails.
        if not res.passed:
            assert len(res.failures) >= 1

    def test_settings_derived_thresholds_gate_real_backtest(self):
        # Exact path the /transition endpoint runs (minus HTTP): build
        # thresholds from settings, score a real walk-forward summary.
        # Also asserts every gate setting exists with the right type.
        from backend.core.config import settings

        th = GateThresholds(
            min_oos_sharpe=settings.STRATEGY_GATE_MIN_OOS_SHARPE,
            min_trades=settings.STRATEGY_GATE_MIN_TRADES,
            max_drawdown_pct=settings.STRATEGY_GATE_MAX_DRAWDOWN_PCT,
            min_consistency=settings.STRATEGY_GATE_MIN_CONSISTENCY,
            require_holdout_positive=settings.STRATEGY_GATE_REQUIRE_HOLDOUT_POSITIVE,
        )
        wf = run_walk_forward(_rsi_strategy(), _make_bars(400), symbol="TEST")
        res = evaluate_gate(wf.to_summary_dict(), th)
        assert isinstance(res.passed, bool)

    def test_impossible_threshold_blocks_everything(self):
        # A min-trades wall no strategy can clear → must block, proving the
        # gate actually rejects (not a rubber stamp).
        wf = run_walk_forward(_rsi_strategy(), _make_bars(400), symbol="TEST")
        res = evaluate_gate(wf.to_summary_dict(), GateThresholds(min_trades=100_000))
        assert res.passed is False
        assert any("trade" in f.lower() for f in res.failures)
