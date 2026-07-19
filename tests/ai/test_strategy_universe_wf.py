"""Multi-symbol (universe) walk-forward + breadth-gate tests.

A strategy that only works on one cherry-picked symbol is overfit. The
universe walk-forward runs the per-symbol walk-forward across many symbols and
adds a breadth score (fraction of symbols profitable); the gate blocks a
universe strategy whose breadth is too low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ai.strategy.backtest import (
    UniverseWalkForwardResult,
    run_universe_walk_forward,
)
from backend.ai.strategy.dsl import Strategy
from backend.ai.strategy.evaluation import GateThresholds, evaluate_gate


def _make_bars(n: int = 400, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    t = np.arange(n)
    base = 100 + 8 * np.sin(t / 14)
    close = base + np.random.normal(0, 0.5, n)
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
        "name": "RSI MR",
        "universe": "nifty50",
        "timeframe": "1d",
        "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 40},
        "exit": {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 60},
        "stop_loss_pct": 3.0,
        "take_profit_pct": 5.0,
        "position_size": {"kind": "percent_of_capital", "value": 10},
        "mode": "backtest",
    })


class TestUniverseWalkForward:
    def test_aggregates_across_symbols(self):
        bars = {f"SYM{i}": _make_bars(400, seed=i) for i in range(5)}
        res = run_universe_walk_forward(_rsi_strategy(), bars, universe="nifty50", folds=4)
        assert isinstance(res, UniverseWalkForwardResult)
        assert res.symbols_tested == 5
        assert 0 <= res.symbols_profitable <= 5
        assert 0.0 <= res.breadth <= 1.0

    def test_summary_has_breadth_and_per_symbol(self):
        bars = {f"S{i}": _make_bars(400, seed=i + 10) for i in range(4)}
        oos = run_universe_walk_forward(_rsi_strategy(), bars, folds=4).to_summary_dict()["out_of_sample"]
        assert oos["symbols_tested"] == 4
        assert "breadth" in oos
        assert len(oos["per_symbol"]) == 4
        # aggregate OOS trades == sum of per-symbol OOS trades
        assert oos["oos_trades"] == sum(p["oos_trades"] for p in oos["per_symbol"])
        for key in ("oos_mean_sharpe", "oos_consistency", "oos_worst_drawdown_pct", "holdout_return_pct"):
            assert key in oos

    def test_skips_symbols_with_insufficient_history(self):
        bars = {"GOOD": _make_bars(400), "TINY": _make_bars(50)}  # 50 < MIN_LOOKBACK+10
        res = run_universe_walk_forward(_rsi_strategy(), bars, folds=4)
        assert res.symbols_tested == 1  # TINY skipped, batch didn't fail


class TestBreadthGate:
    def test_low_breadth_blocks_universe_strategy(self):
        bars = {f"S{i}": _make_bars(400, seed=i) for i in range(4)}
        summary = run_universe_walk_forward(_rsi_strategy(), bars, folds=4).to_summary_dict()
        summary["out_of_sample"]["breadth"] = 0.1
        summary["out_of_sample"]["symbols_profitable"] = 1
        # relax every other bar so ONLY breadth can fail
        th = GateThresholds(
            min_trades=0, min_oos_sharpe=-99.0, max_drawdown_pct=9999.0,
            min_consistency=0.0, require_holdout_positive=False, min_symbol_breadth=0.5,
        )
        res = evaluate_gate(summary, th)
        assert res.passed is False
        assert any("symbol" in f.lower() for f in res.failures)

    def test_breadth_check_skipped_for_single_symbol(self):
        # A single-symbol strategy has no symbols_tested>1 → breadth never blocks.
        single = {
            "out_of_sample": {
                "oos_trades": 40, "oos_mean_sharpe": 1.0, "oos_worst_drawdown_pct": 10.0,
                "oos_consistency": 1.0, "holdout_return_pct": 2.0,
                # no symbols_tested key → single-symbol path
            },
        }
        res = evaluate_gate(single, GateThresholds())
        assert res.passed is True
