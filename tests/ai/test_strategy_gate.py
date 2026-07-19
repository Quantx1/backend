"""Promotion-gate tests — the out-of-sample quality gate that blocks
overfit strategies from reaching live money.

Pure-function tests (no Supabase, no network). The gate reads a stored
``last_backtest`` summary dict (which carries an ``out_of_sample`` block
from the walk-forward backtest) and returns pass/fail + human-readable
reasons.
"""

from __future__ import annotations

from backend.ai.strategy.evaluation import (
    GateResult,
    GateThresholds,
    evaluate_gate,
)


def _passing_backtest() -> dict:
    """A last_backtest summary that should clear the default gate."""
    return {
        "symbol": "TEST",
        "sharpe_ratio": 1.2,
        "win_rate": 0.55,
        "total_trades": 40,
        "out_of_sample": {
            "n_folds": 4,
            "oos_trades": 32,
            "oos_folds_profitable": 3,
            "oos_consistency": 0.75,
            "oos_mean_sharpe": 0.9,
            "oos_worst_drawdown_pct": 18.0,
            "holdout_return_pct": 3.4,
            "holdout_sharpe": 1.0,
            "holdout_trades": 8,
        },
    }


class TestGatePasses:
    def test_good_strategy_passes_default_gate(self):
        res = evaluate_gate(_passing_backtest(), GateThresholds())
        assert isinstance(res, GateResult)
        assert res.passed is True
        assert res.failures == []


class TestGateBlocks:
    def test_missing_backtest_blocks(self):
        res = evaluate_gate(None, GateThresholds())
        assert res.passed is False
        assert any("backtest" in f.lower() for f in res.failures)

    def test_missing_oos_block_blocks(self):
        # An in-sample-only backtest (no walk-forward) must NOT pass — that's
        # the whole point: in-sample numbers can't gate live money.
        bt = {"sharpe_ratio": 3.0, "win_rate": 0.9, "total_trades": 100}
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert any("out-of-sample" in f.lower() or "walk-forward" in f.lower() for f in res.failures)

    def test_too_few_trades_blocks(self):
        bt = _passing_backtest()
        bt["out_of_sample"]["oos_trades"] = 5
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert any("trade" in f.lower() for f in res.failures)

    def test_low_oos_sharpe_blocks(self):
        bt = _passing_backtest()
        bt["out_of_sample"]["oos_mean_sharpe"] = 0.1
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert any("sharpe" in f.lower() for f in res.failures)

    def test_excess_drawdown_blocks(self):
        bt = _passing_backtest()
        bt["out_of_sample"]["oos_worst_drawdown_pct"] = 60.0
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert any("drawdown" in f.lower() for f in res.failures)

    def test_inconsistent_across_folds_blocks(self):
        # Worked in one window, lost in the rest → overfit signature.
        bt = _passing_backtest()
        bt["out_of_sample"]["oos_consistency"] = 0.25
        bt["out_of_sample"]["oos_folds_profitable"] = 1
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert any("consisten" in f.lower() or "fold" in f.lower() for f in res.failures)

    def test_negative_holdout_blocks(self):
        # The most-recent untouched window lost money → block.
        bt = _passing_backtest()
        bt["out_of_sample"]["holdout_return_pct"] = -2.5
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert any("holdout" in f.lower() or "recent" in f.lower() for f in res.failures)

    def test_multiple_failures_all_reported(self):
        bt = _passing_backtest()
        bt["out_of_sample"]["oos_trades"] = 2
        bt["out_of_sample"]["oos_mean_sharpe"] = -0.5
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert len(res.failures) >= 2


class TestRegimeCoverageGate:
    def test_regime_strategy_low_coverage_blocks(self):
        bt = _passing_backtest()
        bt["regime"] = {"used": True, "coverage": 0.3}  # mostly fake regime
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is False
        assert any("regime" in f.lower() for f in res.failures)

    def test_regime_strategy_high_coverage_ok(self):
        bt = _passing_backtest()
        bt["regime"] = {"used": True, "coverage": 0.95}  # real regime
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is True

    def test_regime_free_strategy_unaffected(self):
        # No regime block → the regime guard never fires.
        bt = _passing_backtest()
        assert "regime" not in bt
        res = evaluate_gate(bt, GateThresholds())
        assert res.passed is True


class TestThresholdsTunable:
    def test_relaxed_thresholds_let_marginal_through(self):
        bt = _passing_backtest()
        bt["out_of_sample"]["oos_mean_sharpe"] = 0.3
        strict = evaluate_gate(bt, GateThresholds(min_oos_sharpe=0.5))
        relaxed = evaluate_gate(bt, GateThresholds(min_oos_sharpe=0.2))
        assert strict.passed is False
        assert relaxed.passed is True
