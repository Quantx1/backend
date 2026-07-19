"""Unit tests for the Strategy DSL — PR-D.

Three families:
  1. DSL validation — valid strategies parse; invalid ones reject loudly.
  2. Indicator computation — every registered indicator returns a finite
     value (or NaN with insufficient data); no exceptions.
  3. Interpreter — Condition trees evaluate to expected truthiness on
     synthetic price bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.ai.strategy.dsl import (
    Condition,
    ConditionKind,
    INDICATOR_REGISTRY,
    EngineName,
    Operator,
    PositionSize,
    PositionSizeKind,
    RegimeFilter,
    Strategy,
    StrategyMode,
    Timeframe,
    Universe,
)
from backend.ai.strategy.indicators import (
    compute_indicator,
    compute_indicator_series,
    list_indicators,
)
from backend.ai.strategy.interpreter import (
    EngineSignals,
    InterpreterContext,
    evaluate_condition,
    evaluate_entry,
    evaluate_exit,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def trending_up_bars() -> pd.DataFrame:
    """250 bars of a steady uptrend with realistic OHLCV — long enough
    for EMA200 to settle, short enough to run fast."""
    n = 250
    np.random.seed(42)
    base = np.linspace(100, 200, n)
    noise = np.random.normal(0, 1.5, n)
    close = base + noise
    high = close + np.abs(np.random.normal(0, 0.6, n))
    low = close - np.abs(np.random.normal(0, 0.6, n))
    open_ = close - np.random.normal(0, 0.4, n)
    vol = np.random.randint(100_000, 500_000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    }, index=pd.date_range("2025-01-01", periods=n))


@pytest.fixture
def basic_strategy_dict() -> dict:
    return {
        "name": "RSI Mean Reversion",
        "universe": "nifty50",
        "timeframe": "1d",
        "entry": {
            "kind": "indicator_compare",
            "indicator": "rsi14",
            "op": "<",
            "value": 30,
        },
        "exit": {
            "kind": "indicator_compare",
            "indicator": "rsi14",
            "op": ">",
            "value": 70,
        },
        "stop_loss_pct": 3.0,
        "take_profit_pct": 6.0,
        "position_size": {"kind": "percent_of_capital", "value": 5.0},
        "regime_filter": "any",
        "lookback_days": 90,
        "mode": "backtest",
    }


# ─────────────────────────────────────────────────────────────────────
# 1. DSL validation
# ─────────────────────────────────────────────────────────────────────


class TestDSLValidation:

    def test_valid_basic_strategy_parses(self, basic_strategy_dict):
        s = Strategy.model_validate(basic_strategy_dict)
        assert s.name == "RSI Mean Reversion"
        assert s.universe == Universe.NIFTY_50
        assert s.timeframe == Timeframe.D1

    def test_json_roundtrip_is_identical(self, basic_strategy_dict):
        s1 = Strategy.model_validate(basic_strategy_dict)
        js = s1.model_dump_json()
        s2 = Strategy.model_validate_json(js)
        assert s2.model_dump_json() == js

    def test_unknown_indicator_rejected(self, basic_strategy_dict):
        d = dict(basic_strategy_dict)
        d["entry"] = {
            "kind": "indicator_compare",
            "indicator": "rsi999",
            "op": "<",
            "value": 30,
        }
        with pytest.raises(Exception):
            Strategy.model_validate(d)

    def test_composite_without_children_rejected(self):
        with pytest.raises(Exception):
            Condition.model_validate({"kind": "composite_and"})

    def test_composite_with_one_child_rejected(self):
        with pytest.raises(Exception):
            Condition.model_validate({
                "kind": "composite_and",
                "children": [{"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 30}],
            })

    def test_indicator_cross_requires_cross_op(self):
        with pytest.raises(Exception):
            Condition.model_validate({
                "kind": "indicator_cross",
                "indicator": "ema8",
                "op": "<",
                "value": "ema21",
            })

    def test_indicator_cross_value_must_be_indicator(self):
        with pytest.raises(Exception):
            Condition.model_validate({
                "kind": "indicator_cross",
                "indicator": "ema8",
                "op": "crosses_above",
                "value": 42,
            })

    def test_engine_signal_requires_engine(self):
        with pytest.raises(Exception):
            Condition.model_validate({
                "kind": "engine_signal",
                "op": "==",
                "value": "bullish",
            })

    def test_intraday_without_stop_rejected(self, basic_strategy_dict):
        d = dict(basic_strategy_dict)
        d["timeframe"] = "5m"
        d["stop_loss_pct"] = None
        with pytest.raises(Exception):
            Strategy.model_validate(d)

    def test_single_universe_requires_symbol(self, basic_strategy_dict):
        d = dict(basic_strategy_dict)
        d["universe"] = "single"
        d["symbol"] = None
        with pytest.raises(Exception):
            Strategy.model_validate(d)

    def test_percent_of_capital_value_capped_at_100(self, basic_strategy_dict):
        d = dict(basic_strategy_dict)
        d["position_size"] = {"kind": "percent_of_capital", "value": 150}
        with pytest.raises(Exception):
            Strategy.model_validate(d)

    def test_between_op_requires_two_element_array(self):
        with pytest.raises(Exception):
            Condition.model_validate({
                "kind": "indicator_compare",
                "indicator": "rsi14",
                "op": "between",
                "value": [30, 30, 70],
            })

    def test_between_op_lo_must_be_less_than_hi(self):
        with pytest.raises(Exception):
            Condition.model_validate({
                "kind": "indicator_compare",
                "indicator": "rsi14",
                "op": "between",
                "value": [50, 30],
            })


# ─────────────────────────────────────────────────────────────────────
# 2. Indicator computation
# ─────────────────────────────────────────────────────────────────────


class TestIndicators:

    def test_registry_matches_implementations(self):
        assert set(INDICATOR_REGISTRY) == set(list_indicators())

    @pytest.mark.parametrize("name", list_indicators())
    def test_every_indicator_returns_finite_or_nan(self, name, trending_up_bars):
        """No indicator may raise an exception. Returns float (possibly NaN)."""
        v = compute_indicator(name, trending_up_bars)
        assert isinstance(v, float)

    def test_rsi14_in_valid_range(self, trending_up_bars):
        rsi = compute_indicator("rsi14", trending_up_bars)
        assert 0 <= rsi <= 100

    def test_uptrend_has_high_rsi(self, trending_up_bars):
        rsi = compute_indicator("rsi14", trending_up_bars)
        assert rsi > 50  # uptrend → RSI above 50

    def test_close_above_ema200_in_uptrend(self, trending_up_bars):
        close = compute_indicator("close", trending_up_bars)
        ema200 = compute_indicator("ema200", trending_up_bars)
        assert close > ema200


# ─────────────────────────────────────────────────────────────────────
# 3. Interpreter
# ─────────────────────────────────────────────────────────────────────


class TestInterpreter:

    def test_simple_compare_true(self, trending_up_bars):
        cond = Condition.model_validate({
            "kind": "indicator_compare",
            "indicator": "close",
            "op": ">",
            "value": 0,
        })
        ctx = InterpreterContext(bars=trending_up_bars)
        assert evaluate_condition(cond, ctx) is True

    def test_indicator_to_indicator_compare(self, trending_up_bars):
        # close > ema200 in our uptrend fixture
        cond = Condition.model_validate({
            "kind": "indicator_compare",
            "indicator": "close",
            "op": ">",
            "value": "ema200",
        })
        ctx = InterpreterContext(bars=trending_up_bars)
        assert evaluate_condition(cond, ctx) is True

    def test_composite_and_short_circuits(self, trending_up_bars):
        cond = Condition.model_validate({
            "kind": "composite_and",
            "children": [
                {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 50},
                {"kind": "indicator_compare", "indicator": "close", "op": "<", "value": 0},
            ],
        })
        ctx = InterpreterContext(bars=trending_up_bars)
        # AND with one false child → False
        assert evaluate_condition(cond, ctx) is False

    def test_composite_or_first_match_wins(self, trending_up_bars):
        cond = Condition.model_validate({
            "kind": "composite_or",
            "children": [
                {"kind": "indicator_compare", "indicator": "close", "op": "<", "value": 0},
                {"kind": "indicator_compare", "indicator": "close", "op": ">", "value": 0},
            ],
        })
        ctx = InterpreterContext(bars=trending_up_bars)
        assert evaluate_condition(cond, ctx) is True

    def test_engine_signal_missing_returns_false(self, trending_up_bars):
        cond = Condition.model_validate({
            "kind": "engine_signal",
            "engine": "Regime",
            "op": "==",
            "value": "bull",
        })
        ctx = InterpreterContext(bars=trending_up_bars, engines=EngineSignals())
        # regime=None → false
        assert evaluate_condition(cond, ctx) is False

    def test_engine_signal_match(self, trending_up_bars):
        cond = Condition.model_validate({
            "kind": "engine_signal",
            "engine": "Regime",
            "op": "==",
            "value": "bull",
        })
        ctx = InterpreterContext(
            bars=trending_up_bars,
            engines=EngineSignals(regime="bull"),
        )
        assert evaluate_condition(cond, ctx) is True

    def test_regime_filter_blocks_entry(self, trending_up_bars, basic_strategy_dict):
        d = dict(basic_strategy_dict)
        d["regime_filter"] = "bear_only"
        s = Strategy.model_validate(d)
        ctx = InterpreterContext(
            bars=trending_up_bars,
            engines=EngineSignals(regime="bull"),
        )
        # bull regime + bear_only filter → no entry, regardless of price condition
        assert evaluate_entry(s, ctx) is False

    def test_regime_filter_allows_matching(self, trending_up_bars, basic_strategy_dict):
        # Need to actually make the rsi14 < 30 condition true. Our uptrend
        # has high RSI, so flip the strategy entry to one that fires.
        d = dict(basic_strategy_dict)
        d["entry"] = {
            "kind": "indicator_compare",
            "indicator": "rsi14",
            "op": ">",
            "value": 50,
        }
        d["regime_filter"] = "bull_only"
        s = Strategy.model_validate(d)
        ctx = InterpreterContext(
            bars=trending_up_bars,
            engines=EngineSignals(regime="bull"),
        )
        assert evaluate_entry(s, ctx) is True

    def test_indicator_cross_detection(self, trending_up_bars):
        # In an uptrend, ema8 should be above ema21. Use a strict cross
        # condition that's unlikely to fire on the last bar (we test the
        # mechanism, not a specific market state).
        cond = Condition.model_validate({
            "kind": "indicator_cross",
            "indicator": "ema8",
            "op": "crosses_above",
            "value": "ema21",
        })
        ctx = InterpreterContext(bars=trending_up_bars)
        # Just verify it runs without error and returns bool
        result = evaluate_condition(cond, ctx)
        assert isinstance(result, bool)
