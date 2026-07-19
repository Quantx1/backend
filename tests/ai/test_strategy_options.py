"""PR-J multi-leg options tests.

Three layers:
1. DSL schema — LegSpec + Strategy with instrument_segment=OPTIONS
2. Leg resolver — strike / expiry resolution
3. Multi-leg backtest — synthetic OHLCV, deterministic results
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.ai.strategy.dsl import (
    ExpiryAnchor,
    InstrumentSegment,
    LegSpec,
    OptionSide,
    OptionType,
    Strategy,
    StrikeAnchor,
)
from backend.ai.strategy.options_backtest import (
    OptionsBacktestResult,
    run_options_backtest,
)
from backend.ai.strategy.options_resolver import (
    resolve_expiry,
    resolve_legs,
    resolve_strike,
)


def _make_index_bars(n: int = 300, start: float = 18000.0, seed: int = 42) -> pd.DataFrame:
    """NIFTY-like daily OHLCV — moderate drift + 1% vol."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)  # ~5% annual drift, 16% annual vol
    closes = start * np.cumprod(1 + rets)
    return pd.DataFrame({
        "open": closes + rng.normal(0, 20, n),
        "high": closes + np.abs(rng.normal(0, 40, n)),
        "low":  closes - np.abs(rng.normal(0, 40, n)),
        "close": closes,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n))


# ─────────────────────────────────────────────────────────────────────
# Layer 1: DSL schema
# ─────────────────────────────────────────────────────────────────────

class TestLegSpec:

    def test_atm_leg_validates(self):
        leg = LegSpec.model_validate({
            "side": "buy", "option_type": "CE",
            "strike_anchor": "ATM", "strike_offset": 0,
        })
        assert leg.strike_anchor == StrikeAnchor.ATM

    def test_atm_with_nonzero_offset_rejected(self):
        with pytest.raises(ValueError, match="ATM requires strike_offset=0"):
            LegSpec.model_validate({
                "side": "buy", "option_type": "CE",
                "strike_anchor": "ATM", "strike_offset": 2,
            })

    def test_atm_plus_n_requires_positive_int(self):
        with pytest.raises(ValueError, match="positive integer"):
            LegSpec.model_validate({
                "side": "sell", "option_type": "PE",
                "strike_anchor": "ATM+N", "strike_offset": 0,
            })
        with pytest.raises(ValueError, match="positive integer"):
            LegSpec.model_validate({
                "side": "sell", "option_type": "PE",
                "strike_anchor": "ATM+N", "strike_offset": 2.5,
            })

    def test_atm_plus_n_caps_at_20(self):
        with pytest.raises(ValueError, match="max 20"):
            LegSpec.model_validate({
                "side": "buy", "option_type": "CE",
                "strike_anchor": "ATM+N", "strike_offset": 50,
            })

    def test_otm_delta_bounds(self):
        # Valid
        LegSpec.model_validate({
            "side": "sell", "option_type": "PE",
            "strike_anchor": "OTM_DELTA", "strike_offset": 0.30,
        })
        with pytest.raises(ValueError, match="0, 0.5"):
            LegSpec.model_validate({
                "side": "sell", "option_type": "PE",
                "strike_anchor": "OTM_DELTA", "strike_offset": 0.7,
            })

    def test_pct_offset_rejects_zero(self):
        with pytest.raises(ValueError, match="-50, 50"):
            LegSpec.model_validate({
                "side": "buy", "option_type": "CE",
                "strike_anchor": "PCT_OFFSET", "strike_offset": 0,
            })


class TestStrategyMultiLeg:

    @staticmethod
    def _iron_condor_dsl():
        return {
            "name": "IC",
            "instrument_segment": "OPTIONS",
            "symbol": "NIFTY",
            "universe": "single",
            "timeframe": "1d",
            "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 60},
            "exit":  {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 70},
            "stop_loss_pct": 50,
            "take_profit_pct": 40,
            "position_size": {"kind": "percent_of_capital", "value": 10},
            "legs": [
                {"side": "sell", "option_type": "PE", "strike_anchor": "ATM-N", "strike_offset": 2, "expiry": "current_week"},
                {"side": "buy",  "option_type": "PE", "strike_anchor": "ATM-N", "strike_offset": 4, "expiry": "current_week"},
                {"side": "sell", "option_type": "CE", "strike_anchor": "ATM+N", "strike_offset": 2, "expiry": "current_week"},
                {"side": "buy",  "option_type": "CE", "strike_anchor": "ATM+N", "strike_offset": 4, "expiry": "current_week"},
            ],
        }

    def test_iron_condor_validates(self):
        s = Strategy.model_validate(self._iron_condor_dsl())
        assert s.instrument_segment == InstrumentSegment.OPTIONS
        assert len(s.legs) == 4

    def test_equity_default_segment_is_equity(self):
        s = Strategy.model_validate({
            "name": "X", "universe": "nifty50", "timeframe": "1d",
            "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 30},
            "exit":  {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 70},
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        assert s.instrument_segment == InstrumentSegment.EQUITY
        assert s.legs is None

    def test_options_without_legs_rejected(self):
        dsl = self._iron_condor_dsl()
        dsl["legs"] = []
        with pytest.raises(ValueError, match="OPTIONS requires legs"):
            Strategy.model_validate(dsl)

    def test_options_too_many_legs_rejected(self):
        dsl = self._iron_condor_dsl()
        dsl["legs"] = dsl["legs"] + [
            {"side": "buy", "option_type": "CE", "strike_anchor": "ATM+N", "strike_offset": 6, "expiry": "current_week"},
        ]
        with pytest.raises(ValueError, match="at most 4 legs"):
            Strategy.model_validate(dsl)

    def test_options_with_legs_in_equity_segment_rejected(self):
        dsl = self._iron_condor_dsl()
        dsl["instrument_segment"] = "EQUITY"
        dsl["universe"] = "nifty50"  # EQUITY can have non-single universe
        dsl.pop("symbol")
        with pytest.raises(ValueError, match="EQUITY must not set legs"):
            Strategy.model_validate(dsl)

    def test_options_requires_universe_single(self):
        dsl = self._iron_condor_dsl()
        dsl["universe"] = "nifty50"
        with pytest.raises(ValueError, match="universe='single'"):
            Strategy.model_validate(dsl)

    def test_calendar_spread_mixed_expiries_allowed(self):
        # HIGH #7 (2026-05-31): calendar/diagonal spreads (mixed expiry
        # anchors) are now supported — the validator caps at 4 distinct
        # anchors instead of forcing a single expiry across all legs.
        dsl = self._iron_condor_dsl()
        dsl["legs"][0]["expiry"] = "current_month"  # 2 distinct anchors now
        s = Strategy.model_validate(dsl)            # must NOT raise
        assert {leg.expiry.value for leg in s.legs} == {"current_week", "current_month"}

    def test_four_distinct_expiry_anchors_allowed(self):
        # 4 distinct anchors sits exactly at the cap → still valid.
        dsl = self._iron_condor_dsl()
        for leg, exp in zip(
            dsl["legs"], ("current_week", "next_week", "current_month", "next_month")
        ):
            leg["expiry"] = exp
        s = Strategy.model_validate(dsl)
        assert len({leg.expiry for leg in s.legs}) == 4


# ─────────────────────────────────────────────────────────────────────
# Layer 2: Leg resolver
# ─────────────────────────────────────────────────────────────────────

class TestResolveStrike:

    def test_atm_rounds_to_interval(self):
        # NIFTY interval = 50
        assert resolve_strike(StrikeAnchor.ATM, 0, spot=18234, symbol="NIFTY") == 18250
        assert resolve_strike(StrikeAnchor.ATM, 0, spot=18225, symbol="NIFTY") in (18200, 18250)

    def test_atm_plus_n(self):
        assert resolve_strike(StrikeAnchor.ATM_PLUS_N, 2, spot=18234, symbol="NIFTY") == 18350

    def test_atm_minus_n(self):
        assert resolve_strike(StrikeAnchor.ATM_MINUS_N, 3, spot=18234, symbol="NIFTY") == 18100

    def test_atm_banknifty_uses_100_interval(self):
        assert resolve_strike(StrikeAnchor.ATM, 0, spot=44150, symbol="BANKNIFTY") in (44100, 44200)

    def test_pct_offset(self):
        # +5% of 18000 = 18900, round to nearest 50 → 18900
        k = resolve_strike(StrikeAnchor.PCT_OFFSET, 5, spot=18000, symbol="NIFTY")
        assert abs(k - 18900) <= 50


class TestResolveExpiry:

    def test_current_week_strictly_future(self):
        # Wed 2025-06-04 — next Thursday is 2025-06-05 for NIFTY
        d = resolve_expiry(ExpiryAnchor.CURRENT_WEEK, "NIFTY", today=date(2025, 6, 4))
        assert d == date(2025, 6, 5)
        assert d > date(2025, 6, 4)

    def test_next_week_is_week_after_current(self):
        cur = resolve_expiry(ExpiryAnchor.CURRENT_WEEK, "NIFTY", today=date(2025, 6, 4))
        nxt = resolve_expiry(ExpiryAnchor.NEXT_WEEK, "NIFTY", today=date(2025, 6, 4))
        assert nxt > cur

    def test_banknifty_uses_wednesday(self):
        # Tue 2025-06-03 → Wed 2025-06-04 for BANKNIFTY
        d = resolve_expiry(ExpiryAnchor.CURRENT_WEEK, "BANKNIFTY", today=date(2025, 6, 3))
        assert d.weekday() == 2  # Wednesday


class TestResolveLegs:

    def test_iron_condor_legs_resolved(self):
        legs = [
            LegSpec(side=OptionSide.SELL, option_type=OptionType.PE, strike_anchor=StrikeAnchor.ATM_MINUS_N, strike_offset=2, expiry=ExpiryAnchor.CURRENT_WEEK),
            LegSpec(side=OptionSide.BUY,  option_type=OptionType.PE, strike_anchor=StrikeAnchor.ATM_MINUS_N, strike_offset=4, expiry=ExpiryAnchor.CURRENT_WEEK),
            LegSpec(side=OptionSide.SELL, option_type=OptionType.CE, strike_anchor=StrikeAnchor.ATM_PLUS_N,  strike_offset=2, expiry=ExpiryAnchor.CURRENT_WEEK),
            LegSpec(side=OptionSide.BUY,  option_type=OptionType.CE, strike_anchor=StrikeAnchor.ATM_PLUS_N,  strike_offset=4, expiry=ExpiryAnchor.CURRENT_WEEK),
        ]
        resolved = resolve_legs(legs, spot=18000, symbol="NIFTY", today=date(2025, 6, 4))
        # All legs should share the same expiry
        assert len({r.expiry for r in resolved}) == 1
        # Strikes in expected pattern: 17900, 17800, 18100, 18200
        strikes = sorted(r.strike for r in resolved)
        assert strikes == [17800, 17900, 18100, 18200]


# ─────────────────────────────────────────────────────────────────────
# Layer 3: Backtest
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def iron_condor_strategy() -> Strategy:
    return Strategy.model_validate({
        "name": "IC test",
        "instrument_segment": "OPTIONS",
        "symbol": "NIFTY",
        "universe": "single",
        "timeframe": "1d",
        "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 60},
        "exit":  {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 80},
        "stop_loss_pct": 50,
        "take_profit_pct": 50,
        "position_size": {"kind": "percent_of_capital", "value": 20},
        "legs": [
            {"side": "sell", "option_type": "PE", "strike_anchor": "ATM-N", "strike_offset": 2, "expiry": "current_week"},
            {"side": "buy",  "option_type": "PE", "strike_anchor": "ATM-N", "strike_offset": 4, "expiry": "current_week"},
            {"side": "sell", "option_type": "CE", "strike_anchor": "ATM+N", "strike_offset": 2, "expiry": "current_week"},
            {"side": "buy",  "option_type": "CE", "strike_anchor": "ATM+N", "strike_offset": 4, "expiry": "current_week"},
        ],
    })


@pytest.fixture
def long_straddle_strategy() -> Strategy:
    return Strategy.model_validate({
        "name": "LS test",
        "instrument_segment": "OPTIONS",
        "symbol": "NIFTY",
        "universe": "single",
        "timeframe": "1d",
        "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 60},
        "exit":  {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 90},
        "stop_loss_pct": 50,
        "take_profit_pct": 100,
        "position_size": {"kind": "percent_of_capital", "value": 15},
        "legs": [
            {"side": "buy", "option_type": "CE", "strike_anchor": "ATM", "strike_offset": 0, "expiry": "current_week"},
            {"side": "buy", "option_type": "PE", "strike_anchor": "ATM", "strike_offset": 0, "expiry": "current_week"},
        ],
    })


class TestOptionsBacktest:

    def test_iron_condor_runs_end_to_end(self, iron_condor_strategy):
        bars = _make_index_bars(300)
        result = run_options_backtest(iron_condor_strategy, bars, symbol="NIFTY")
        assert isinstance(result, OptionsBacktestResult)
        assert result.symbol == "NIFTY"
        assert result.strategy_name == "IC test"
        assert result.total_trades >= 1

    def test_results_are_sensible(self, iron_condor_strategy):
        bars = _make_index_bars(300)
        result = run_options_backtest(iron_condor_strategy, bars, symbol="NIFTY")
        # Win rate in [0, 1]
        assert 0.0 <= result.win_rate <= 1.0
        # Final capital can be above OR below initial, but never negative
        assert result.final_capital > 0
        # Max DD is non-negative percent
        assert result.max_drawdown_pct >= 0.0

    def test_long_straddle_takes_trades(self, long_straddle_strategy):
        bars = _make_index_bars(300)
        result = run_options_backtest(long_straddle_strategy, bars, symbol="NIFTY")
        assert result.total_trades >= 1

    def test_summary_is_json_safe(self, iron_condor_strategy):
        import json
        bars = _make_index_bars(300)
        result = run_options_backtest(iron_condor_strategy, bars, symbol="NIFTY")
        summary = result.to_summary_dict()
        json.dumps(summary)
        assert summary["segment"] == "OPTIONS"
        assert summary["synthetic_backtest"] is True

    def test_full_dict_has_trades_and_curve(self, iron_condor_strategy):
        bars = _make_index_bars(300)
        result = run_options_backtest(iron_condor_strategy, bars, symbol="NIFTY")
        full = result.to_full_dict()
        assert "trades" in full
        assert "equity_curve" in full
        assert "sigma_assumptions" in full
        assert len(full["equity_curve"]) > 0

    def test_trade_pnl_is_realistic(self, iron_condor_strategy):
        """Sanity: a single Iron Condor trade should not return more than
        20% of position margin. Real ICs make 2-5%. The bug we fixed (lot
        sizing double-counting) was returning hundreds of percent."""
        bars = _make_index_bars(300)
        result = run_options_backtest(iron_condor_strategy, bars, symbol="NIFTY")
        for t in result.trades:
            assert abs(t.net_pnl_pct) < 60, (
                f"Trade P&L {t.net_pnl_pct}% is unrealistic for Iron Condor — "
                f"likely lot-sizing bug regression"
            )

    def test_equity_only_strategy_rejected(self):
        equity = Strategy.model_validate({
            "name": "X", "universe": "nifty50", "timeframe": "1d",
            "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 30},
            "exit":  {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 70},
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        bars = _make_index_bars(300)
        with pytest.raises(ValueError, match="requires Strategy.legs"):
            run_options_backtest(equity, bars, symbol="NIFTY")

    def test_too_few_bars_rejected(self, iron_condor_strategy):
        bars = _make_index_bars(20)
        with pytest.raises(ValueError, match="insufficient bars"):
            run_options_backtest(iron_condor_strategy, bars, symbol="NIFTY")
