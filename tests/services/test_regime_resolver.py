"""Regime resolver tests — PR-? (regime morning-gap fix).

Pure-Python tests with a mocked Supabase client. Validates the
fail-open-to-sideways policy + carry-forward behaviour.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from backend.services.regime.resolver import (
    DEFAULT_REGIME,
    resolve_regime_at,
    resolve_regime_history,
)


def _mock_sb(rows):
    """Mock supabase whose regime_history query returns ``rows``."""
    sb = MagicMock()
    chain = MagicMock()
    chain.execute.return_value = MagicMock(data=rows)
    for attr in ("select", "eq", "lte", "gte", "order", "limit"):
        getattr(chain, attr).return_value = chain
    sb.table.return_value = chain
    return sb


class TestResolveRegimeAt:

    def test_default_when_supabase_none(self):
        assert resolve_regime_at(None) == DEFAULT_REGIME

    def test_default_when_empty_history(self):
        sb = _mock_sb([])
        assert resolve_regime_at(sb, at=date(2026, 5, 25)) == "sideways"

    def test_returns_latest_match(self):
        sb = _mock_sb([{"regime": "bull", "detected_at": "2026-05-24T08:15:00Z"}])
        assert resolve_regime_at(sb, at=date(2026, 5, 25)) == "bull"

    def test_normalises_unknown_regime_to_default(self):
        sb = _mock_sb([{"regime": "garbage", "detected_at": "2026-05-24T08:15:00Z"}])
        assert resolve_regime_at(sb) == DEFAULT_REGIME

    def test_handles_supabase_exception(self):
        sb = MagicMock()
        sb.table.side_effect = RuntimeError("simulated outage")
        # Must NEVER raise — falls through to default
        assert resolve_regime_at(sb) == DEFAULT_REGIME


class TestResolveRegimeHistory:

    def test_empty_history_fills_default(self):
        sb = _mock_sb([])
        result = resolve_regime_history(sb, start=date(2026, 5, 20), end=date(2026, 5, 22))
        assert len(result) == 3
        for d in result:
            assert result[d] == DEFAULT_REGIME

    def test_carry_forward_fills_gaps(self):
        sb = _mock_sb([
            {"regime": "bull", "detected_at": "2026-05-20T08:15:00Z"},
            # Skip 21st
            {"regime": "bear", "detected_at": "2026-05-22T08:15:00Z"},
        ])
        result = resolve_regime_history(sb, start=date(2026, 5, 20), end=date(2026, 5, 23))
        assert result[date(2026, 5, 20)] == "bull"
        assert result[date(2026, 5, 21)] == "bull"   # carry-forward
        assert result[date(2026, 5, 22)] == "bear"
        assert result[date(2026, 5, 23)] == "bear"   # carry-forward

    def test_inverted_range_returns_empty(self):
        sb = _mock_sb([])
        assert resolve_regime_history(sb, start=date(2026, 5, 25), end=date(2026, 5, 20)) == {}

    def test_prior_row_seeds_start(self):
        """If we have a regime row BEFORE the requested start, the start
        date inherits it via carry-forward — not the default."""
        sb = _mock_sb([
            {"regime": "bear", "detected_at": "2026-05-01T08:15:00Z"},
        ])
        # Range Apr-30..May-02 — start=Apr-30 has no row, but prior row exists
        result = resolve_regime_history(sb, start=date(2026, 5, 2), end=date(2026, 5, 3))
        # The prior bear row at May-01 should carry forward
        assert result[date(2026, 5, 2)] == "bear"
        assert result[date(2026, 5, 3)] == "bear"


class TestBacktestInjection:
    """End-to-end smoke: the API helper builds the right shape."""

    def test_strategy_uses_regime_heuristic(self):
        from backend.api.strategies_routes import _strategy_uses_regime
        from backend.ai.strategy.dsl import Strategy

        # Pure-technical → no regime needed
        s_no = Strategy.model_validate({
            "name": "x", "universe": "nifty50", "timeframe": "1d",
            "entry": {"kind": "indicator_compare", "indicator": "rsi14",
                      "op": "<", "value": 30},
            "exit":  {"kind": "indicator_compare", "indicator": "rsi14",
                      "op": ">", "value": 70},
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        assert _strategy_uses_regime(s_no) is False

        # regime_filter set → yes
        s_filter = Strategy.model_validate({**s_no.model_dump(mode="json"),
                                            "regime_filter": "bull_only"})
        assert _strategy_uses_regime(s_filter) is True

        # nested engine_signal Regime → yes
        s_engine = Strategy.model_validate({
            "name": "x", "universe": "nifty50", "timeframe": "1d",
            "entry": {
                "kind": "composite_and",
                "children": [
                    {"kind": "indicator_compare", "indicator": "close",
                     "op": ">", "value": "ema21"},
                    {"kind": "engine_signal", "engine": "Regime",
                     "op": "==", "value": "bull"},
                ],
            },
            "exit": {"kind": "indicator_compare", "indicator": "rsi14",
                     "op": ">", "value": 70},
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        assert _strategy_uses_regime(s_engine) is True

    def test_helper_skips_db_when_strategy_doesnt_use_regime(self):
        from backend.api.strategies_routes import _maybe_load_engine_signals
        from backend.ai.strategy.dsl import Strategy
        import pandas as pd

        s = Strategy.model_validate({
            "name": "x", "universe": "nifty50", "timeframe": "1d",
            "entry": {"kind": "indicator_compare", "indicator": "rsi14",
                      "op": "<", "value": 30},
            "exit":  {"kind": "indicator_compare", "indicator": "rsi14",
                      "op": ">", "value": 70},
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        sb = MagicMock()
        sb.table.side_effect = RuntimeError("should not be called")
        ohlcv = pd.DataFrame({"close": [100, 101]},
                              index=pd.date_range("2026-05-20", periods=2))
        # Must return None and NOT touch supabase
        assert _maybe_load_engine_signals(sb, s, ohlcv) is None

    def test_helper_returns_engine_signals_when_strategy_uses_regime(self):
        from backend.api.strategies_routes import _maybe_load_engine_signals
        from backend.ai.strategy.dsl import Strategy
        from backend.ai.strategy.interpreter import EngineSignals
        import pandas as pd

        s = Strategy.model_validate({
            "name": "x", "universe": "nifty50", "timeframe": "1d",
            "entry": {"kind": "engine_signal", "engine": "Regime",
                      "op": "==", "value": "bull"},
            "exit":  {"kind": "indicator_compare", "indicator": "rsi14",
                      "op": ">", "value": 70},
            "position_size": {"kind": "percent_of_capital", "value": 10},
        })
        sb = _mock_sb([
            {"regime": "bull", "detected_at": "2026-05-20T08:15:00Z"},
            {"regime": "bear", "detected_at": "2026-05-22T08:15:00Z"},
        ])
        ohlcv = pd.DataFrame(
            {"close": [100, 101, 102, 103]},
            index=pd.date_range("2026-05-20", periods=4),
        )
        result = _maybe_load_engine_signals(sb, s, ohlcv)
        assert result is not None
        assert len(result) == 4
        # Every entry should be EngineSignals with a regime populated
        for ts, eng in result.items():
            assert isinstance(eng, EngineSignals)
            assert eng.regime in ("bull", "sideways", "bear")
        # Carry-forward: 21st should inherit 20th's bull
        assert result[pd.Timestamp("2026-05-21")].regime == "bull"
        assert result[pd.Timestamp("2026-05-22")].regime == "bear"
        assert result[pd.Timestamp("2026-05-23")].regime == "bear"  # carry-forward
