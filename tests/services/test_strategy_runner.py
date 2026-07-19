"""StrategyRunner + AI overlay tests — PR-FAN.

Pure-Python with mocked Supabase + mocked market data.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.services.strategy_runner import (
    AIOverlayDecision,
    AIOverlaySettings,
    StrategyRunner,
    apply_ai_overlay,
    expand_universe,
    load_overlay_settings,
)


# ─────────────────────────────────────────────────────────────────────
# Universe expander
# ─────────────────────────────────────────────────────────────────────


class TestUniverseExpander:

    def test_single_requires_symbol(self):
        with pytest.raises(ValueError, match="requires single_symbol"):
            expand_universe("single", None)

    def test_single_returns_one_symbol(self):
        # cache-busting suffix because expand_universe is lru_cached
        assert expand_universe("single", "RELIANCE") == ["RELIANCE"]

    def test_sector_returns_curated_list(self):
        syms = expand_universe("sector:IT")
        assert "TCS" in syms
        assert "INFY" in syms
        assert len(syms) >= 5

    def test_unknown_universe_returns_empty(self):
        # Use a unique name so we don't pollute the cache for other tests
        assert expand_universe("unknown:CRYPTO") == []


# ─────────────────────────────────────────────────────────────────────
# AI Overlay
# ─────────────────────────────────────────────────────────────────────


class TestAIOverlay:

    def test_default_settings_block_bear_regime(self):
        settings = AIOverlaySettings()
        assert "bear" in settings.blocked_regimes
        assert settings.regime_gate_enabled is True

    def test_bear_regime_blocked(self):
        settings = AIOverlaySettings()
        decision = apply_ai_overlay(
            supabase=None, settings=settings, user_id="u1",
            symbol="RELIANCE", current_regime="bear", current_vix=15.0,
        )
        assert decision.allowed is False
        assert decision.block_reason and "regime_gate" in decision.block_reason

    def test_bull_regime_allowed(self):
        settings = AIOverlaySettings()
        decision = apply_ai_overlay(
            supabase=None, settings=settings, user_id="u1",
            symbol="RELIANCE", current_regime="bull", current_vix=14.0,
        )
        assert decision.allowed is True
        assert decision.size_multiplier == 1.0

    def test_vix_hard_block(self):
        settings = AIOverlaySettings()
        decision = apply_ai_overlay(
            supabase=None, settings=settings, user_id="u1",
            symbol="RELIANCE", current_regime="bull", current_vix=40.0,
        )
        assert decision.allowed is False
        assert "vix_hard_block" in decision.block_reason

    def test_vix_size_scaling(self):
        settings = AIOverlaySettings()
        # VIX 20 → 0.8x
        d20 = apply_ai_overlay(
            supabase=None, settings=settings, user_id="u1",
            symbol="X", current_regime="bull", current_vix=20.0,
        )
        assert d20.allowed is True
        assert d20.size_multiplier == 0.8
        # VIX 26 → 0.5x
        d26 = apply_ai_overlay(
            supabase=None, settings=settings, user_id="u1",
            symbol="X", current_regime="bull", current_vix=26.0,
        )
        assert d26.size_multiplier == 0.5
        # VIX 32 (between size-30 and hard-35) → 0.25x
        d32 = apply_ai_overlay(
            supabase=None, settings=settings, user_id="u1",
            symbol="X", current_regime="bull", current_vix=32.0,
        )
        assert d32.allowed is True
        assert d32.size_multiplier == 0.25

    def test_disabling_regime_gate_lets_bear_through(self):
        settings = AIOverlaySettings(regime_gate_enabled=False)
        decision = apply_ai_overlay(
            supabase=None, settings=settings, user_id="u1",
            symbol="X", current_regime="bear", current_vix=15.0,
        )
        assert decision.allowed is True


# ─────────────────────────────────────────────────────────────────────
# Runner end-to-end (mocked deps)
# ─────────────────────────────────────────────────────────────────────


def _mock_supabase():
    sb = MagicMock()
    chain = MagicMock()
    chain.execute.return_value = MagicMock(data=[], count=0)
    for attr in ("select", "eq", "neq", "in_", "lt", "gte", "lte",
                 "order", "limit", "insert", "update", "upsert", "delete"):
        getattr(chain, attr).return_value = chain
    sb.table.return_value = chain
    return sb


def _bullish_bars(n: int = 250) -> pd.DataFrame:
    """OHLCV with a clean uptrend so trend-following DSL fires."""
    rng = np.random.default_rng(42)
    base = np.linspace(100, 180, n) + rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "open":  base + rng.normal(0, 0.5, n),
        "high":  base + np.abs(rng.normal(0, 1, n)),
        "low":   base - np.abs(rng.normal(0, 1, n)),
        "close": base,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n))


class TestRunner:

    @pytest.mark.asyncio
    async def test_empty_user_list_returns_clean_report(self):
        sb = _mock_supabase()
        runner = StrategyRunner(sb)
        report = await runner.run_daily_tick()
        assert report.tick_kind == "daily"
        assert report.users_processed == 0
        assert report.signals_emitted == 0
        assert report.error is None

    @pytest.mark.asyncio
    async def test_runner_skips_users_without_live_strategies(self, monkeypatch):
        sb = _mock_supabase()
        runner = StrategyRunner(sb)
        # _load_active_users returns empty by default
        report = await runner.run_daily_tick()
        assert report.users_processed == 0

    @pytest.mark.asyncio
    async def test_runner_with_one_live_strategy_evaluates(self, monkeypatch):
        """Happy path: one user, one DSL strategy on a single symbol with
        bullish bars → at least one signal emitted."""
        sb = _mock_supabase()
        runner = StrategyRunner(sb)

        # Override the DB-reading methods so we don't have to mock supabase chains
        async def _fake_bars(self, sym):  # noqa: ARG001
            return _bullish_bars(250)

        monkeypatch.setattr(StrategyRunner, "_load_bars", _fake_bars)
        monkeypatch.setattr(StrategyRunner, "_current_vix", AsyncMock(return_value=14.0))
        monkeypatch.setattr(
            StrategyRunner, "_load_active_users",
            lambda self, *, user_filter=None: [{"id": "user-1"}],
        )
        monkeypatch.setattr(
            StrategyRunner, "_load_live_strategies_for",
            lambda self, user_id, timeframes: [{
                "id": "strat-1", "status": "live", "name": "EMA cross",
                "dsl": {
                    "name": "EMA cross",
                    "symbol": "RELIANCE",
                    "universe": "single",
                    "timeframe": "1d",
                    "entry": {"kind": "indicator_compare", "indicator": "close",
                              "op": ">", "value": "ema50"},
                    "exit":  {"kind": "indicator_compare", "indicator": "close",
                              "op": "<", "value": "ema50"},
                    "position_size": {"kind": "percent_of_capital", "value": 10},
                },
            }],
        )
        monkeypatch.setattr(
            StrategyRunner, "_load_open_positions",
            lambda self, user_id, strategy_id: [],
        )
        # Resolver returns bull → overlay allows
        from backend.services.strategy_runner import ai_overlay as ovr_mod
        monkeypatch.setattr(ovr_mod, "resolve_regime_at", lambda *a, **kw: "bull")
        from backend.services.strategy_runner import runner as runner_mod
        monkeypatch.setattr(runner_mod, "resolve_regime_at", lambda *a, **kw: "bull")

        report = await runner.run_daily_tick()
        assert report.users_processed == 1
        assert report.strategies_evaluated == 1
        assert report.symbol_evaluations >= 1
        # Bullish series + close > EMA50 → entry should fire
        assert report.entries_emitted >= 1

    @pytest.mark.asyncio
    async def test_runner_skipped_by_overlay_when_bear(self, monkeypatch):
        """When regime is bear and overlay blocks bear → entry skipped."""
        sb = _mock_supabase()
        runner = StrategyRunner(sb)

        async def _fake_bars(self, sym):  # noqa: ARG001
            return _bullish_bars(250)

        monkeypatch.setattr(StrategyRunner, "_load_bars", _fake_bars)
        monkeypatch.setattr(StrategyRunner, "_current_vix", AsyncMock(return_value=14.0))
        monkeypatch.setattr(
            StrategyRunner, "_load_active_users",
            lambda self, *, user_filter=None: [{"id": "user-1"}],
        )
        monkeypatch.setattr(
            StrategyRunner, "_load_live_strategies_for",
            lambda self, user_id, timeframes: [{
                "id": "strat-1", "status": "live", "name": "EMA cross",
                "dsl": {
                    "name": "EMA cross",
                    "symbol": "RELIANCE",
                    "universe": "single",
                    "timeframe": "1d",
                    "entry": {"kind": "indicator_compare", "indicator": "close",
                              "op": ">", "value": "ema50"},
                    "exit":  {"kind": "indicator_compare", "indicator": "close",
                              "op": "<", "value": "ema50"},
                    "position_size": {"kind": "percent_of_capital", "value": 10},
                },
            }],
        )
        monkeypatch.setattr(
            StrategyRunner, "_load_open_positions",
            lambda self, user_id, strategy_id: [],
        )
        # Bear regime — overlay blocks
        from backend.services.strategy_runner import ai_overlay as ovr_mod
        from backend.services.strategy_runner import runner as runner_mod
        monkeypatch.setattr(ovr_mod, "resolve_regime_at", lambda *a, **kw: "bear")
        monkeypatch.setattr(runner_mod, "resolve_regime_at", lambda *a, **kw: "bear")

        report = await runner.run_daily_tick()
        assert report.entries_emitted == 0
        assert report.skipped_by_overlay >= 1

    @pytest.mark.asyncio
    async def test_runner_handles_invalid_dsl_without_crashing(self, monkeypatch):
        """A strategy with junk DSL must be skipped, not crash the tick."""
        sb = _mock_supabase()
        runner = StrategyRunner(sb)
        monkeypatch.setattr(
            StrategyRunner, "_load_active_users",
            lambda self, *, user_filter=None: [{"id": "user-1"}],
        )
        monkeypatch.setattr(
            StrategyRunner, "_load_live_strategies_for",
            lambda self, user_id, timeframes: [{
                "id": "strat-bad", "status": "live", "name": "broken",
                "dsl": {"name": "broken", "this": "is not a valid strategy"},
            }],
        )
        monkeypatch.setattr(StrategyRunner, "_current_vix", AsyncMock(return_value=14.0))

        report = await runner.run_daily_tick()
        assert report.error is None
        assert report.entries_emitted == 0


# ─────────────────────────────────────────────────────────────────────
# Memory lock guards
# ─────────────────────────────────────────────────────────────────────


class TestMemoryLockGuards:
    """Runner + overlay must not import any LLM module."""

    def test_runner_does_not_import_llm(self):
        for mod_name in (
            "backend.services.strategy_runner.runner",
            "backend.services.strategy_runner.ai_overlay",
        ):
            import importlib
            mod = importlib.import_module(mod_name)
            src = open(mod.__file__).read()
            for forbidden in (
                "from ..copilot",
                "from ..assistant",
                "AssistantLLM",
                "AnthropicWrapper",
                "import openai",
                "import anthropic",
                "langchain",
            ):
                assert forbidden not in src, (
                    f"{mod_name} contains '{forbidden}' — this would breach the "
                    f"LLM-never-gates-trades lock."
                )
