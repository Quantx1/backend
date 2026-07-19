"""PR-DEPTH tests — micro features + tick exit + premium gate + intent gate.

Pure unit tests with no DB / network. Mocks Supabase chains where needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────
# Micro features
# ─────────────────────────────────────────────────────────────────────


class TestMicroFeatures:

    def _make_ticks(self, n=20, base_price=100.0, with_book=True) -> pd.DataFrame:
        ts = pd.date_range(end="2026-05-25 10:00:00", periods=n, freq="1s")
        rng = np.random.default_rng(42)
        prices = base_price + np.cumsum(rng.normal(0, 0.02, n))
        rows = {
            "timestamp": ts, "symbol": ["NIFTY"] * n,
            "price": prices, "volume": rng.integers(50, 200, n),
        }
        if with_book:
            rows["bid_price"] = prices - 0.05
            rows["ask_price"] = prices + 0.05
            rows["bid_qty"] = rng.integers(100, 500, n)
            rows["ask_qty"] = rng.integers(100, 500, n)
        return pd.DataFrame(rows)

    def test_compute_micro_features_with_book(self):
        from backend.ai.microstructure import compute_micro_features
        ticks = self._make_ticks(n=30, with_book=True)
        result = compute_micro_features(ticks, symbol="NIFTY")
        assert result is not None
        assert result.symbol == "NIFTY"
        assert result.n_ticks_in_window >= 5
        # Spread should be ~0.10 (book made it 0.10 wide)
        assert 0.05 <= result.bid_ask_spread <= 0.20
        # Imbalance should be in [-1, 1]
        assert -1.0 <= result.order_imbalance <= 1.0
        # Tick momentum should be a finite number
        assert not np.isnan(result.tick_momentum)

    def test_compute_micro_features_without_book(self):
        """When bid/ask absent, gracefully falls back to half-spread + slope-based momentum."""
        from backend.ai.microstructure import compute_micro_features
        ticks = self._make_ticks(n=20, with_book=False)
        result = compute_micro_features(ticks, symbol="NIFTY")
        assert result is not None
        assert result.bid_ask_spread > 0  # fallback estimate
        assert -1.0 <= result.tick_momentum <= 1.0

    def test_too_few_ticks_returns_none(self):
        from backend.ai.microstructure import compute_micro_features
        ticks = self._make_ticks(n=2)
        result = compute_micro_features(ticks, symbol="NIFTY")
        assert result is None

    def test_compute_premium_slope(self):
        from backend.ai.microstructure import compute_premium_slope
        # Build a series that falls 1% over 30 seconds
        ts = pd.date_range(end="2026-05-25 10:00:00", periods=30, freq="1s")
        prices = np.linspace(100, 99, 30)  # 1% drop
        df = pd.DataFrame({"timestamp": ts, "price": prices, "volume": [100] * 30})
        slope = compute_premium_slope(df, window_seconds=30)
        assert slope is not None
        assert -0.02 < slope < 0  # roughly -1%

    def test_premium_slope_returns_none_for_short_window(self):
        from backend.ai.microstructure import compute_premium_slope
        df = pd.DataFrame({
            "timestamp": pd.date_range(end="2026-05-25 10:00:00", periods=3, freq="1s"),
            "price": [100, 99.5, 99],
        })
        assert compute_premium_slope(df) is None


# ─────────────────────────────────────────────────────────────────────
# Tick exit engine
# ─────────────────────────────────────────────────────────────────────


class TestTickExitEngine:

    def _ticks(self, prices, *, start_ts=None):
        start = start_ts or pd.Timestamp("2026-05-25 10:00:00")
        ts = [start + pd.Timedelta(seconds=i) for i in range(len(prices))]
        return pd.DataFrame({"timestamp": ts, "price": prices, "volume": [10] * len(prices)})

    def test_sl_hit_within_window(self):
        from backend.ai.exit_engine import walk_ticks_for_exit, TickExitConfig
        cfg = TickExitConfig(stop_loss_pct=0.10, take_profit_pct=0.20)
        # Entry 100, SL 90. Window walks down to 89.
        ticks = self._ticks([99, 95, 91, 89])
        decision = walk_ticks_for_exit(
            ticks, entry_price=100, current_sl=90, current_target=120,
            current_peak=100, trailing_active=False, config=cfg,
        )
        assert decision.exit_reason == "stop_loss"
        assert decision.exit_price == 90

    def test_target_hit_within_window(self):
        from backend.ai.exit_engine import walk_ticks_for_exit, TickExitConfig
        cfg = TickExitConfig(stop_loss_pct=0.10, take_profit_pct=0.20)
        ticks = self._ticks([105, 110, 118, 125])
        decision = walk_ticks_for_exit(
            ticks, entry_price=100, current_sl=90, current_target=120,
            current_peak=100, trailing_active=False, config=cfg,
        )
        assert decision.exit_reason == "target"
        assert decision.exit_price == 120

    def test_trailing_ratchets_during_walk(self):
        """Walk that hits +25% then reverses ~10% — trailing should activate."""
        from backend.ai.exit_engine import walk_ticks_for_exit, TickExitConfig
        cfg = TickExitConfig(
            stop_loss_pct=0.10, take_profit_pct=0.50,
            trailing_trigger_pct=0.12, trailing_lock_pct=0.08,
        )
        ticks = self._ticks([108, 115, 125, 130, 128, 126])
        decision = walk_ticks_for_exit(
            ticks, entry_price=100, current_sl=90, current_target=150,
            current_peak=100, trailing_active=False, config=cfg,
        )
        # We hit peak 130 = +30% gain
        assert decision.new_peak >= 130
        # Trailing should have activated, SL moved above entry
        assert decision.trailing_active is True
        assert decision.new_sl > 90      # ratcheted up from initial 90

    def test_no_exit_returns_updated_state(self):
        """Walk with no SL/TP hit just updates trailing state."""
        from backend.ai.exit_engine import walk_ticks_for_exit, TickExitConfig
        cfg = TickExitConfig(stop_loss_pct=0.10, take_profit_pct=0.50)
        ticks = self._ticks([102, 104, 106, 108])
        decision = walk_ticks_for_exit(
            ticks, entry_price=100, current_sl=90, current_target=150,
            current_peak=100, trailing_active=False, config=cfg,
        )
        assert decision.exit_price is None
        assert decision.new_peak == 108  # updated to last high
        assert decision.ticks_walked == 4

    def test_empty_window_no_op(self):
        from backend.ai.exit_engine import walk_ticks_for_exit, TickExitConfig
        cfg = TickExitConfig(stop_loss_pct=0.10, take_profit_pct=0.20)
        decision = walk_ticks_for_exit(
            pd.DataFrame(columns=["timestamp", "price"]),
            entry_price=100, current_sl=90, current_target=120,
            current_peak=100, trailing_active=False, config=cfg,
        )
        assert decision.exit_price is None
        assert decision.ticks_walked == 0

    def test_engine_lifecycle(self):
        from backend.ai.exit_engine import TickExitEngine
        engine = TickExitEngine(entry_price=100, stop_loss_pct=0.10, take_profit_pct=0.50)
        # No exit yet
        engine.step(self._ticks([102, 105]))
        assert engine.is_closed is False
        # SL hit on next window
        engine.step(self._ticks([95, 91, 89]))
        assert engine.is_closed is True
        assert engine.close_reason in ("stop_loss", "trailing_sl")


# ─────────────────────────────────────────────────────────────────────
# Stagnation trailing
# ─────────────────────────────────────────────────────────────────────


class TestStagnationTrailing:

    def test_no_boost_when_recent_peak(self):
        from backend.ai.exit_engine.stagnation_trailing import update_stagnation_trailing, StagnationTrailingState
        state = StagnationTrailingState(peak_price=130, peak_bar_idx=10, current_sl=110)
        # Only 2 bars since peak — no boost
        out = update_stagnation_trailing(
            state, entry_price=100, current_bar_idx=12, base_retention=0.50,
        )
        assert out == 0.50

    def test_gentle_boost_5_to_10_bars(self):
        from backend.ai.exit_engine.stagnation_trailing import update_stagnation_trailing, StagnationTrailingState
        state = StagnationTrailingState(peak_price=130, peak_bar_idx=10, current_sl=110)
        out = update_stagnation_trailing(
            state, entry_price=100, current_bar_idx=17, base_retention=0.50,
        )
        assert out == pytest.approx(0.56, abs=0.01)

    def test_hard_boost_20_bars(self):
        from backend.ai.exit_engine.stagnation_trailing import update_stagnation_trailing, StagnationTrailingState
        state = StagnationTrailingState(peak_price=130, peak_bar_idx=10, current_sl=110)
        out = update_stagnation_trailing(
            state, entry_price=100, current_bar_idx=35, base_retention=0.50,
        )
        # 20+ bars → +20% boost, capped at 0.90
        assert out == pytest.approx(0.70, abs=0.01)

    def test_skipped_when_gain_too_small(self):
        from backend.ai.exit_engine.stagnation_trailing import update_stagnation_trailing, StagnationTrailingState
        # Only 10% gain → below 15% threshold
        state = StagnationTrailingState(peak_price=110, peak_bar_idx=10, current_sl=105)
        out = update_stagnation_trailing(
            state, entry_price=100, current_bar_idx=30, base_retention=0.50,
        )
        assert out == 0.50  # no boost


# ─────────────────────────────────────────────────────────────────────
# Premium gate
# ─────────────────────────────────────────────────────────────────────


class TestPremiumGate:

    def _mock_sb_with_ticks(self, ticks):
        sb = MagicMock()
        chain = MagicMock()
        chain.execute.return_value = MagicMock(data=ticks)
        for attr in ("select", "eq", "gte", "order", "limit"):
            getattr(chain, attr).return_value = chain
        sb.table.return_value = chain
        return sb

    def test_no_supabase_fails_open(self):
        from backend.services.strategy_runner.premium_gate import check_premium_slope
        result = check_premium_slope(None, symbol="X")
        assert result.allowed is True

    def test_insufficient_ticks_fails_open(self):
        from backend.services.strategy_runner.premium_gate import check_premium_slope
        sb = self._mock_sb_with_ticks([{"timestamp": "2026-01-01T00:00:00", "price": 100}])
        result = check_premium_slope(sb, symbol="X")
        assert result.allowed is True

    def test_blocks_on_steep_drop(self):
        """5+ ticks showing 2% drop should block."""
        from backend.services.strategy_runner.premium_gate import check_premium_slope
        ticks = [
            {"timestamp": "2026-01-01T10:00:00", "price": 100},
            {"timestamp": "2026-01-01T10:00:10", "price": 99},
            {"timestamp": "2026-01-01T10:00:20", "price": 98.5},
            {"timestamp": "2026-01-01T10:00:25", "price": 98},
            {"timestamp": "2026-01-01T10:00:30", "price": 97.5},
        ]
        sb = self._mock_sb_with_ticks(ticks)
        result = check_premium_slope(sb, symbol="X")
        assert result.allowed is False
        assert result.slope_pct < -0.008
        assert "premium_slope" in result.block_reason

    def test_allows_on_flat_or_rising(self):
        from backend.services.strategy_runner.premium_gate import check_premium_slope
        ticks = [
            {"timestamp": "2026-01-01T10:00:00", "price": 100},
            {"timestamp": "2026-01-01T10:00:10", "price": 100.5},
            {"timestamp": "2026-01-01T10:00:15", "price": 100.2},
            {"timestamp": "2026-01-01T10:00:20", "price": 100.8},
            {"timestamp": "2026-01-01T10:00:25", "price": 101.0},
        ]
        sb = self._mock_sb_with_ticks(ticks)
        result = check_premium_slope(sb, symbol="X")
        assert result.allowed is True


# ─────────────────────────────────────────────────────────────────────
# Continuation intent gate (StrategyRunner._prev_bar_confirms_long)
# ─────────────────────────────────────────────────────────────────────


class TestContinuationIntent:

    def _runner_with_mock_sb(self):
        from backend.services.strategy_runner.runner import StrategyRunner
        sb = MagicMock()
        return StrategyRunner(sb)

    def test_prev_bar_bullish_allows(self):
        runner = self._runner_with_mock_sb()
        # Prev bar closed above its open by 1% → allow continuation
        bars = pd.DataFrame({
            "open":  [100, 100, 100],
            "high":  [102, 102, 102],
            "low":   [99, 99, 99],
            "close": [100, 101, 101],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2026-01-01", periods=3))
        assert runner._prev_bar_confirms_long(bars) is True

    def test_prev_bar_strongly_bearish_blocks(self):
        runner = self._runner_with_mock_sb()
        # Prev bar dropped 2% → continuation should be blocked
        bars = pd.DataFrame({
            "open":  [100, 100, 100],
            "high":  [102, 102, 102],
            "low":   [97, 97, 97],
            "close": [100, 98, 98],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2026-01-01", periods=3))
        assert runner._prev_bar_confirms_long(bars) is False

    def test_too_few_bars_fails_open(self):
        runner = self._runner_with_mock_sb()
        bars = pd.DataFrame({
            "open": [100], "high": [102], "low": [99], "close": [101], "volume": [1000],
        }, index=pd.date_range("2026-01-01", periods=1))
        assert runner._prev_bar_confirms_long(bars) is True


# ─────────────────────────────────────────────────────────────────────
# RL exit agent scaffold (MUST be inert when not enabled)
# ─────────────────────────────────────────────────────────────────────


class TestRLExitScaffold:
    """RL is ENABLED by default per user 2026-05-25 approval — see
    memory/project_rl_exit_enabled_2026_05_25.md. These tests verify
    the agent behaves correctly when enabled OR disabled."""

    def test_enabled_by_default(self):
        from backend.ai.exit_engine.rl_exit_scaffold import ENABLE_RL_EXIT
        # Default ENABLE_RL_EXIT should be True after the 2026-05-25 user decision
        assert ENABLE_RL_EXIT is True

    def test_decide_returns_hold_when_qtable_empty(self):
        """Even when enabled, an empty Q-table → HOLD (cold-start safety)."""
        from backend.ai.exit_engine.rl_exit_scaffold import RLExitAgent, compute_rl_state
        agent = RLExitAgent()
        # is_loaded is False until Q-table loaded; decide() falls back to HOLD
        assert agent.is_loaded is False
        state = compute_rl_state(
            entry_price=100, current_price=105, bars_held=10, max_hold_bars=40,
            sl=95, target=130, trailing_active=False, peak_price=105,
            price_history=[100, 102, 105],
        )
        assert agent.decide(state) == "HOLD"

    def test_decide_returns_hold_when_disabled_via_flag(self, monkeypatch):
        """Setting agent.is_enabled=False at runtime still cleanly disables."""
        from backend.ai.exit_engine.rl_exit_scaffold import RLExitAgent, compute_rl_state
        agent = RLExitAgent()
        agent.is_enabled = False
        agent.is_loaded = True  # pretend Q-table is loaded
        agent.q_table = {(0, 0, 0, 0, 0, 0, 0, 0): [0.0, 1.0, 0.0]}  # would say EXIT
        state = compute_rl_state(
            entry_price=100, current_price=105, bars_held=10, max_hold_bars=40,
            sl=95, target=130, trailing_active=False, peak_price=105,
            price_history=[100, 102, 105],
        )
        # Disabled → HOLD regardless of Q-table contents
        assert agent.decide(state) == "HOLD"

    def test_status_documents_user_approval(self):
        from backend.ai.exit_engine.rl_exit_scaffold import rl_exit_status
        status = rl_exit_status()
        assert status["enabled_flag"] is True
        assert "2026-05-25" in status["user_approval"]
        assert "narrow_exit_only" in status["scope"]
        # Hard safety statement must still be present
        assert "Hard SL" in status["safety"]


# ─────────────────────────────────────────────────────────────────────
# Outcome model trainer
# ─────────────────────────────────────────────────────────────────────


class TestOutcomeModelScaffold:

    def test_insufficient_samples_skipped(self):
        from backend.ai.outcome_models import OutcomeModelTrainer, OutcomeModelConfig

        sb = MagicMock()
        chain = MagicMock()
        chain.execute.return_value = MagicMock(data=[
            {"won": True, "features_at_entry": {"rsi14": 35}, "regime_at_entry": "bull",
             "vix_at_entry": 14, "pnl_pct": 1.2, "result": "TARGET",
             "exit_at": "2026-01-01T15:00:00"},
        ] * 5)  # Only 5 samples, threshold is 30
        for attr in ("select", "eq", "order", "limit"):
            getattr(chain, attr).return_value = chain
        sb.table.return_value = chain

        trainer = OutcomeModelTrainer(sb)
        result = trainer.train(OutcomeModelConfig(template_slug="rsi-mean-reversion"))
        assert result.trained is False
        assert "insufficient" in result.skipped_reason

    def test_features_builder_returns_dict(self):
        from backend.ai.outcome_models import build_outcome_features
        bars = pd.DataFrame({
            "open": np.linspace(100, 110, 200),
            "high": np.linspace(101, 111, 200),
            "low":  np.linspace(99, 109, 200),
            "close": np.linspace(100, 110, 200),
            "volume": [1000] * 200,
        }, index=pd.date_range("2026-01-01", periods=200))
        features = build_outcome_features(bars, regime="bull", vix=14.0)
        assert isinstance(features, dict)
        assert "close" in features
        assert features.get("regime_bull") == 1.0
        assert features.get("vix") == 14.0


# ─────────────────────────────────────────────────────────────────────
# Memory lock guards
# ─────────────────────────────────────────────────────────────────────


class TestMemoryLockGuards:

    def test_no_llm_imports_in_pr_depth_modules(self):
        """PR-DEPTH modules are pure rules/math/ML, no LLM allowed."""
        modules = (
            "backend.ai.microstructure.features",
            "backend.ai.exit_engine.tick_exit",
            "backend.ai.exit_engine.stagnation_trailing",
            "backend.ai.exit_engine.rl_exit_scaffold",
            "backend.ai.outcome_models.trainer",
            "backend.services.strategy_runner.premium_gate",
            "backend.data.tick_collector.collector",
        )
        for mod_name in modules:
            import importlib
            mod = importlib.import_module(mod_name)
            src = open(mod.__file__).read()
            for forbidden in (
                "from ..copilot",
                "AssistantLLM",
                "AnthropicWrapper",
                "import openai",
                "import anthropic",
                "langchain",
            ):
                assert forbidden not in src, (
                    f"{mod_name} contains '{forbidden}' — breaches "
                    f"LLM-never-gates-trades lock."
                )
