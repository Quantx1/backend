"""Studio agent tests — PR-E.

Tests the NL→DSL compiler logic without making real LLM calls.
We mock ``_generate_dsl`` so tests are deterministic + fast.
"""

from __future__ import annotations

import json

import pytest

from backend.ai.strategy import studio
from backend.ai.strategy.dsl import Strategy
from backend.ai.strategy.studio import (
    StudioError,
    compile_strategy,
)


_VALID_DSL_JSON = json.dumps({
    "name": "RSI Mean Reversion",
    "universe": "nifty50",
    "timeframe": "1d",
    "entry": {"kind": "indicator_compare", "indicator": "rsi14", "op": "<", "value": 30},
    "exit":  {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 70},
    "stop_loss_pct": 4,
    "take_profit_pct": 8,
    "position_size": {"kind": "percent_of_capital", "value": 5},
    "regime_filter": "any",
    "lookback_days": 90,
    "mode": "backtest",
})


_INVALID_DSL_JSON = json.dumps({
    "name": "Bad strategy with unknown indicator",
    "universe": "nifty50",
    "timeframe": "1d",
    "entry": {"kind": "indicator_compare", "indicator": "BAD_INDICATOR", "op": "<", "value": 30},
    "exit": {"kind": "indicator_compare", "indicator": "rsi14", "op": ">", "value": 70},
    "stop_loss_pct": 4,
    "position_size": {"kind": "percent_of_capital", "value": 5},
})


_MALFORMED_JSON = "{name: 'broken', mode: backtest}"  # bareword keys/values


_FULL_PROMPT = "Buy RELIANCE when RSI drops below 30, exit when RSI rises above 70, stop loss 3%"


@pytest.fixture(autouse=True)
def _no_studio_cache(monkeypatch):
    """compile_strategy caches successful compiles (in-process L1 + shared
    Supabase L2) keyed by the normalized prompt. Several tests reuse the SAME
    prompt with different mocked generators (valid vs invalid vs malformed), so
    make the cache inert for this module to keep every case hermetic —
    otherwise an earlier success would satisfy a later 'must-raise' case from
    cache. compile_strategy imports these names at call time, so patching the
    module attributes takes effect."""
    from backend.ai.agents import response_cache
    monkeypatch.setattr(response_cache, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(response_cache, "cache_set", lambda *a, **k: None)


class TestCompileSuccess:

    def test_valid_dsl_first_try(self, monkeypatch):
        monkeypatch.setattr(studio, "_generate_dsl", lambda prompt: _VALID_DSL_JSON)
        strategy = compile_strategy("RSI mean reversion on Nifty 50")
        assert isinstance(strategy, Strategy)
        assert strategy.name == "RSI Mean Reversion"
        assert strategy.universe.value == "nifty50"
        assert strategy.mode.value == "backtest"

    def test_strips_code_fences(self, monkeypatch):
        fenced = "```json\n" + _VALID_DSL_JSON + "\n```"
        monkeypatch.setattr(studio, "_generate_dsl", lambda prompt: fenced)
        strategy = compile_strategy(_FULL_PROMPT)
        assert strategy.name == "RSI Mean Reversion"

    def test_forces_mode_backtest_even_if_llm_says_live(self, monkeypatch):
        live_attempt = json.loads(_VALID_DSL_JSON)
        live_attempt["mode"] = "live"
        monkeypatch.setattr(studio, "_generate_dsl", lambda prompt: json.dumps(live_attempt))
        strategy = compile_strategy(_FULL_PROMPT)
        assert strategy.mode.value == "backtest"


class TestCompileFailures:

    def test_empty_prompt_rejected(self):
        with pytest.raises(StudioError):
            compile_strategy("")

    def test_whitespace_only_prompt_rejected(self):
        with pytest.raises(StudioError):
            compile_strategy("   \n   ")

    def test_invalid_dsl_retried_then_raised(self, monkeypatch):
        calls = {"n": 0}

        def fake_call(prompt):
            calls["n"] += 1
            return _INVALID_DSL_JSON  # always invalid

        monkeypatch.setattr(studio, "_generate_dsl", fake_call)
        with pytest.raises(StudioError) as exc_info:
            compile_strategy(_FULL_PROMPT, max_retries=1)
        # 1 retry = 2 total attempts
        assert calls["n"] == 2
        assert "attempt 1" in str(exc_info.value)
        assert "attempt 2" in str(exc_info.value)

    def test_malformed_json_retried(self, monkeypatch):
        calls = {"n": 0}

        def fake_call(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return _MALFORMED_JSON
            return _VALID_DSL_JSON

        monkeypatch.setattr(studio, "_generate_dsl", fake_call)
        strategy = compile_strategy(_FULL_PROMPT, max_retries=1)
        assert calls["n"] == 2
        assert strategy.name == "RSI Mean Reversion"


class TestPromptIncludesIndicators:

    def test_prompt_lists_every_indicator(self):
        # _PROMPT must enumerate every indicator so the LLM never invents names
        from backend.ai.strategy.dsl import INDICATOR_REGISTRY
        from backend.ai.strategy.studio import _PROMPT

        for ind in INDICATOR_REGISTRY:
            assert ind in _PROMPT, f"indicator {ind} missing from Studio prompt"

    def test_prompt_lists_engine_value_semantics(self):
        from backend.ai.strategy.studio import _PROMPT
        # Every engine in the whitelist must appear with its value type
        # Horizon removed in PR-M cut (2026-05-25); Vision/Verdict/Pulse
        # removed in the engines cleanup later same day — no PROD model behind them.
        for engine in ("Regime", "Alpha", "Mood"):
            assert engine in _PROMPT


class TestClarification:

    def test_bare_prompt_returns_clarification_without_generator(self, monkeypatch):
        from backend.ai.strategy.studio import ClarificationNeeded
        called = {"n": 0}

        def spy(prompt):
            called["n"] += 1
            return _VALID_DSL_JSON

        monkeypatch.setattr(studio, "_generate_dsl", spy)
        result = compile_strategy("make me a profitable strategy")
        assert isinstance(result, ClarificationNeeded)
        assert called["n"] == 0
        assert "instrument" in result.missing

    def test_instrument_present_but_no_entry_or_exit_clarifies(self, monkeypatch):
        from backend.ai.strategy.studio import ClarificationNeeded
        monkeypatch.setattr(studio, "_generate_dsl", lambda p: _VALID_DSL_JSON)
        result = compile_strategy("trade nifty")
        assert isinstance(result, ClarificationNeeded)

    def test_generator_fold_clarification(self, monkeypatch):
        from backend.ai.strategy.studio import ClarificationNeeded
        clarify = json.dumps({
            "needs_clarification": True,
            "missing": ["exit"],
            "question": "When should the trade exit?",
            "assumptions": [],
        })
        called = {"n": 0}

        def fake(prompt):
            called["n"] += 1
            return clarify

        monkeypatch.setattr(studio, "_generate_dsl", fake)
        # Non-bare prompt so the deterministic pre-gate passes it to the generator.
        result = compile_strategy("Buy RELIANCE when 20EMA crosses above 50EMA and exit on the reverse cross")
        assert called["n"] == 1
        assert isinstance(result, ClarificationNeeded)
        assert result.missing == ["exit"]

    def test_precheck_passes_complete_prompt(self):
        from backend.ai.strategy.studio import precheck_clarification
        assert precheck_clarification(
            "Buy Nifty 50 when 20EMA crosses above 50EMA, exit on cross below, SL 3%"
        ) is None

    def test_full_prompt_still_compiles_to_strategy(self, monkeypatch):
        monkeypatch.setattr(studio, "_generate_dsl", lambda p: _VALID_DSL_JSON)
        result = compile_strategy(_FULL_PROMPT)
        assert isinstance(result, Strategy)
