"""LLM (OpenRouter-only) routing + budget kill-switch tests.

Gemini is removed from the agent path: an LLM is enabled iff OPENROUTER_API_KEY
is set, defaults to LLM_DEFAULT_MODEL when unmapped, and degrades gracefully
when over budget. No live HTTP — budget short-circuits before any call.
"""

from __future__ import annotations

import asyncio

from backend.ai.agents.llm import LLM, build_models
from backend.core.config import settings
import backend.observability.llm_budget as budmod


class TestRouting:
    def test_disabled_without_key(self, monkeypatch):
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
        assert LLM().enabled is False

    def test_enabled_with_key(self, monkeypatch):
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-or-test")
        assert LLM().enabled is True

    def test_default_model_when_unmapped(self, monkeypatch):
        monkeypatch.setattr(settings, "LLM_DEFAULT_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        assert LLM()._model == "meta-llama/llama-3.3-70b-instruct:free"

    def test_explicit_model(self):
        assert LLM(model="qwen/qwen3-32b")._model == "qwen/qwen3-32b"


class TestFallbackChain:
    def test_free_chain_has_paid_last_capped_at_3(self):
        ms = build_models("meta-llama/llama-3.3-70b-instruct:free", allow_paid=True)
        assert ms[0] == "meta-llama/llama-3.3-70b-instruct:free"
        assert len(ms) <= 3                                  # OpenRouter cap
        assert any(not m.endswith(":free") for m in ms)      # paid fallback present

    def test_over_budget_strips_paid(self):
        ms = build_models("meta-llama/llama-3.3-70b-instruct:free", allow_paid=False)
        assert ms is None or all(m.endswith(":free") for m in ms)

    def test_non_free_model_no_fallback(self):
        assert build_models("qwen/qwen3-32b", allow_paid=True) is None


class TestBudgetGuard:
    def test_free_model_bypasses_budget_even_when_over(self, monkeypatch):
        m = budmod.UsageMeter()
        m.record_micros(999_000_000)
        monkeypatch.setattr(budmod, "_meter", m)
        monkeypatch.setattr(budmod.UsageMeter, "maybe_refresh", lambda self, sb, ttl_seconds=60.0: None)
        LLM(model="x")._guard_budget("qwen/qwen3-coder:free")  # free → must not raise

    def test_paid_call_over_budget_degrades_gracefully(self, monkeypatch):
        m = budmod.UsageMeter()
        m.record_micros(999_000_000)
        monkeypatch.setattr(budmod, "_meter", m)
        monkeypatch.setattr(budmod.UsageMeter, "maybe_refresh", lambda self, sb, ttl_seconds=60.0: None)
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-or-test")
        out = asyncio.run(LLM(model="qwen/qwen3-32b").complete("hi"))
        assert "budget" in out.lower()

    def test_generate_json_over_budget_returns_empty(self, monkeypatch):
        m = budmod.UsageMeter()
        m.record_micros(999_000_000)
        monkeypatch.setattr(budmod, "_meter", m)
        monkeypatch.setattr(budmod.UsageMeter, "maybe_refresh", lambda self, sb, ttl_seconds=60.0: None)
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-or-test")
        out = asyncio.run(LLM(model="qwen/qwen3-32b").generate_json("plan", '{"x":1}'))
        assert out == {}

    def test_no_key_returns_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
        assert asyncio.run(LLM(model="qwen/qwen3-32b").complete("hi")) == ""
