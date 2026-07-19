"""Persistent per-(entity, day) cache for the ad-hoc LLM thesis callers.

Covers chart_patterns/explain.py `_llm_thesis` (sync, complete_sync) and
screener_v2/enrich.py `_llm_thesis` (async, llm_for("scanner_thesis")).
Convention follows tests/agents/test_grounded_cache.py: clear rc._L1 per
test + stub rc._sb so no test ever touches Supabase.
"""
import asyncio

import backend.ai.agents.response_cache as rc
from ml.features.patterns import BreakoutSignal, PatternResult
from backend.services.chart_patterns.explain import (
    SuggestedLevels as PatternLevels,
)
from backend.services.chart_patterns.explain import _llm_thesis as pattern_thesis
from backend.services.screener_v2 import enrich


def setup_function(_):
    rc._L1.clear()


# ── chart_patterns/explain.py fixtures ──────────────────────────────


def _pattern_fixtures():
    pat = PatternResult(
        pattern_type="ascending_triangle",
        support_line=None,
        resistance_line=None,
        duration_bars=30,
        breakout_level=105.0,
        support_level=100.0,
        pattern_height=5.0,
        quality_score=0.8,
    )
    sig = BreakoutSignal(
        pattern=pat, entry_price=105.5, stop_loss=101.0,
        target=112.0, volume_ratio=1.6,
    )
    levels = PatternLevels(
        entry=105.5, stop=101.0, stop_basis="pattern_low",
        target1=112.0, target1_basis="pattern_height", risk_reward=1.44,
    )
    return pat, sig, levels


def test_pattern_thesis_served_from_cache_on_repeat(monkeypatch):
    monkeypatch.setattr(rc, "_sb", lambda: None)
    calls = {"n": 0}

    def _fake_complete_sync(*_a, **_k):
        calls["n"] += 1
        return "A factual thesis."

    monkeypatch.setattr("backend.ai.agents.llm.complete_sync", _fake_complete_sync)
    pat, sig, levels = _pattern_fixtures()

    t1 = pattern_thesis("TCS", pat, sig, levels, "bull", 0.7)
    t2 = pattern_thesis("TCS", pat, sig, levels, "bull", 0.7)
    assert t1 == "A factual thesis."
    assert t2 == t1
    assert calls["n"] == 1


def test_pattern_thesis_empty_is_not_cached(monkeypatch):
    monkeypatch.setattr(rc, "_sb", lambda: None)
    calls = {"n": 0}

    def _fake_complete_sync(*_a, **_k):
        calls["n"] += 1
        return ""

    monkeypatch.setattr("backend.ai.agents.llm.complete_sync", _fake_complete_sync)
    pat, sig, levels = _pattern_fixtures()

    assert pattern_thesis("TCS", pat, sig, levels, "bull", 0.7) is None
    assert pattern_thesis("TCS", pat, sig, levels, "bull", 0.7) is None
    assert calls["n"] == 2   # failures retry — no negative caching


# ── screener_v2/enrich.py fixtures ──────────────────────────────────


class _FakeAgent:
    """Stand-in for llm_for("scanner_thesis") — async counting .complete()."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def complete(self, *_a, **_k):
        self.calls += 1
        return self.reply


def _enrich_inputs():
    indicators = [
        enrich.IndicatorReading(name="RSI(14)", value=45.0, status="bullish"),
        enrich.IndicatorReading(name="MACD histogram", value=-0.2, status="bearish"),
    ]
    levels = enrich.SuggestedLevels(
        entry=100.0, stop=96.0, target1=106.0, risk_reward=1.5,
    )
    return indicators, levels


def test_enrich_thesis_served_from_cache_on_repeat(monkeypatch):
    monkeypatch.setattr(rc, "_sb", lambda: None)
    fake = _FakeAgent("A factual thesis.")
    monkeypatch.setattr("backend.ai.agents.llm.llm_for", lambda *_a, **_k: fake)
    indicators, levels = _enrich_inputs()

    t1 = asyncio.run(enrich._llm_thesis(
        "INFY", indicators, levels, None, None, None, "bull"))
    t2 = asyncio.run(enrich._llm_thesis(
        "INFY", indicators, levels, None, None, None, "bull"))
    assert t1 == "A factual thesis."
    assert t2 == t1
    assert fake.calls == 1


def test_enrich_thesis_empty_is_not_cached(monkeypatch):
    monkeypatch.setattr(rc, "_sb", lambda: None)
    fake = _FakeAgent("")
    monkeypatch.setattr("backend.ai.agents.llm.llm_for", lambda *_a, **_k: fake)
    indicators, levels = _enrich_inputs()

    r1 = asyncio.run(enrich._llm_thesis(
        "INFY", indicators, levels, None, None, None, "bull"))
    r2 = asyncio.run(enrich._llm_thesis(
        "INFY", indicators, levels, None, None, None, "bull"))
    assert r1 is None
    assert r2 is None
    assert fake.calls == 2   # failures retry — no negative caching
