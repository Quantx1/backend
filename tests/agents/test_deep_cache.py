import backend.ai.agents.response_cache as rc
from backend.ai.agents import llm
from backend.core.config import settings


def test_seconds_to_ist_eod_positive():
    assert rc.seconds_to_ist_eod() >= 300


def test_deep_routing_only_for_elite_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "LLM_DEEP_MODE_ENABLED", True)
    assert llm.resolve_model("doctor", deep=True, tier="elite") == settings.LLM_DEEP_MODEL
    assert llm.resolve_model("doctor", deep=True, tier="pro") == settings.LLM_STRONG_MODEL
    assert llm.resolve_model("debate", deep=True, tier="elite") == settings.LLM_DEEP_MODEL


def test_deep_off_by_default(monkeypatch):
    monkeypatch.setattr(settings, "LLM_DEEP_MODE_ENABLED", False)
    assert llm.resolve_model("doctor", deep=True, tier="elite") == settings.LLM_STRONG_MODEL
