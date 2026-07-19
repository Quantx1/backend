import backend.ai.agents.llm as llm
from backend.core.config import settings


def test_chat_fast_roles_use_chat_model():
    # Chat-critical roles run on the fast chat model (ultra-quick copilot).
    for role in ("responder", "classifier", "tool_planner"):
        assert llm.resolve_model(role) == settings.LLM_CHAT_MODEL


def test_fast_roles_use_free_model():
    assert llm.resolve_model("sentiment") == settings.LLM_FAST_MODEL


def test_strong_roles_use_strong_model():
    for role in ("grounded_reason", "doctor", "debate", "strategy_gen", "scanner_thesis", "fno_advisor"):
        assert llm.resolve_model(role) == settings.LLM_STRONG_MODEL


def test_unknown_role_falls_back_to_default():
    assert llm.resolve_model("nonsense") == settings.LLM_DEFAULT_MODEL


def test_env_map_overrides_in_code_default(monkeypatch):
    monkeypatch.setattr(llm, "_MODEL_MAP_CACHE", {"responder": "x/custom-model"})
    assert llm.resolve_model("responder") == "x/custom-model"


def test_deep_mode_off_by_default_keeps_strong(monkeypatch):
    monkeypatch.setattr(settings, "LLM_DEEP_MODE_ENABLED", False)
    assert llm.resolve_model("doctor", deep=True, tier="elite") == settings.LLM_STRONG_MODEL


def test_deep_mode_on_for_elite_uses_deep_model(monkeypatch):
    monkeypatch.setattr(settings, "LLM_DEEP_MODE_ENABLED", True)
    assert llm.resolve_model("doctor", deep=True, tier="elite") == settings.LLM_DEEP_MODEL
    assert llm.resolve_model("doctor", deep=True, tier="pro") == settings.LLM_STRONG_MODEL
    assert llm.resolve_model("responder", deep=True, tier="elite") == settings.LLM_STRONG_MODEL


def test_strong_model_has_free_fallback_for_429_spill():
    models = llm.build_models(settings.LLM_STRONG_MODEL, allow_paid=True)
    assert models and models[0] == settings.LLM_STRONG_MODEL
    assert any(m.endswith(":free") for m in models)
