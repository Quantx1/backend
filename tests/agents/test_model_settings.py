from backend.core.config import settings
from backend.observability.llm_pricing import is_paid, micros_usd_for


def test_new_model_settings_exist_with_defaults():
    assert settings.LLM_STRONG_MODEL == "qwen/qwen3-235b-a22b-2507"
    assert settings.LLM_FAST_MODEL == "meta-llama/llama-3.3-70b-instruct:free"
    assert settings.LLM_DEEP_MODEL == "deepseek/deepseek-v3.2"
    assert settings.LLM_DEEP_MODE_ENABLED is False


def test_budget_default_is_fifty():
    assert settings.LLM_MONTHLY_BUDGET_USD == 50.0


def test_strong_and_deep_models_are_priced_so_budget_gating_works():
    assert is_paid("openrouter", settings.LLM_STRONG_MODEL) is True
    assert is_paid("openrouter", settings.LLM_DEEP_MODEL) is True
    assert micros_usd_for("openrouter", settings.LLM_STRONG_MODEL, 1000, 1000) > 0


def test_fast_model_is_free():
    assert is_paid("openrouter", settings.LLM_FAST_MODEL) is False
