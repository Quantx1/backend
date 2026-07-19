"""LLM price card — per-token USD pricing for the providers we actually use.

Prices are in **dollars per 1M tokens** (the standard provider quoting unit).
We convert to **micro-USD** at write time so ``llm_usage_events.micros_usd``
is a single BIGINT a SQL SUM can roll up.

Prices as of 2026-05. Update when providers re-price. A model not in the
table returns 0 — we'd rather under-report than block the calling code.

References:
  * OpenRouter — https://openrouter.ai/models
  * OpenAI — https://openai.com/api/pricing/
"""
from __future__ import annotations

from typing import Dict

# (input, output) USD per 1M tokens. Cache reads/writes use a
# 0.1x / 1.25x convention by default (see below).
_PRICE_PER_M_TOKENS: Dict[tuple[str, str], tuple[float, float]] = {
    # ── OpenAI (placeholder — we don't currently call OpenAI but the
    # ApiError class supports it if a feature ever needs it). ────────
    ("openai", "gpt-4o"): (2.5, 10.0),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),

    # ── Open-weight models via OpenRouter (the migration target). Slugs +
    # prices are LIVE-VERIFIED from openrouter.ai/api/v1/models on 2026-06-03.
    # Use the exact OpenRouter slug (org/model) in AGENT_MODEL_MAP. ``:free``
    # slugs are priced 0 so they never count toward the $20 budget.
    ("openrouter", "qwen/qwen3-8b"): (0.05, 0.40),
    ("openrouter", "qwen/qwen3-32b"): (0.08, 0.28),
    ("openrouter", "qwen/qwen3-coder-30b-a3b-instruct"): (0.07, 0.27),
    ("openrouter", "qwen/qwen3-235b-a22b-2507"): (0.071, 0.10),
    ("openrouter", "qwen/qwen3.5-9b"): (0.04, 0.15),
    ("openrouter", "deepseek/deepseek-v4-flash"): (0.098, 0.197),
    ("openrouter", "deepseek/deepseek-v3.2"): (0.229, 0.343),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"): (0.10, 0.32),
    ("openrouter", "z-ai/glm-4.7-flash"): (0.06, 0.40),
    ("openrouter", "mistralai/mistral-small-3.2-24b-instruct"): (0.075, 0.20),
    ("openrouter", "google/gemma-3-27b-it"): (0.08, 0.16),
    ("openrouter", "openai/gpt-oss-120b"): (0.039, 0.18),  # paid fallback for :free
    # Free-tier slugs (cost $0 → bypass the budget kill-switch):
    ("openrouter", "qwen/qwen3-coder:free"): (0.0, 0.0),
    ("openrouter", "z-ai/glm-4.5-air:free"): (0.0, 0.0),
    ("openrouter", "openai/gpt-oss-120b:free"): (0.0, 0.0),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"): (0.0, 0.0),
    ("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free"): (0.0, 0.0),
}

# Prompt-cache pricing convention (relative to input price).
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 1.25


def micros_usd_for(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> int:
    """Return total cost in micro-USD for one LLM call.

    Returns 0 when the (provider, model) pair isn't priced — caller
    still gets the row, just with cost=0 (visible as "no price card"
    on the admin panel).
    """
    key = (provider.lower().strip(), model.lower().strip())
    if key not in _PRICE_PER_M_TOKENS:
        return 0

    in_price_per_m, out_price_per_m = _PRICE_PER_M_TOKENS[key]
    # USD = (tokens / 1_000_000) * price_per_m  →  micros = tokens * price_per_m
    micros = int(input_tokens) * in_price_per_m
    micros += int(output_tokens) * out_price_per_m
    micros += int(cache_read_tokens) * in_price_per_m * _CACHE_READ_MULT
    micros += int(cache_write_tokens) * in_price_per_m * _CACHE_WRITE_MULT
    return int(round(micros))


def is_paid(provider: str, model: str) -> bool:
    """True when (provider, model) has a non-zero price. Free-tier models
    (priced 0) and unpriced models return False — so the kill-switch only
    blocks calls that actually spend money."""
    key = (provider.lower().strip(), model.lower().strip())
    price = _PRICE_PER_M_TOKENS.get(key)
    return bool(price) and (price[0] > 0 or price[1] > 0)
