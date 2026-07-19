"""PR-V — observability layer tests.

Covers:
  1. llm_pricing.micros_usd_for — known model maths + unknown model
     short-circuits to 0 (rather than raising) so callers never block.
  2. track() dual-write — Supabase mirror is called even when PostHog
     is disabled.
  3. track_llm_usage() — builds the correct row shape + handles missing
     pricing without crashing.

All pricing assertions use OpenRouter open-model slugs — the only
providers we bill against post-migration (Gemini/Anthropic removed
2026-06-04).

Heavy stuff (real Supabase, real PostHog) is mocked at the helper
boundary so these run on CPU in <1s.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────
# llm_pricing.micros_usd_for
# ─────────────────────────────────────────────────────────────────────────

def test_known_model_returns_expected_micros():
    from backend.observability.llm_pricing import micros_usd_for

    # llama-3.3-70b: $0.10/M input, $0.32/M output.
    # 1000 input + 500 output = 100 + 160 = 260 micros (= $0.00026)
    assert micros_usd_for(
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct",
        input_tokens=1000,
        output_tokens=500,
    ) == 260


def test_unknown_model_returns_zero():
    from backend.observability.llm_pricing import micros_usd_for

    out = micros_usd_for(
        provider="openrouter",
        model="qwen/nonexistent-9000",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert out == 0


def test_cache_read_uses_0_1x_input_price():
    from backend.observability.llm_pricing import micros_usd_for

    # llama-3.3-70b input is $0.10/M → cache_read is $0.01/M (0.1x).
    # 10_000 cache_read tokens = 10000 * 0.10 * 0.1 = 100 micros
    out = micros_usd_for(
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=10_000,
    )
    assert out == 100


def test_provider_case_insensitive():
    from backend.observability.llm_pricing import micros_usd_for

    a = micros_usd_for("openrouter", "meta-llama/llama-3.3-70b-instruct", 1000, 0)
    b = micros_usd_for("OpenRouter", "META-LLAMA/Llama-3.3-70B-Instruct", 1000, 0)
    assert a == b == 100


# ─────────────────────────────────────────────────────────────────────────
# track() dual-write
# ─────────────────────────────────────────────────────────────────────────

def _fake_supabase():
    sb = MagicMock()
    chain = MagicMock()
    chain.insert.return_value = chain
    chain.execute.return_value = SimpleNamespace(data=None)
    sb.table.return_value = chain
    return sb, chain


def test_track_mirrors_to_supabase_even_when_posthog_disabled():
    from backend.observability import posthog_events

    sb, chain = _fake_supabase()

    # Force PostHog client to None so we hit the Supabase-only path.
    with patch.object(posthog_events, "_get_client", return_value=None), \
         patch("backend.api.app.get_supabase_admin", return_value=sb):
        posthog_events.track("tier_gate_hit", "user-123", {"feature": "debate"})

    sb.table.assert_called_with("analytics_events")
    chain.insert.assert_called_once()
    payload = chain.insert.call_args[0][0]
    assert payload["event"] == "tier_gate_hit"
    assert payload["user_id"] == "user-123"
    assert payload["properties"] == {"feature": "debate"}


def test_track_swallows_supabase_failure():
    """A DB blip must never propagate up to the user-facing request."""
    from backend.observability import posthog_events

    sb = MagicMock()
    sb.table.side_effect = RuntimeError("supabase down")

    with patch.object(posthog_events, "_get_client", return_value=None), \
         patch("backend.api.app.get_supabase_admin", return_value=sb):
        # Must not raise.
        posthog_events.track("tier_gate_hit", "user-1", {})


def test_track_passes_eventname_enum_correctly():
    from backend.observability import posthog_events
    from backend.observability.posthog_events import EventName

    sb, chain = _fake_supabase()
    with patch.object(posthog_events, "_get_client", return_value=None), \
         patch("backend.api.app.get_supabase_admin", return_value=sb):
        posthog_events.track(EventName.TIER_GATE_HIT, "u1", {"feature": "x"})

    payload = chain.insert.call_args[0][0]
    # EventName.TIER_GATE_HIT.value should be the string written.
    assert payload["event"] == EventName.TIER_GATE_HIT.value


# ─────────────────────────────────────────────────────────────────────────
# track_llm_usage
# ─────────────────────────────────────────────────────────────────────────

def test_track_llm_usage_builds_correct_row():
    from backend.observability import posthog_events

    sb, chain = _fake_supabase()
    with patch("backend.api.app.get_supabase_admin", return_value=sb):
        posthog_events.track_llm_usage(
            user_id="user-xyz",
            feature="copilot",
            provider="openrouter",
            model="meta-llama/llama-3.3-70b-instruct",
            input_tokens=2000,
            output_tokens=1000,
            latency_ms=412,
            metadata={"hop": 3},
        )

    sb.table.assert_called_with("llm_usage_events")
    payload = chain.insert.call_args[0][0]
    assert payload["user_id"] == "user-xyz"
    assert payload["feature"] == "copilot"
    assert payload["provider"] == "openrouter"
    assert payload["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert payload["input_tokens"] == 2000
    assert payload["output_tokens"] == 1000
    assert payload["latency_ms"] == 412
    # llama-3.3-70b: $0.10 input, $0.32 output per 1M.
    # 2000 * 0.10 + 1000 * 0.32 = 200 + 320 = 520 micros
    assert payload["micros_usd"] == 520
    assert payload["metadata"] == {"hop": 3}


def test_track_llm_usage_unknown_model_does_not_raise():
    """Model not in price card → micros=0, row still written."""
    from backend.observability import posthog_events

    sb, chain = _fake_supabase()
    with patch("backend.api.app.get_supabase_admin", return_value=sb):
        posthog_events.track_llm_usage(
            user_id="u1",
            feature="copilot",
            provider="openrouter",
            model="qwen/fictional-9000",
            input_tokens=1000,
            output_tokens=500,
        )

    payload = chain.insert.call_args[0][0]
    assert payload["micros_usd"] == 0
    assert payload["input_tokens"] == 1000


def test_track_llm_usage_swallows_supabase_failure():
    from backend.observability import posthog_events

    sb = MagicMock()
    sb.table.side_effect = RuntimeError("supabase down")

    with patch("backend.api.app.get_supabase_admin", return_value=sb):
        # Must not raise — LLM-call telemetry must never break the caller.
        posthog_events.track_llm_usage(
            user_id="u1",
            feature="copilot",
            provider="openrouter",
            model="meta-llama/llama-3.3-70b-instruct",
            input_tokens=100,
            output_tokens=50,
        )
