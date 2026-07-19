"""Pricing v2 (2026-06-12) — AutoPilot tier packaging tests.

AutoPilot Lite on PRO (₹2L deployed cap, 8 positions, equity-only),
unlimited on ELITE, paper-only on FREE. Pure logic — no DB, no network.
"""
from backend.core.tiers import (
    Tier,
    auto_trader_limits,
    llm_feature_cap,
    required_tier,
    resolve_autopilot_mode,
)
from backend.trading.autopilot_service import apply_tier_limits


# ── feature matrix placement ─────────────────────────────────────────────


def test_auto_trader_is_pro_minimum():
    assert required_tier("auto_trader") == Tier.PRO


def test_auto_trader_unlimited_is_elite():
    assert required_tier("auto_trader_unlimited") == Tier.ELITE


def test_chat_caps_match_pricing_page():
    # The pricing page prints these numbers — backend must match.
    assert llm_feature_cap("chat", Tier.FREE) == 5
    assert llm_feature_cap("chat", Tier.PRO) == 150
    assert llm_feature_cap("chat", Tier.ELITE) == 400


# ── tier limits ──────────────────────────────────────────────────────────


def test_pro_limits_are_lite():
    lim = auto_trader_limits(Tier.PRO)
    assert lim["max_deployed_capital"] == 200_000.0
    assert lim["max_concurrent_positions"] == 8
    assert lim["allow_fno"] is False
    assert lim["paper_only"] is False


def test_elite_limits_uncapped():
    lim = auto_trader_limits("elite")  # str form accepted
    assert lim["max_deployed_capital"] is None
    assert lim["max_concurrent_positions"] == 15
    assert lim["allow_fno"] is True


def test_free_limits_paper_only_lite_shaped():
    # Free's paper demo mirrors exactly what Pro buys.
    lim = auto_trader_limits(Tier.FREE)
    assert lim["paper_only"] is True
    assert lim["max_deployed_capital"] == 200_000.0
    assert lim["max_concurrent_positions"] == 8


# ── mode resolution ──────────────────────────────────────────────────────


def test_free_is_always_paper():
    assert resolve_autopilot_mode(Tier.FREE, None) == "paper"
    assert resolve_autopilot_mode(Tier.FREE, {"mode": "live"}) == "paper"


def test_pro_defaults_live_but_paper_opt_in_persists():
    assert resolve_autopilot_mode(Tier.PRO, None) == "live"
    assert resolve_autopilot_mode(Tier.PRO, {}) == "live"
    # The upgrade-safety invariant: a Free user's paper opt-in survives a
    # tier upgrade — going live is always an explicit action.
    assert resolve_autopilot_mode(Tier.PRO, {"mode": "paper"}) == "paper"
    assert resolve_autopilot_mode(Tier.ELITE, {"mode": "paper"}) == "paper"


# ── apply_tier_limits (used inside _rebalance_one) ───────────────────────


def test_capital_clamped_to_deployed_cap():
    w, c = apply_tier_limits({"A": 0.05}, 1_000_000.0, auto_trader_limits(Tier.PRO))
    assert c == 200_000.0
    assert w == {"A": 0.05}


def test_capital_unclamped_for_elite():
    w, c = apply_tier_limits({"A": 0.05}, 1_000_000.0, auto_trader_limits(Tier.ELITE))
    assert c == 1_000_000.0


def test_small_capital_not_raised_to_cap():
    _, c = apply_tier_limits({}, 50_000.0, auto_trader_limits(Tier.PRO))
    assert c == 50_000.0


def test_weights_trimmed_to_top_k_by_conviction():
    weights = {f"S{i}": 0.05 - i * 0.002 for i in range(12)}  # S0 highest
    w, _ = apply_tier_limits(weights, 100_000.0, auto_trader_limits(Tier.PRO))
    assert len(w) == 8
    assert set(w) == {f"S{i}" for i in range(8)}  # top-8 kept


def test_weights_untouched_when_under_k():
    weights = {"A": 0.05, "B": 0.04}
    w, _ = apply_tier_limits(weights, 100_000.0, auto_trader_limits(Tier.PRO))
    assert w == weights
