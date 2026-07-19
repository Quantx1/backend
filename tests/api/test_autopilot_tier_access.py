"""AutoPilot tier-access contract (pricing v2: paper bot Free, live Pro+).

Locks the rule wired into auto_trader_routes: the fully-automated PAPER bot is
available to every tier; only going LIVE (real money) requires Pro+.
"""

from backend.core.tiers import Tier, resolve_autopilot_mode
from backend.middleware.tier_gate import can_use_feature


def test_free_user_is_always_paper():
    assert resolve_autopilot_mode(Tier.FREE, None) == "paper"
    # Even a stray live opt-in in config never flips a Free user to real money.
    assert resolve_autopilot_mode(Tier.FREE, {"mode": "live"}) == "paper"


def test_paid_tiers_default_live_but_respect_paper_optin():
    assert resolve_autopilot_mode(Tier.PRO, None) == "live"
    assert resolve_autopilot_mode(Tier.PRO, {"mode": "paper"}) == "paper"
    assert resolve_autopilot_mode(Tier.ELITE, None) == "live"


def test_live_autopilot_is_pro_gated_paper_is_free():
    # can_use_feature("auto_trader") == may go LIVE. Free may not; Pro/Elite may.
    assert can_use_feature(Tier.FREE, "auto_trader") is False
    assert can_use_feature(Tier.PRO, "auto_trader") is True
    assert can_use_feature(Tier.ELITE, "auto_trader") is True
