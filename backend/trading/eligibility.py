"""
PR 130 — live-trade eligibility checker.

Centralises the preconditions every live-execution path must satisfy
before a real-money order leaves the building. Called by:

  * F4 AutoPilot (PR 131) — before each daily rebalance order
  * ``TradeExecutionService._execute_live_trade`` — before manual live
    execution (defense in depth; the route layer also gates by tier)

Returns a structured ``LiveTradeEligibility`` so callers can:
  - report a precise failure reason in the UI / logs / telemetry
  - distinguish between user-correctable (no broker connected) vs
    ops-level (global kill switch active) failures
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from supabase import Client

from ..core.tiers import Tier, meets_tier, resolve_user_tier
from ..platform.system_flags import global_halt_reason, is_globally_halted

logger = logging.getLogger(__name__)


@dataclass
class LiveTradeEligibility:
    eligible: bool
    reason: Optional[str] = None
    code: Optional[str] = None  # machine-readable failure category

    def __bool__(self) -> bool:  # convenience: ``if eligibility:``
        return self.eligible


# Failure code → human reason map. Keep slugs short + stable; analytics
# joins on these.
_REASONS = {
    "global_kill_switch": "Live trading is currently halted by ops.",
    "tier_too_low": "Live execution requires Elite (AutoPilot) tier.",
    "no_broker": "Connect a broker (Zerodha / Upstox / Angel One) before going live.",
    "broker_disconnected": "Your broker session expired. Reconnect in Settings → Broker.",
    "user_kill_switch": "Your account has live execution paused.",
    "missing_user_id": "Cannot resolve user_id for the live trade.",
}


def check_live_trade_eligibility(
    *,
    user_id: Optional[str],
    supabase: Client,
    require_tier: Tier = Tier.ELITE,
) -> LiveTradeEligibility:
    """Run every live-execution precondition and return a structured verdict.

    Order matters: the global kill switch comes first because it's the
    fastest deny path and never depends on per-user state.
    """
    if is_globally_halted(supabase_client=supabase):
        return LiveTradeEligibility(
            eligible=False,
            code="global_kill_switch",
            reason=global_halt_reason(supabase_client=supabase) or _REASONS["global_kill_switch"],
        )

    if not user_id:
        return LiveTradeEligibility(False, code="missing_user_id", reason=_REASONS["missing_user_id"])

    # Tier gate — AutoPilot is Elite-locked per Step 1 §5.E1.
    try:
        ut = resolve_user_tier(user_id)
    except Exception as exc:
        logger.warning("tier resolution failed for %s: %s", user_id, exc)
        return LiveTradeEligibility(False, code="tier_too_low", reason=_REASONS["tier_too_low"])

    if not meets_tier(ut.tier, require_tier):
        return LiveTradeEligibility(False, code="tier_too_low", reason=_REASONS["tier_too_low"])

    # Per-user kill switch — if the user paused live execution from
    # Settings → Kill Switch, AutoPilot must not run for them.
    try:
        prof = (
            supabase.table("user_profiles")
            .select("live_trading_paused, live_trading_paused_until")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        row = (prof.data or [{}])[0]
        if row.get("live_trading_paused"):
            return LiveTradeEligibility(False, code="user_kill_switch", reason=_REASONS["user_kill_switch"])
    except Exception as exc:
        # Schema may not have the columns yet on older deployments.
        # Treat as not-paused so we don't block legitimate users on a
        # cosmetic schema gap.
        logger.debug("user kill-switch check skipped: %s", exc)

    # Broker connection.
    try:
        conn = (
            supabase.table("broker_connections")
            .select("broker_name, status")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("broker_connections lookup failed for %s: %s", user_id, exc)
        return LiveTradeEligibility(False, code="no_broker", reason=_REASONS["no_broker"])

    if not conn.data:
        return LiveTradeEligibility(False, code="no_broker", reason=_REASONS["no_broker"])
    if conn.data[0].get("status") != "connected":
        return LiveTradeEligibility(False, code="broker_disconnected", reason=_REASONS["broker_disconnected"])

    return LiveTradeEligibility(True)


__all__ = ["LiveTradeEligibility", "check_live_trade_eligibility"]
