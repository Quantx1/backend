"""
Tier-gate FastAPI dependencies.

Two primary surfaces:

    1. ``RequireTier(Tier.ELITE)`` — gate a route by minimum tier.
    2. ``RequireFeature("auto_trader")`` — gate by feature key that maps
       through ``FEATURE_MATRIX`` in ``core/tiers.py``.

Both return a ``UserTier`` so the route handler has tier context and
doesn't need to re-query.

Admin bypass: ``is_admin`` always passes the gate.

On failure we raise ``HTTPException(status=402, detail={...})`` with a
structured payload the frontend can read:

    {
        "error": "tier_gate",
        "required_tier": "elite",
        "current_tier": "free",
        "feature": "debate",         # when RequireFeature was used
        "upgrade_url": "/pricing",
    }

Status 402 = "Payment Required" per RFC 7231 — semantically accurate
for paid-feature gates. The frontend `handleApiError` treats 402
specially: show an `UpgradeModal` instead of a generic error toast.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException, status

from ..core.security import get_current_user
from ..core.tiers import (
    LLM_FEATURE_CAPS,
    Tier,
    UserTier,
    feature_access_map,
    meets_tier,
    required_tier,
    resolve_user_tier,
)

logger = logging.getLogger(__name__)


def _gate_payload(
    *,
    required: Tier,
    current: Tier,
    feature: Optional[str] = None,
) -> dict:
    payload = {
        "error": "tier_gate",
        "required_tier": required.value,
        "current_tier": current.value,
        "upgrade_url": "/pricing",
    }
    if feature:
        payload["feature"] = feature
    return payload


# ---------------------------------------------------------------- dependencies


async def current_user_tier(user: Any = Depends(get_current_user)) -> UserTier:
    """Any authenticated user → their :class:`UserTier`, with NO feature/tier gate.

    Use where a surface is available to ALL tiers but the handler still needs
    tier context to branch inside (e.g. the paper AutoPilot bot is Free; only
    *going live* checks the tier — see ``can_use_feature``)."""
    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return resolve_user_tier(str(user_id))


def can_use_feature(user_tier: Tier, feature: str) -> bool:
    """True if ``user_tier`` meets the minimum tier required for ``feature``."""
    return meets_tier(user_tier, required_tier(feature))


class RequireTier:
    """Dependency: raise 402 if ``user.tier`` < ``min_tier``.

    Usage::

        @router.post("/ai/debate/signal/{id}")
        async def run_debate(
            signal_id: str,
            user: UserTier = Depends(RequireTier(Tier.ELITE)),
        ):
            ...
    """

    def __init__(self, min_tier: Tier):
        if not isinstance(min_tier, Tier):
            min_tier = Tier(min_tier)
        self.min_tier = min_tier

    async def __call__(self, user: Any = Depends(get_current_user)) -> UserTier:
        user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
        if not user_id:
            raise HTTPException(status_code=401, detail="unauthenticated")

        ut = resolve_user_tier(str(user_id))

        # Admins always pass.
        if ut.is_admin:
            return ut

        if not meets_tier(ut.tier, self.min_tier):
            # PR 16 — product analytics: track gate-hit for conversion funnel.
            try:
                from ..observability import EventName, track
                track(EventName.TIER_GATE_HIT, ut.user_id, {
                    "current_tier": ut.tier.value,
                    "required_tier": self.min_tier.value,
                    "gate_type": "tier",
                })
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=_gate_payload(required=self.min_tier, current=ut.tier),
            )
        return ut


class RequireFeature:
    """Dependency: raise 402 if ``user.tier`` lacks ``feature``.

    Prefer this over ``RequireTier`` when the gated thing corresponds to
    a feature key in ``FEATURE_MATRIX`` — keeps the source of truth in
    one place.

    Usage::

        @router.post("/ai/fo-strategies/preview")
        async def preview(user: UserTier = Depends(RequireFeature("fo_strategies"))):
            ...
    """

    def __init__(self, feature: str):
        self.feature = feature
        self.min_tier = required_tier(feature)

    async def __call__(self, user: Any = Depends(get_current_user)) -> UserTier:
        user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
        if not user_id:
            raise HTTPException(status_code=401, detail="unauthenticated")

        ut = resolve_user_tier(str(user_id))
        if ut.is_admin:
            return ut
        if not meets_tier(ut.tier, self.min_tier):
            try:
                from ..observability import EventName, track
                track(EventName.TIER_GATE_HIT, ut.user_id, {
                    "current_tier": ut.tier.value,
                    "required_tier": self.min_tier.value,
                    "feature": self.feature,
                    "gate_type": "feature",
                })
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=_gate_payload(
                    required=self.min_tier,
                    current=ut.tier,
                    feature=self.feature,
                ),
            )
        return ut


async def current_user_tier(user: Any = Depends(get_current_user)) -> UserTier:
    """Dependency that just resolves the current tier (no gating).
    Useful in routes that render tier-dependent content but don't want
    to reject outright."""
    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return resolve_user_tier(str(user_id))


# ---------------------------------------------------- copilot credit limits

# Daily chat-message caps per tier — DERIVED from the canonical
# core.tiers.LLM_FEATURE_CAPS["chat"] (5 free / 150 pro / 400 elite, pricing
# v2). This alias used to hold its own literal values and drifted to 50/200;
# deriving kills that class of bug. The $20 kill-switch remains the hard spend
# cap, and routine chat runs on free models so it costs $0.
COPILOT_DAILY_CAPS: dict = dict(LLM_FEATURE_CAPS["chat"])


def copilot_daily_cap(tier: Tier | str) -> int:
    t = Tier(tier) if not isinstance(tier, Tier) else tier
    return COPILOT_DAILY_CAPS.get(t, COPILOT_DAILY_CAPS[Tier.FREE])


__all__ = [
    "COPILOT_DAILY_CAPS",
    "RequireFeature",
    "RequireTier",
    "copilot_daily_cap",
    "current_user_tier",
    "feature_access_map",
]
