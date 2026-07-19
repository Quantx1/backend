"""
Tier enum + feature matrix — source of truth for Free / Pro / Elite.

The tier itself is persisted on ``user_profiles.tier`` (added in PR 2).
This module answers two questions everyone needs to ask:

    1. Which tier does this user have?  (``resolve_user_tier``)
    2. Does this tier have access to <feature>?  (``FEATURE_MATRIX``)

FastAPI dependencies live in ``backend/middleware/tier_gate.py``
and consume everything defined here.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Tier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ELITE = "elite"


# Ordered weakest → strongest. Used for ``rank(A) >= rank(B)`` gate checks.
TIER_ORDER: Dict[Tier, int] = {Tier.FREE: 0, Tier.PRO: 1, Tier.ELITE: 2}


def tier_rank(tier: Tier | str) -> int:
    t = Tier(tier) if not isinstance(tier, Tier) else tier
    return TIER_ORDER.get(t, 0)


def meets_tier(current: Tier | str, minimum: Tier | str) -> bool:
    return tier_rank(current) >= tier_rank(minimum)


# ============================================================================
# FEATURE MATRIX — maps feature key → minimum tier
# ============================================================================
# Source of truth: Step 1 §5 (master feature list) + §A–E tier table.
# Keep this in one place so the frontend, tier-gate middleware, and
# admin UI all read the same map.
# ============================================================================

FEATURE_MATRIX: Dict[str, Tier] = {
    # ── Acquisition / public / activation (FREE) ──────────────────────────
    "landing": Tier.FREE,
    "pricing": Tier.FREE,
    "public_regime": Tier.FREE,
    "public_track_record": Tier.FREE,
    "public_models": Tier.FREE,
    "signup": Tier.FREE,
    "onboarding_quiz": Tier.FREE,
    "paper_portfolio_seed": Tier.FREE,
    "paper_trading": Tier.FREE,    # F11 — conversion funnel, stays free
    "first_paper_trade": Tier.FREE,
    "telegram_digest": Tier.FREE,    # WhatsApp is Pro

    # ── Core engagement ───────────────────────────────────────────────────
    "dashboard_basic": Tier.FREE,    # limited widgets
    "dashboard_full": Tier.PRO,
    "signal_daily": Tier.FREE,    # 1 signal/day
    "signal_unlimited": Tier.PRO,     # F2
    "intraday_signals": Tier.PRO,     # F1
    "momentum_weekly": Tier.PRO,     # F3
    "regime_size_gating": Tier.PRO,     # F8 (free sees banner only)
    "ai_dossier_basic": Tier.FREE,    # N2 stock page basic
    "ai_dossier_full": Tier.PRO,     # full model-output grid
    "scanner_lab": Tier.PRO,     # C7 — screeners + patterns
    "copilot_chat": Tier.FREE,    # 5 msgs/day (credit-metered)
    "copilot_pro": Tier.PRO,     # 150 msgs/day
    "copilot_elite": Tier.ELITE,   # unlimited
    "watchlist_basic": Tier.FREE,    # 5 symbols
    "watchlist_unlimited": Tier.PRO,
    "whatsapp_digest": Tier.PRO,     # F12
    "alert_studio": Tier.PRO,     # C12 full studio
    "finagent_vision": Tier.PRO,     # B2 on signals (Elite on-demand)
    "finagent_vision_anywhere": Tier.ELITE,

    # ── Retention / Elite expansion ───────────────────────────────────────
    "weekly_review": Tier.PRO,     # N10
    "paper_league": Tier.FREE,    # N6
    "referrals": Tier.FREE,    # N12

    # ── Elite-only flagships ──────────────────────────────────────────────
    # Pricing v2 2026-06-12 — AutoPilot Lite moves to PRO (live auto-trading
    # capped by AUTO_TRADER_TIER_LIMITS below: ≤₹2L deployed, 8 positions,
    # equity only). Elite = uncapped capital + F&O + 15 positions. Free gets
    # paper-mode AutoPilot only (virtual, Lite-shaped so the demo mirrors
    # what Pro buys).
    "auto_trader": Tier.PRO,     # F4 — AutoPilot Lite
    "auto_trader_unlimited": Tier.ELITE,   # F4 — no capital cap, F&O streams
    "fo_strategies": Tier.ELITE,   # F6
    "portfolio_doctor_free": Tier.FREE,    # one-off ₹199 product
    "portfolio_doctor_pro": Tier.PRO,     # included
    "portfolio_doctor_unlim": Tier.ELITE,   # unlimited re-runs
    "earnings_basic": Tier.PRO,     # F9 earnings calendar (Pro)
    "debate": Tier.ELITE,   # B1 Bull/Bear
    "marketplace_browse": Tier.FREE,    # B3 browse
    "marketplace_deploy": Tier.PRO,
    "marketplace_publish": Tier.ELITE,

    # ── Trust / safety (all tiers) ────────────────────────────────────────
    "kill_switch": Tier.FREE,
}


def required_tier(feature: str) -> Tier:
    """Look up the minimum tier for a feature key. Unknown keys default to Free."""
    return FEATURE_MATRIX.get(feature, Tier.FREE)


# ============================================================================
# AutoPilot tier limits — pricing v2 (2026-06-12)
# ============================================================================
# The price-discrimination axis for AutoPilot is DEPLOYED CAPITAL — exactly
# how its value scales. Free runs paper-only with Lite-shaped limits so the
# free demo honestly mirrors what Pro buys.
# ============================================================================

AUTO_TRADER_TIER_LIMITS: Dict[Tier, Dict[str, Any]] = {
    Tier.FREE: {
        "max_deployed_capital": 200_000.0,
        "max_concurrent_positions": 8,
        "allow_fno": False,
        "paper_only": True,
    },
    Tier.PRO: {
        "max_deployed_capital": 200_000.0,
        "max_concurrent_positions": 8,
        "allow_fno": False,
        "paper_only": False,
    },
    Tier.ELITE: {
        "max_deployed_capital": None,   # uncapped
        "max_concurrent_positions": 15,
        "allow_fno": True,
        "paper_only": False,
    },
}


def auto_trader_limits(tier: Tier | str) -> Dict[str, Any]:
    """Tier-resolved AutoPilot limits (always returns a dict)."""
    t = Tier(tier) if not isinstance(tier, Tier) else tier
    return dict(AUTO_TRADER_TIER_LIMITS[t])


def resolve_autopilot_mode(tier: Tier | str, config: Optional[Dict[str, Any]]) -> str:
    """'paper' or 'live' for a user's AutoPilot runs.

    Free is ALWAYS paper. Pro/Elite default to live, but a raw
    ``auto_trader_config["mode"] == "paper"`` (set by the managed-mode
    paper toggle) keeps them virtual until they explicitly go live —
    so a Free user who upgrades never silently flips to real money.
    """
    t = Tier(tier) if not isinstance(tier, Tier) else tier
    if t == Tier.FREE:
        return "paper"
    return "paper" if (config or {}).get("mode") == "paper" else "live"


# ──────────────────────────────────────────────────────────────────────────
# Per-tier LLM query caps — protect the $50/mo OpenRouter budget.
#
# Cost model: routine roles run on FREE models ($0). The $50 paid budget is
# only touched by peak-hour fallbacks when free models rate-limit, and the
# hard kill-switch (observability/llm_budget.py UsageMeter) stops ALL paid
# calls once monthly spend hits $50 — so $50 can never be exceeded. These
# caps are per-user ABUSE CEILINGS so no single account drains the shared
# free-model quota or the budget. Expected usage sits far below the caps.
#
# 0 = feature not available to that tier; value = max calls per window.
# Window is daily (UTC midnight reset) unless listed in LLM_FEATURE_CAP_WINDOW.
LLM_FEATURE_CAPS: Dict[str, Dict[Tier, int]] = {
    # Pricing v2 2026-06-12 — Pro raised 50→150 (the pricing page already
    # sold 150; backend now matches the public promise). Elite 200→400.
    # Still pure abuse ceilings: every role rides free models + the $50
    # kill-switch, so these are not cost exposure.
    "chat": {Tier.FREE: 5, Tier.PRO: 150, Tier.ELITE: 400},  # copilot/assistant msgs/day
    "strategy_gen": {Tier.FREE: 1, Tier.PRO: 10, Tier.ELITE: 30},   # Studio NL→DSL /day
    "scanner_thesis": {Tier.FREE: 0, Tier.PRO: 30, Tier.ELITE: 100},  # screener/scanner AI thesis /day
    "chart_vision": {Tier.FREE: 0, Tier.PRO: 20, Tier.ELITE: 60},   # B2 chart vision /day
    "debate": {Tier.FREE: 0, Tier.PRO: 0, Tier.ELITE: 10},   # bull/bear debate /day
    "fno_advisor": {Tier.FREE: 0, Tier.PRO: 0, Tier.ELITE: 20},   # F&O AI advisor /day
    "portfolio_doctor": {Tier.FREE: 1, Tier.PRO: 10, Tier.ELITE: 60},   # 4-agent doctor /MONTH
    "watchlist_digest": {Tier.FREE: 1, Tier.PRO: 10, Tier.ELITE: 30},  # watchlist daily-digest narrative /day
    # News Intelligence enrichment (event-type + impact). Runs on the FREE
    # fast model ($0) so these are generous abuse ceilings, not cost gates.
    "news_intel": {Tier.FREE: 10, Tier.PRO: 60, Tier.ELITE: 200},  # per day
}

# Features whose cap window is the calendar month (rest reset daily at UTC midnight).
LLM_FEATURE_CAP_WINDOW: Dict[str, str] = {
    "portfolio_doctor": "month",
}


def llm_feature_cap(feature: str, tier: Tier | str) -> int:
    """Max LLM calls for a (feature, tier) in its window. 0 = not available."""
    t = Tier(tier) if not isinstance(tier, Tier) else tier
    return LLM_FEATURE_CAPS.get(feature, {}).get(t, 0)


def llm_feature_window(feature: str) -> str:
    """'day' (default) or 'month' for a feature's cap reset window."""
    return LLM_FEATURE_CAP_WINDOW.get(feature, "day")


def feature_access_map(tier: Tier | str) -> Dict[str, bool]:
    """Return ``{feature_key: has_access}`` for the whole matrix.
    Useful for the frontend to render tier-gated UI up front."""
    t = Tier(tier) if not isinstance(tier, Tier) else tier
    return {key: tier_rank(t) >= tier_rank(v) for key, v in FEATURE_MATRIX.items()}


# ============================================================================
# USER TIER RESOLVER — one query per user per cache window
# ============================================================================


@dataclass
class UserTier:
    user_id: str
    tier: Tier
    is_admin: bool = False
    email: Optional[str] = None


_CACHE: Dict[str, tuple] = {}  # user_id → (UserTier, expires_ts)
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 60  # 1 minute — tier upgrades propagate within this


def resolve_user_tier(user_id: str, *, supabase_client=None) -> UserTier:
    """Fetch the user's tier + admin flag. In-memory 60s cache.

    On any failure defaults to ``Free`` — errs on the side of not
    leaking Elite features rather than of failing open.
    """
    # DEV-ONLY: the auth-bypass mock user resolves to full Elite + admin so local
    # testing can exercise every tier-gated agent without a seeded profile. Hard
    # gated (never honored when APP_ENV=production) and scoped to the mock id.
    from .security import _dev_auth_enabled, _DEV_USER
    if _dev_auth_enabled() and user_id == _DEV_USER.id:
        return UserTier(user_id=user_id, tier=Tier.ELITE, is_admin=True, email=_DEV_USER.email)

    now = time.time()

    with _CACHE_LOCK:
        cached = _CACHE.get(user_id)
        if cached and cached[1] > now:
            return cached[0]

    resolved = UserTier(user_id=user_id, tier=Tier.FREE, is_admin=False)
    if supabase_client is None:
        try:
            from ..core.database import get_supabase_admin
            supabase_client = get_supabase_admin()
        except Exception:
            supabase_client = None

    if supabase_client is not None:
        try:
            result = (
                supabase_client.table("user_profiles")
                .select("tier, is_admin, email")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
            rows = result.data or []
            if rows:
                row = rows[0]
                tier_str = str(row.get("tier") or "free").lower()
                try:
                    resolved.tier = Tier(tier_str)
                except ValueError:
                    resolved.tier = Tier.FREE
                resolved.is_admin = bool(row.get("is_admin", False))
                resolved.email = row.get("email")
        except Exception as exc:
            logger.debug("resolve_user_tier(%s) failed: %s", user_id, exc)

    with _CACHE_LOCK:
        _CACHE[user_id] = (resolved, now + _CACHE_TTL_SECONDS)
    return resolved


def invalidate_user_tier_cache(user_id: Optional[str] = None) -> None:
    """Drop cached tier for one user (on tier change webhook) or all users."""
    with _CACHE_LOCK:
        if user_id is None:
            _CACHE.clear()
        else:
            _CACHE.pop(user_id, None)


__all__ = [
    "FEATURE_MATRIX",
    "LLM_FEATURE_CAPS",
    "LLM_FEATURE_CAP_WINDOW",
    "llm_feature_cap",
    "llm_feature_window",
    "Tier",
    "UserTier",
    "feature_access_map",
    "invalidate_user_tier_cache",
    "meets_tier",
    "required_tier",
    "resolve_user_tier",
    "tier_rank",
]
