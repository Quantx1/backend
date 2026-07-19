"""
Assistant daily credit limiter with database-backed persistence.

Uses an in-memory cache for speed with periodic Supabase sync so credits
survive server restarts. Falls back to in-memory-only when the DB is
unavailable (dev / tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CreditUsage:
    tier: str
    credits_limit: int
    credits_used: int
    credits_remaining: int
    reset_at: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "tier": self.tier,
            "credits_limit": self.credits_limit,
            "credits_used": self.credits_used,
            "credits_remaining": self.credits_remaining,
            "reset_at": self.reset_at,
        }


class AssistantCreditLimiter:
    """
    Daily assistant credits by user id.

    - In-memory dict for fast lookups (hot path).
    - Syncs to ``user_profiles.assistant_credits_used`` / ``assistant_credits_date``
      columns when a Supabase client is provided so usage persists across
      deploys and restarts.
    - UTC day boundary for reset.
    """

    def __init__(self, supabase_client=None):
        self._usage: Dict[str, Tuple[str, int]] = {}
        self._lock = Lock()
        self._supabase = supabase_client

    def set_supabase(self, client):
        """Late-bind the Supabase client (useful when created after init)."""
        self._supabase = client

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_today_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _next_reset_iso() -> str:
        now = datetime.now(timezone.utc)
        next_day = (now + timedelta(days=1)).date()
        reset_at = datetime.combine(next_day, datetime.min.time(), tzinfo=timezone.utc)
        return reset_at.isoformat()

    @staticmethod
    def _resolve_tier(profile: Dict) -> str:
        """Tier from the canonical ``user_profiles.tier`` column (PR 2).
        Falls back to subscription_status / plan_name for pre-PR-2 rows."""
        direct = str(profile.get("tier") or "").strip().lower()
        if direct in {"free", "pro", "elite"}:
            return direct
        subscription_status = str(profile.get("subscription_status", "free")).lower()
        plan = profile.get("subscription_plans") or {}
        plan_name = str(plan.get("name", "")).lower()
        if plan_name == "elite":
            return "elite"
        if subscription_status in {"active", "trial"} or plan_name in {"starter", "pro"}:
            return "pro"
        return "free"

    @staticmethod
    def _limit_for_tier(tier: str) -> int:
        """Delegate to the canonical ``copilot_daily_cap`` so Free / Pro /
        Elite caps stay aligned with what the tier panel advertises and
        the /api/user/tier endpoint reports (PR 65 — was previously
        out-of-sync at 5/100/100 via legacy ``ASSISTANT_DAILY_CREDITS_*``
        env vars that didn't know about Elite)."""
        from ...core.tiers import Tier
        from ...middleware.tier_gate import copilot_daily_cap
        try:
            return max(copilot_daily_cap(Tier(tier)), 1)
        except (ValueError, KeyError):
            return max(copilot_daily_cap(Tier.FREE), 1)

    # ------------------------------------------------------------------
    # DB sync (best-effort, never blocks the request on failure)
    # ------------------------------------------------------------------

    def _load_from_db(self, user_id: str, today_key: str) -> Optional[int]:
        """Try to load today's usage from the database."""
        if not self._supabase:
            return None
        try:
            result = self._supabase.table("user_profiles").select(
                "assistant_credits_used, assistant_credits_date"
            ).eq("id", user_id).single().execute()
            data = result.data
            if data and str(data.get("assistant_credits_date", "")) == today_key:
                return int(data.get("assistant_credits_used", 0))
        except Exception as e:
            logger.debug(f"Credit DB load failed for {user_id}: {e}")
        return None

    def _save_to_db(self, user_id: str, today_key: str, used: int):
        """Best-effort persist current usage to the database."""
        if not self._supabase:
            return
        try:
            self._supabase.table("user_profiles").update({
                "assistant_credits_used": used,
                "assistant_credits_date": today_key,
            }).eq("id", user_id).execute()
        except Exception as e:
            logger.debug(f"Credit DB save failed for {user_id}: {e}")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _current_used(self, user_id: str, today_key: str) -> int:
        row = self._usage.get(user_id)
        if row:
            day_key, used = row
            if day_key == today_key:
                return used

        # Cache miss — try loading from DB (cold start / after restart)
        db_used = self._load_from_db(user_id, today_key)
        if db_used is not None:
            self._usage[user_id] = (today_key, db_used)
            return db_used
        return 0

    def get_usage(self, user_id: str, profile: Dict) -> CreditUsage:
        today_key = self._utc_today_key()
        tier = self._resolve_tier(profile)
        limit = self._limit_for_tier(tier)
        with self._lock:
            used = self._current_used(user_id, today_key)
            remaining = max(limit - used, 0)
            return CreditUsage(
                tier=tier,
                credits_limit=limit,
                credits_used=used,
                credits_remaining=remaining,
                reset_at=self._next_reset_iso(),
            )

    def consume_if_available(self, user_id: str, profile: Dict, cost: int = 1) -> Tuple[bool, CreditUsage]:
        cost = max(cost, 1)
        today_key = self._utc_today_key()
        tier = self._resolve_tier(profile)
        limit = self._limit_for_tier(tier)

        with self._lock:
            used = self._current_used(user_id, today_key)
            if used + cost > limit:
                usage = CreditUsage(
                    tier=tier,
                    credits_limit=limit,
                    credits_used=used,
                    credits_remaining=max(limit - used, 0),
                    reset_at=self._next_reset_iso(),
                )
                return False, usage

            new_used = used + cost
            self._usage[user_id] = (today_key, new_used)

        # Persist outside the lock to avoid blocking
        self._save_to_db(user_id, today_key, new_used)

        usage = CreditUsage(
            tier=tier,
            credits_limit=limit,
            credits_used=new_used,
            credits_remaining=max(limit - new_used, 0),
            reset_at=self._next_reset_iso(),
        )
        return True, usage
