"""Per-feature LLM call caps — the generic version of the chat-only
AssistantCreditLimiter. Reads cap + window from ``core.tiers``
(LLM_FEATURE_CAPS / LLM_FEATURE_CAP_WINDOW), keeps a hot in-process counter,
and best-effort persists to ``llm_feature_usage`` so caps survive restarts.

Fail-open on metering errors (never block a paying user on a glitch); the $50
budget kill-switch in ``observability/llm_budget`` is the real spend backstop.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException

from ..core.tiers import Tier, llm_feature_cap, llm_feature_window
from .tier_gate import UserTier, current_user_tier

logger = logging.getLogger(__name__)


class LlmFeatureLimiter:
    def __init__(self, supabase_client: Any = None) -> None:
        self._usage: Dict[Tuple[str, str, str], int] = {}
        self._lock = Lock()
        self._sb = supabase_client

    @staticmethod
    def _window_key(feature: str) -> str:
        now = datetime.now(timezone.utc)
        if llm_feature_window(feature) == "month":
            return now.strftime("%Y-%m")
        return now.strftime("%Y-%m-%d")

    def _load(self, user_id: str, feature: str, wkey: str) -> Optional[int]:
        if not self._sb:
            return None
        try:
            r = (
                self._sb.table("llm_feature_usage")
                .select("used")
                .eq("user_id", user_id).eq("feature", feature).eq("window_key", wkey)
                .limit(1).execute()
            )
            if r.data:
                return int(r.data[0].get("used") or 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("llm_feature_usage load failed: %s", exc)
        return None

    def _save(self, user_id: str, feature: str, wkey: str, used: int) -> None:
        if not self._sb:
            return
        try:
            self._sb.table("llm_feature_usage").upsert({
                "user_id": user_id, "feature": feature, "window_key": wkey,
                "used": used, "updated_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="user_id,feature,window_key").execute()
        except Exception as exc:  # noqa: BLE001
            logger.debug("llm_feature_usage save failed: %s", exc)

    def _current(self, user_id: str, feature: str, wkey: str) -> int:
        k = (user_id, feature, wkey)
        if k in self._usage:
            return self._usage[k]
        db = self._load(user_id, feature, wkey)
        if db is not None:
            self._usage[k] = db
            return db
        return 0

    def consume(self, user_id: str, feature: str, tier: Tier | str, cost: int = 1) -> Tuple[bool, int, int]:
        """Try to consume `cost` from (user, feature) in its window. Returns
        (allowed, used_after, cap). cap<=0 → feature unavailable to that tier."""
        cost = max(cost, 1)
        cap = llm_feature_cap(feature, tier)
        wkey = self._window_key(feature)
        with self._lock:
            used = self._current(user_id, feature, wkey)
            if cap <= 0 or used + cost > cap:
                return False, used, cap
            used += cost
            self._usage[(user_id, feature, wkey)] = used
        self._save(user_id, feature, wkey, used)
        return True, used, cap


_limiter: Optional[LlmFeatureLimiter] = None


def get_llm_feature_limiter() -> LlmFeatureLimiter:
    global _limiter
    if _limiter is None:
        try:
            from ..api.app import get_supabase_admin
            _limiter = LlmFeatureLimiter(get_supabase_admin())
        except Exception:  # noqa: BLE001
            _limiter = LlmFeatureLimiter(None)
    return _limiter


def enforce_llm_cap(feature: str):
    """FastAPI dependency factory: consume one per-feature LLM credit for the
    current user; raise HTTP 402 when over the per-tier cap. Admins bypass.
    Fail-open on limiter errors (the $50 budget kill-switch still applies)."""
    async def _dep(user: UserTier = Depends(current_user_tier)) -> UserTier:
        consume_llm_cap_or_raise(user, feature)
        return user
    return _dep


def consume_llm_cap_or_raise(user: UserTier, feature: str) -> None:
    """Imperative counterpart to enforce_llm_cap: consume one credit for `user`
    on `feature`, raising HTTP 402 when over the tier cap. Admins bypass;
    fail-open on limiter errors. Use inside a handler when the credit must be
    spent conditionally (e.g. only after a 0-token pre-gate decides to call the
    generator)."""
    if getattr(user, "is_admin", False):
        return
    try:
        allowed, _used, cap = get_llm_feature_limiter().consume(
            user.user_id, feature, user.tier)
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm cap consume skipped (%s) — proceeding", exc)
        return
    if not allowed:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "credit_cap",
                "feature": feature,
                "current_tier": user.tier.value,
                "limit": cap,
                "window": llm_feature_window(feature),
                "upgrade_url": "/pricing",
            },
        )
