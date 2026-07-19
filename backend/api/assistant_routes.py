"""
Assistant usage route — the daily chat-credit meter.

The legacy finance-assistant chat brain (``POST /api/assistant/chat`` +
services/assistant DomainGuard/AssistantService) was removed in the
2026-07-11 chat unification: the Copilot graph (backend/ai/agents/copilot.py)
is the ONE brain, and this endpoint had no UI caller left. What remains is
the credit-usage read that the quota UI (CopilotQuotaModal, /copilot footer)
polls; it reports the same daily window the copilot cap enforcement consumes
(middleware/tier_gate.py → AssistantCreditLimiter → user_profiles), so it is
NOT gated behind the retired ENABLE_FINANCE_ASSISTANT flag.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..schemas import AssistantUsageResponse
from ..services.assistant import AssistantCreditLimiter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Assistant"])


def _get_user_profile_dep():
    from .app import get_user_profile
    return get_user_profile


@router.get("/api/assistant/usage", response_model=AssistantUsageResponse)
async def get_assistant_usage(profile=Depends(_get_user_profile_dep())):
    """Current user's daily chat-credit usage (the Copilot cap window)."""
    from ..core.database import get_supabase_admin

    limiter = AssistantCreditLimiter(get_supabase_admin())
    usage = limiter.get_usage(user_id=profile["id"], profile=profile)
    return {"usage": usage.to_dict()}
