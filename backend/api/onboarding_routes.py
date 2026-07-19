"""
================================================================================
ONBOARDING ROUTES — N5 risk-profile quiz (PR 37)
================================================================================
First-login wizard. 5 questions → weighted score → risk profile +
recommended tier + signal-filter preset + auto-trader config defaults.

Endpoints:
    GET  /api/onboarding/status   — is the user onboarded? quiz payload?
    POST /api/onboarding/quiz     — submit answers, persist, return result
    POST /api/onboarding/skip     — mark complete without answering (uses defaults)

Quiz axes:
    1. experience   — how many years trading
    2. risk_tol     — tolerance for a 15% drawdown month
    3. horizon      — primary holding window
    4. loss_cap     — acceptable max-loss per trade
    5. goal         — primary goal (preservation / income / growth / aggressive_growth)
================================================================================
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from ..core.database import get_supabase_admin
from ..core.security import get_current_user
from ..core.tiers import invalidate_user_tier_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


# ============================================================================
# Quiz definition — single source of truth (frontend reads this shape)
# ============================================================================


QUIZ: List[Dict[str, Any]] = [
    {
        "key": "experience",
        "question": "How long have you actively traded Indian equities?",
        "options": [
            {"value": "new", "label": "First year — still learning", "score": 0},
            {"value": "1_3_years", "label": "1–3 years", "score": 1},
            {"value": "3_7_years", "label": "3–7 years", "score": 2},
            {"value": "7_plus", "label": "7+ years", "score": 3},
        ],
    },
    {
        "key": "risk_tol",
        "question": "If your portfolio dropped 15% in a single month, you would:",
        "options": [
            {"value": "sell_all", "label": "Sell everything — stop the pain", "score": 0},
            {"value": "sell_some", "label": "Trim positions, raise cash", "score": 1},
            {"value": "hold", "label": "Hold and wait it out", "score": 2},
            {"value": "buy_more", "label": "Buy more at lower prices", "score": 3},
        ],
    },
    {
        "key": "horizon",
        "question": "How long do you typically hold a winning trade?",
        "options": [
            {"value": "intraday", "label": "Intraday — square off same day", "score": 3},
            {"value": "swing", "label": "3–10 days (swing)", "score": 2},
            {"value": "positional", "label": "1–3 months", "score": 1},
            {"value": "long_term", "label": "1+ year — build a portfolio", "score": 0},
        ],
    },
    {
        "key": "loss_cap",
        "question": "Largest loss you will accept on a single trade:",
        "options": [
            {"value": "lte_1", "label": "≤ 1% of capital", "score": 0},
            {"value": "1_to_3", "label": "1–3%", "score": 1},
            {"value": "3_to_5", "label": "3–5%", "score": 2},
            {"value": "gt_5", "label": "> 5% — I swing for the fences", "score": 3},
        ],
    },
    {
        "key": "goal",
        "question": "Primary goal for trading with us:",
        "options": [
            {"value": "preservation", "label": "Preserve capital, beat FDs", "score": 0},
            {"value": "income", "label": "Generate steady income", "score": 1},
            {"value": "growth", "label": "Grow capital over 3–5 years", "score": 2},
            {"value": "aggressive", "label": "Aggressive compounding", "score": 3},
        ],
    },
]

VALID_KEYS = {q["key"] for q in QUIZ}
VALID_VALUES: Dict[str, set] = {
    q["key"]: {opt["value"] for opt in q["options"]}
    for q in QUIZ
}


# ============================================================================
# Pydantic
# ============================================================================


class QuizAnswers(BaseModel):
    answers: Dict[str, str] = Field(..., description="{key: value}")

    @validator("answers")
    def _validate(cls, v: Dict[str, str]) -> Dict[str, str]:
        missing = VALID_KEYS - set(v.keys())
        if missing:
            raise ValueError(f"missing answers for: {sorted(missing)}")
        for k, val in v.items():
            if k not in VALID_VALUES:
                raise ValueError(f"unknown question key: {k}")
            if val not in VALID_VALUES[k]:
                raise ValueError(f"invalid value for {k}: {val}")
        return v


class QuizResult(BaseModel):
    risk_profile: str         # conservative | moderate | aggressive
    recommended_tier: str     # free | pro | elite
    score: int                # raw 0..15
    rationale: str            # short one-line explanation
    suggested_filters: Dict[str, Any]   # signal list preset
    auto_trader_defaults: Dict[str, Any]


class OnboardingStatus(BaseModel):
    completed: bool
    completed_at: Optional[str]
    current_tier: str
    current_risk_profile: Optional[str]
    recommended_tier: Optional[str]


# ============================================================================
# Scoring
# ============================================================================


def _score(answers: Dict[str, str]) -> int:
    total = 0
    for q in QUIZ:
        ans = answers.get(q["key"])
        opt = next((o for o in q["options"] if o["value"] == ans), None)
        if opt:
            total += int(opt["score"])
    return total


def _derive_profile(score: int) -> str:
    # Range 0..15 (5 questions × 0..3).
    if score <= 4:
        return "conservative"
    if score <= 10:
        return "moderate"
    return "aggressive"


def _recommended_tier(profile: str, answers: Dict[str, str]) -> str:
    # Intraday traders → Pro (they need F1). Aggressive goal + intraday +
    # 7+ years → Elite. Long-term preservation → Free.
    horizon = answers.get("horizon")
    goal = answers.get("goal")
    experience = answers.get("experience")

    if profile == "aggressive" and goal == "aggressive" and experience in {"3_7_years", "7_plus"}:
        return "elite"
    if profile == "conservative" and goal == "preservation":
        return "free"
    if horizon == "intraday" or profile == "aggressive":
        return "pro"
    if profile == "moderate":
        return "pro"
    return "free"


def _suggested_filters(profile: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """Default signal list presets the frontend can apply on first load."""
    horizon = answers.get("horizon")
    filters: Dict[str, Any] = {
        "segment": "EQUITY",
        "min_confidence": 60 if profile == "aggressive" else 70 if profile == "moderate" else 80,
        "include_intraday": horizon == "intraday" or profile == "aggressive",
        "include_fno": profile == "aggressive" and answers.get("experience") in {"3_7_years", "7_plus"},
    }
    return filters


def _auto_trader_defaults(profile: str) -> Dict[str, Any]:
    mapping = {
        "conservative": {
            "risk_profile": "conservative",
            "max_position_pct": 5.0,
            "daily_loss_limit_pct": 1.5,
            "max_concurrent_positions": 8,
            "allow_fno": False,
        },
        "moderate": {
            "risk_profile": "moderate",
            "max_position_pct": 7.0,
            "daily_loss_limit_pct": 2.0,
            "max_concurrent_positions": 12,
            "allow_fno": False,
        },
        "aggressive": {
            "risk_profile": "aggressive",
            "max_position_pct": 10.0,
            "daily_loss_limit_pct": 3.0,
            "max_concurrent_positions": 15,
            "allow_fno": True,
        },
    }
    return mapping[profile]


def _rationale(profile: str, recommended_tier: str, score: int) -> str:
    profile_copy = {
        "conservative": "Capital-preservation first — tight SLs, larger cash buffer, long horizons",
        "moderate": "Balanced — full signal access with regime-aware sizing",
        "aggressive": "High-conviction, concentrated bets — shorter horizons welcome",
    }
    tier_copy = {
        "free": "Free tier is enough to start — watch AutoPilot trade virtual money, upgrade when ready.",
        "pro": "Pro tier matches — unlimited signals + AutoPilot Lite (live auto-trading up to ₹2L) + Portfolio Doctor.",
        "elite": "Elite matches — AutoPilot without capital caps, F&O strategies, Counterpoint debate, unlimited everything.",
    }
    return f"{profile_copy[profile]} (score {score}/15). {tier_copy[recommended_tier]}"


# ============================================================================
# Routes
# ============================================================================


def _user_id_from(user: Any) -> str:
    uid = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    if not uid:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return str(uid)


@router.get("/quiz")
async def get_quiz() -> Dict[str, Any]:
    """Public quiz shape — no auth. Frontend renders from here so the
    questions + options stay in one place."""
    return {"quiz": QUIZ}


@router.get("/status", response_model=OnboardingStatus)
async def get_status(user: Any = Depends(get_current_user)) -> OnboardingStatus:
    uid = _user_id_from(user)
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_profiles")
            .select(
                "tier, risk_profile, recommended_tier, "
                "onboarding_completed, onboarding_completed_at"
            )
            .eq("id", uid)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("onboarding status lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="lookup_failed")
    row = (rows.data or [None])[0]
    if row is None:
        raise HTTPException(status_code=404, detail="profile_not_found")
    return OnboardingStatus(
        completed=bool(row.get("onboarding_completed", False)),
        completed_at=str(row["onboarding_completed_at"]) if row.get("onboarding_completed_at") else None,
        current_tier=str(row.get("tier") or "free"),
        current_risk_profile=row.get("risk_profile"),
        recommended_tier=row.get("recommended_tier"),
    )


@router.post("/quiz", response_model=QuizResult)
async def submit_quiz(
    body: QuizAnswers,
    user: Any = Depends(get_current_user),
) -> QuizResult:
    uid = _user_id_from(user)
    score = _score(body.answers)
    profile = _derive_profile(score)
    tier = _recommended_tier(profile, body.answers)
    filters = _suggested_filters(profile, body.answers)
    auto_cfg = _auto_trader_defaults(profile)
    rationale = _rationale(profile, tier, score)

    sb = get_supabase_admin()
    try:
        sb.table("user_profiles").update({
            "risk_profile": profile,
            "recommended_tier": tier,
            "risk_quiz_answers": body.answers,
            "onboarding_completed": True,
            "onboarding_completed_at": datetime.utcnow().isoformat(),
            "auto_trader_config": auto_cfg,
            # PR 79 — persist the derived signal filter preset so
            # downstream surfaces (signals list, screener, watchlist
            # alerts) can default to the onboarded user's risk profile
            # instead of falling back to global page defaults.
            "signal_filter_defaults": filters,
        }).eq("id", uid).execute()
    except Exception as exc:
        logger.error("onboarding persist failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")

    invalidate_user_tier_cache(uid)

    try:
        from ..observability import EventName, track
        track(EventName.ONBOARDING_QUIZ_COMPLETED, uid, {
            "risk_profile": profile,
            "recommended_tier": tier,
            "score": score,
            "horizon": body.answers.get("horizon"),
            "goal": body.answers.get("goal"),
        })
    except Exception:
        pass

    return QuizResult(
        risk_profile=profile,
        recommended_tier=tier,
        score=score,
        rationale=rationale,
        suggested_filters=filters,
        auto_trader_defaults=auto_cfg,
    )


@router.post("/skip")
async def skip_quiz(user: Any = Depends(get_current_user)) -> Dict[str, Any]:
    """Mark onboarding complete without answering. Defaults stay
    ('moderate' risk_profile, no recommended tier)."""
    uid = _user_id_from(user)
    sb = get_supabase_admin()
    try:
        sb.table("user_profiles").update({
            "onboarding_completed": True,
            "onboarding_completed_at": datetime.utcnow().isoformat(),
        }).eq("id", uid).execute()
    except Exception as exc:
        logger.error("onboarding skip failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")
    return {"completed": True, "skipped": True}


__all__ = ["router"]
