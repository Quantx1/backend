"""
================================================================================
PORTFOLIO DOCTOR ROUTES — F7 InsightAI portfolio-level analysis (PR 34)
================================================================================
``/portfolio/doctor`` — 4-agent Chain-of-Thought analysis over the user's
whole portfolio:

    Fundamental  — balance sheet / earnings trajectory per holding
    Management   — concall tone, guidance track record
    Promoter     — shareholding changes, pledge flags
    Peer         — relative valuation / growth vs sector median

Outputs are per-position scores + composite portfolio score + risk
flags (concentration, sector skew, stale stops, drawdown streak) +
one-line action.

Tier split:
    ``portfolio_doctor_free``  — Free — one-off, payment-gated at /pricing
    ``portfolio_doctor_pro``   — Pro  — one run / month included
    ``portfolio_doctor_unlim`` — Elite — unlimited reruns

Persistence: ``portfolio_doctor_reports`` (PR 34 migration) stores every
run so users can revisit prior assessments.
================================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.database import get_supabase_admin
from ..core.public_models import public_label
from ..core.tiers import Tier, UserTier, tier_rank
from ..middleware.llm_caps import enforce_llm_cap
from ..middleware.tier_gate import RequireFeature
from ..data.market_calendar import IST
from ..ai.agents.response_cache import cache_get, cache_set, seconds_to_ist_eod

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio/doctor", tags=["portfolio-doctor"])

MAX_PORTFOLIO_SIZE = 30
PRO_MONTHLY_QUOTA = 1  # Pro gets 1 full doctor run per calendar month


def _portfolio_hash(positions) -> str:
    norm = sorted([(p.symbol.upper(), round(p.weight, 3)) for p in positions])
    return hashlib.sha256(_json.dumps(norm, separators=(",", ":")).encode()).hexdigest()[:16]


def _rebal_cache_key(positions) -> str:
    """Per-(portfolio, IST-day) cache key for the rebalance narrative."""
    return f"rebal:{_portfolio_hash(positions)}:{datetime.now(IST).date().isoformat()}"


# ============================================================================
# Models
# ============================================================================


class PositionInput(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    weight: float = Field(..., ge=0.0, le=1.0)     # 0..1 portfolio weight
    qty: Optional[int] = None
    entry_price: Optional[float] = None
    current_price: Optional[float] = None


class AnalyzeRequest(BaseModel):
    source: str = Field("manual", pattern="^(manual|broker|csv)$")
    capital: Optional[float] = Field(None, ge=0)
    positions: List[PositionInput] = Field(..., min_items=1, max_items=MAX_PORTFOLIO_SIZE)


class PerPositionResult(BaseModel):
    symbol: str
    weight: float
    composite_score: int
    action: str
    narrative: str


class RiskFlag(BaseModel):
    kind: str       # 'concentration' | 'sector_skew' | 'drawdown' | 'stale_stop'
    severity: str   # 'low' | 'medium' | 'high'
    message: str
    meta: Dict[str, Any] = {}


class DoctorReport(BaseModel):
    id: str
    created_at: str
    source: str
    position_count: int
    capital: Optional[float]
    composite_score: int
    diversification_score: int = 0    # 0-100, higher = better diversified (1/HHI)
    risk_score: int = 0               # 0-100, higher = lower risk (health framing)
    action: str                       # rebalance | hold | reduce_risk | increase_risk
    narrative: str
    per_position: List[PerPositionResult]
    risk_flags: List[RiskFlag]
    agents: Dict[str, Any]            # raw agent output keyed by role
    quota: Dict[str, Any]             # {tier, runs_this_month, quota, remaining}


class ReportRow(BaseModel):
    id: str
    created_at: str
    source: str
    position_count: int
    composite_score: int
    action: str


# ============================================================================
# Quota helpers
# ============================================================================


def _month_start_utc() -> datetime:
    now = datetime.now(IST).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.astimezone(timezone.utc)


def _current_month_runs(user_id: str) -> int:
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("portfolio_doctor_reports")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .gte("created_at", _month_start_utc().isoformat())
            .execute()
        )
        return int(getattr(rows, "count", 0) or 0)
    except Exception as exc:
        logger.debug("doctor month-run count failed: %s", exc)
        return 0


def _quota_for(tier: Tier) -> Optional[int]:
    """Return monthly quota. None = unlimited."""
    if tier_rank(tier) >= tier_rank(Tier.ELITE):
        return None
    if tier_rank(tier) >= tier_rank(Tier.PRO):
        return PRO_MONTHLY_QUOTA
    # Free — one-off paid product; the checkout flow grants a single
    # run via an admin-issued report-token (out of scope here). For
    # now, block Free at the feature gate.
    return 0


def _assert_quota(user: UserTier) -> Dict[str, Any]:
    if user.is_admin:
        return {"tier": user.tier.value, "runs_this_month": 0, "quota": None, "remaining": None}
    quota = _quota_for(user.tier)
    runs = _current_month_runs(user.user_id)
    if quota is None:
        return {"tier": user.tier.value, "runs_this_month": runs, "quota": None, "remaining": None}
    if runs >= quota:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "doctor_quota_exhausted",
                "tier": user.tier.value,
                "runs_this_month": runs,
                "quota": quota,
                "upgrade_url": "/pricing",
            },
        )
    return {"tier": user.tier.value, "runs_this_month": runs, "quota": quota, "remaining": quota - runs}


# ============================================================================
# Analysis
# ============================================================================


def _concentration_flags(positions: List[PositionInput]) -> List[RiskFlag]:
    flags: List[RiskFlag] = []
    for p in positions:
        if p.weight >= 0.20:
            flags.append(
                RiskFlag(
                    kind="concentration",
                    severity="high" if p.weight >= 0.30 else "medium",
                    message=(
                        f"{p.symbol.upper()} is {round(p.weight * 100, 1)}% "
                        "of portfolio — above 20% concentration threshold"
                    ),
                    meta={
                        "symbol": p.symbol.upper(),
                        "weight": round(p.weight, 4)},
                ))
    top_three = sum(
        sorted((p.weight for p in positions), reverse=True)[:3]
    )
    if top_three >= 0.60 and len(positions) > 3:
        flags.append(RiskFlag(
            kind="concentration",
            severity="medium",
            message=f"Top 3 holdings account for {round(top_three * 100, 1)}% of portfolio",
            meta={"top_three_weight": round(top_three, 4)},
        ))
    return flags


def _sector_flags(positions: List[PositionInput]) -> List[RiskFlag]:
    try:
        from ..ai.sector_taxonomy import sector_for_symbol
    except Exception:
        return []
    by_sector: Dict[str, float] = {}
    for p in positions:
        sec = sector_for_symbol(p.symbol) or "Uncategorized"
        by_sector[sec] = by_sector.get(sec, 0.0) + p.weight
    flags: List[RiskFlag] = []
    for sec, w in by_sector.items():
        if w >= 0.40 and sec != "Uncategorized":
            flags.append(RiskFlag(
                kind="sector_skew",
                severity="high" if w >= 0.55 else "medium",
                message=f"{sec} is {round(w * 100, 1)}% of portfolio — sector concentration risk",
                meta={"sector": sec, "weight": round(w, 4)},
            ))
    return flags


def _regime_risk_flag() -> List[RiskFlag]:
    """Flag when current regime is bear — reduce_risk bias."""
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("regime_history")
            .select("regime, prob_bear, vix")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
        )
        if rows.data:
            r = rows.data[0]
            if r.get("regime") == "bear":
                return [
                    RiskFlag(
                        kind="drawdown", severity="high",
                        message=(
                            f"{public_label('regime_detector')} currently "
                            "reads bear — consider reducing equity exposure"
                        ),
                        meta={
                            "prob_bear": r.get("prob_bear"),
                            "vix": r.get("vix")}, )]
    except Exception:
        pass
    return []


async def _run_per_position(
    positions: List[PositionInput],
    user_id: str,
    *,
    deep: bool = False,
    tier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run InsightAI CoT for each position with bounded parallelism.
    Returns list of raw agent outputs indexed by position order."""
    try:
        from ..ai.agents.finrobot import run_finrobot_doctor
    except Exception as exc:
        logger.error("finrobot import failed: %s", exc)
        return [{} for _ in positions]

    sem = asyncio.Semaphore(4)  # 4-way parallel — OpenRouter gateway rate-limit safe

    async def one(p: PositionInput) -> Dict[str, Any]:
        sym = p.symbol.upper()
        # Per-symbol CoT depends only on the stock (fundamentals + technicals),
        # not the user's position size — so cache cross-user by symbol+depth for
        # the IST trading day. A 30-stock portfolio (and repeat holdings across
        # users) then mostly hits cache instead of 120 cold LLM calls (~58s -> fast).
        ck = f"doctor:pos:{sym}:{int(deep)}:{tier or 'na'}"
        cached = cache_get(ck)
        if cached:
            return cached
        async with sem:
            try:
                # Inject real fundamentals (screener.in) so the CoT agents grade
                # actual ROE/growth/promoter-holding instead of empty JSON.
                # to_thread: the scrape is blocking; honest-empty if unavailable.
                doctor_inputs: Dict[str, Any] = {}
                try:
                    from ..data.fundamentals.screener_in import to_doctor_inputs
                    doctor_inputs = await asyncio.to_thread(
                        to_doctor_inputs, sym
                    )
                except Exception as fexc:
                    logger.debug("fundamentals fetch failed %s: %s", p.symbol, fexc)
                res = await run_finrobot_doctor(
                    user_id=user_id,
                    symbol=sym,
                    deep=deep,
                    tier=tier,
                    **doctor_inputs,
                )
                if res:
                    cache_set(ck, res, ttl_seconds=seconds_to_ist_eod(),
                              surface="doctor_position", model="")
                return res
            except Exception as exc:
                logger.warning("finrobot run failed %s: %s", p.symbol, exc)
                return {
                    "symbol": p.symbol.upper(),
                    "narrative": "Analysis unavailable — engine did not return a result for this holding.",
                    "action": "hold",
                    "composite_score": 50,
                    "agents": {},
                }

    return await asyncio.gather(*[one(p) for p in positions])


def _diversification_score(weights: List[float]) -> int:
    """0-100, higher = better diversified. Effective number of holdings
    (1/HHI of normalized weights); ~10 effective holdings -> 100."""
    total = sum(weights) or 1.0
    hhi = sum((w / total) ** 2 for w in weights)
    if hhi <= 0:
        return 0
    effective_n = 1.0 / hhi
    return max(0, min(100, round(effective_n / 10.0 * 100)))


def _risk_score(weights: List[float], risk_flags: List[RiskFlag]) -> int:
    """0-100, higher = LOWER risk (matches the health-score framing). Starts at
    100 and deducts for flagged risks + single-name concentration."""
    penalty = 0
    for f in risk_flags:
        penalty += {"high": 28, "medium": 14, "low": 6}.get(f.severity, 6)
    total = sum(weights) or 1.0
    top = max((w / total for w in weights), default=0.0)
    if top >= 0.30:
        penalty += 16
    elif top >= 0.20:
        penalty += 8
    return max(0, min(100, 100 - penalty))


def _portfolio_action(composite: int, risk_flags: List[RiskFlag]) -> str:
    has_high_risk = any(f.severity == "high" for f in risk_flags)
    if has_high_risk:
        return "reduce_risk"
    if composite >= 70:
        return "hold"
    if composite >= 55:
        return "rebalance"
    return "reduce_risk"


def _portfolio_narrative(
    composite: int,
    action: str,
    risk_flags: List[RiskFlag],
    per_position: List[PerPositionResult],
    position_count: int,
) -> str:
    lines: List[str] = []
    lines.append(
        f"{position_count}-position portfolio scores {composite}/100 overall. "
        f"{public_label('cot_agents')} reviewed fundamentals, management, promoter, and peer data for each holding."
    )
    if risk_flags:
        high = [f for f in risk_flags if f.severity == "high"]
        med = [f for f in risk_flags if f.severity == "medium"]
        if high:
            lines.append(f"Flagged {len(high)} high-severity risk(s): " + "; ".join(f.message for f in high[:3]))
        if med:
            lines.append(f"{len(med)} medium-severity note(s): " + "; ".join(f.message for f in med[:3]))
    weakest = sorted(per_position, key=lambda p: p.composite_score)[:3]
    if weakest and weakest[0].composite_score < 50:
        lines.append(
            "Weakest scorers: "
            + ", ".join(f"{p.symbol} ({p.composite_score})" for p in weakest)
            + " — review before next rebalance."
        )
    action_copy = {
        "hold": "Action: hold as-is. Next review in ~30 days.",
        "rebalance": "Action: trim weakest holdings, rotate into stronger alternatives.",
        "reduce_risk": "Action: reduce equity exposure or raise cash until risk flags clear.",
        "increase_risk": "Action: composite is strong — scope is to deploy remaining cash.",
    }.get(action, "")
    if action_copy:
        lines.append(action_copy)
    return " ".join(lines)


# ============================================================================
# Routes
# ============================================================================


@router.post("/analyze", response_model=DoctorReport)
async def analyze_portfolio(
    body: AnalyzeRequest,
    user: UserTier = Depends(RequireFeature("portfolio_doctor_pro")),
    _cap: UserTier = Depends(enforce_llm_cap("portfolio_doctor")),
) -> DoctorReport:
    """Run the Portfolio Doctor over the submitted holdings.

    The 4-agent ``InsightAI`` engine fires per symbol (bounded to 4-way
    parallel). Returns per-position scores + portfolio composite + risk
    flags + a saved report ID.

    Tier+quota: Pro gets 1/month included, Elite unlimited, Free blocked
    (upgrade path via ``/pricing``).
    """
    quota_before = _assert_quota(user)

    positions = body.positions[:MAX_PORTFOLIO_SIZE]

    # Per-(portfolio, day) cache — Elite gets the deep model (separate key).
    deep = settings.LLM_DEEP_MODE_ENABLED and (user.tier == Tier.ELITE)
    ck = f"doctor:{_portfolio_hash(positions)}:{datetime.now(IST).date().isoformat()}:{'deep' if deep else 'std'}"
    cached = cache_get(ck)
    if cached:
        return DoctorReport(**cached)

    # Risk flags — pure Python, no LLM cost.
    risk_flags: List[RiskFlag] = []
    risk_flags.extend(_concentration_flags(positions))
    risk_flags.extend(_sector_flags(positions))
    risk_flags.extend(_regime_risk_flag())

    # Per-position CoT.
    raw = await _run_per_position(positions, user.user_id, deep=deep, tier=user.tier.value)

    per_position: List[PerPositionResult] = []
    agents_rollup: Dict[str, Any] = {}
    for p, r in zip(positions, raw):
        score = int(r.get("composite_score") or 50)
        per_position.append(PerPositionResult(
            symbol=p.symbol.upper(),
            weight=round(p.weight, 4),
            composite_score=max(0, min(100, score)),
            action=str(r.get("action") or "hold"),
            narrative=str(r.get("narrative") or ""),
        ))
        if r.get("agents"):
            agents_rollup[p.symbol.upper()] = r["agents"]

    # Weighted portfolio composite.
    total_w = sum(p.weight for p in positions) or 1.0
    composite = int(round(
        sum(pp.composite_score * (pp.weight / total_w) for pp in per_position)
    ))
    action = _portfolio_action(composite, risk_flags)
    narrative = _portfolio_narrative(composite, action, risk_flags, per_position, len(positions))
    weights = [p.weight for p in positions]
    diversification = _diversification_score(weights)
    risk_score = _risk_score(weights, risk_flags)

    # Persist.
    sb = get_supabase_admin()
    report_id: Optional[str] = None
    try:
        ins = sb.table("portfolio_doctor_reports").insert({
            "user_id": user.user_id,
            "source": body.source,
            "position_count": len(positions),
            "capital": body.capital,
            "composite_score": composite,
            "action": action,
            "narrative": narrative,
            "per_position": [pp.dict() for pp in per_position],
            "risk_flags": [f.dict() for f in risk_flags],
            "agents": agents_rollup,
        }).execute()
        if ins.data:
            report_id = str(ins.data[0].get("id"))
    except Exception as exc:
        logger.error("doctor report insert failed: %s", exc)

    try:
        from ..observability import EventName, track
        track(EventName.FINROBOT_ANALYSIS_COMPLETED, user.user_id, {
            "tier": user.tier.value,
            "source": "portfolio_doctor",
            "position_count": len(positions),
            "composite_score": composite,
            "action": action,
        })
    except Exception:
        pass

    runs_after = quota_before["runs_this_month"] + 1
    remaining = (quota_before["quota"] - runs_after) if quota_before.get("quota") is not None else None
    quota_out = {
        **quota_before,
        "runs_this_month": runs_after,
        "remaining": remaining,
    }

    report = DoctorReport(
        id=report_id or "unsaved",
        created_at=datetime.utcnow().isoformat(),
        source=body.source,
        position_count=len(positions),
        capital=body.capital,
        composite_score=composite,
        diversification_score=diversification,
        risk_score=risk_score,
        action=action,
        narrative=narrative,
        per_position=per_position,
        risk_flags=risk_flags,
        agents=agents_rollup,
        quota=quota_out,
    )
    cache_set(ck, report.dict(), ttl_seconds=seconds_to_ist_eod(), surface="doctor",
              model=settings.LLM_DEEP_MODEL if deep else settings.LLM_STRONG_MODEL)
    return report


def _position_returns(symbols: List[str]) -> Dict[str, List[float]]:
    """Daily returns per symbol (3mo) for correlation."""
    from ..data.market import get_market_data_provider
    mp = get_market_data_provider()
    out: Dict[str, List[float]] = {}
    for s in symbols:
        try:
            df = mp.get_historical(s, period="3mo", interval="1d")
            if df is None or len(df) <= 20:
                continue
            df.columns = [c.lower() for c in df.columns]
            closes = [float(c) for c in df["close"].tolist() if c == c]
            out[s] = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1]]
        except Exception:
            continue
    return out


@router.post("/rebalance")
async def rebalance(
    body: AnalyzeRequest,
    use_llm: bool = Query(True, description="Include a grounded rebalancing rationale"),
    user: UserTier = Depends(RequireFeature("portfolio_doctor_pro")),
):
    """AI Rebalancing (#19) — holdings correlation + concrete actions (trim
    overweight / cut weakest / diversify a concentrated sector / de-risk a
    correlated pair) + an optional grounded rationale. Real suggestions, not
    just the word 'rebalance'."""
    from ..ai.sector_taxonomy import sector_for_symbol
    from ..services.portfolio.portfolio_analytics import compute_correlation, rebalancing_suggestions

    positions = body.positions[:MAX_PORTFOLIO_SIZE]
    pos = [{"symbol": p.symbol.upper(), "weight": p.weight} for p in positions]
    syms = [p["symbol"] for p in pos]

    returns = await asyncio.to_thread(_position_returns, syms)
    corr = compute_correlation(returns)
    secmap = {s: (sector_for_symbol(s) or None) for s in syms}
    top_pair = corr["pairs"][0] if corr.get("pairs") else None
    suggestions = rebalancing_suggestions(pos, sector_by_symbol=secmap, top_corr_pair=top_pair)

    narrative = None
    if use_llm and (suggestions or corr.get("avg_corr") is not None):
        from ..ai.agents.grounded import grounded_reason
        narrative = await asyncio.to_thread(
            lambda: grounded_reason(
                {"positions": pos, "correlation": corr, "suggestions": suggestions, "sectors": secmap},
                "Give a concise rebalancing rationale: what to trim or add and why, given the weights, "
                "sector concentration and correlation.",
                cache_key=_rebal_cache_key(positions),
                user_id=user.user_id,
            ))
    return {"success": True, "correlation": corr, "suggestions": suggestions, "narrative": narrative}


@router.get("/reports", response_model=List[ReportRow])
async def list_reports(
    user: UserTier = Depends(RequireFeature("portfolio_doctor_pro")),
    limit: int = 20,
) -> List[ReportRow]:
    limit = max(1, min(50, int(limit)))
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("portfolio_doctor_reports")
            .select("id, created_at, source, position_count, composite_score, action")
            .eq("user_id", user.user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [ReportRow(**r) for r in rows.data or []]
    except Exception as exc:
        logger.warning("doctor reports list failed: %s", exc)
        return []


@router.get("/reports/{report_id}", response_model=DoctorReport)
async def get_report(
    report_id: str,
    user: UserTier = Depends(RequireFeature("portfolio_doctor_pro")),
) -> DoctorReport:
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("portfolio_doctor_reports")
            .select("*")
            .eq("id", report_id)
            .eq("user_id", user.user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("doctor report fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail="lookup_failed")
    row = (rows.data or [None])[0]
    if row is None:
        raise HTTPException(status_code=404, detail="not_found")
    hist_positions = [PerPositionResult(**pp) for pp in (row.get("per_position") or [])]
    hist_flags = [RiskFlag(**f) for f in (row.get("risk_flags") or [])]
    hist_weights = [pp.weight for pp in hist_positions]
    return DoctorReport(
        id=str(row["id"]),
        created_at=str(row["created_at"]),
        source=str(row.get("source") or "manual"),
        position_count=int(row.get("position_count") or 0),
        capital=row.get("capital"),
        composite_score=int(row.get("composite_score") or 0),
        diversification_score=_diversification_score(hist_weights),
        risk_score=_risk_score(hist_weights, hist_flags),
        action=str(row.get("action") or "hold"),
        narrative=str(row.get("narrative") or ""),
        per_position=hist_positions,
        risk_flags=hist_flags,
        agents=row.get("agents") or {},
        quota={
            "tier": user.tier.value,
            "runs_this_month": _current_month_runs(user.user_id),
            "quota": _quota_for(user.tier),
            "remaining": None,
        },
    )


@router.get("/quota")
async def get_quota(
    user: UserTier = Depends(RequireFeature("portfolio_doctor_pro")),
) -> Dict[str, Any]:
    quota = _quota_for(user.tier)
    runs = _current_month_runs(user.user_id)
    remaining = (quota - runs) if quota is not None else None
    return {
        "tier": user.tier.value,
        "runs_this_month": runs,
        "quota": quota,
        "remaining": remaining,
        "engine": public_label("cot_agents"),
    }


__all__ = ["router"]
