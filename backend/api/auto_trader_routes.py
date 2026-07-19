"""
================================================================================
AUTO-TRADER ROUTES — F4 AutoPilot dashboard (PR 28)
================================================================================
HTTP surface for ``/auto-trader`` — the F4 AutoPilot control plane + state
surface. The live engine is the SUPERVISED stack (Qlib top-N ranker + HMM
regime sizing + VIX overlay + Kelly-decayed weights), run by the daily 15:50
IST rebalance cron (trading/autopilot_service.py). Entry-side RL was removed
2026-05-23 (failed the Sharpe gate); narrow RL early-EXIT is separate.

Pricing v2: the PAPER (practice) bot is FREE for all tiers — every endpoint
takes ``current_user_tier`` (any authenticated user). Only *going live* (real
money) is gated to Pro+, enforced inside ``/toggle`` (402 upgrade otherwise).

Endpoints:

    GET  /api/auto-trader/status     — status strip payload
    GET  /api/auto-trader/config     — user safety rails
    PATCH /api/auto-trader/config    — update safety rails
    POST /api/auto-trader/toggle     — enable / pause
    GET  /api/auto-trader/trades     — today + last-7d auto-trader actions
    GET  /api/auto-trader/weekly     — weekly report summary

Kill switch lives at ``/api/trades/kill-switch`` (already wired in app.py).

================================================================================
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier, required_tier
from ..middleware.tier_gate import can_use_feature, current_user_tier
from ..platform.events import emit_event
from ..platform.realtime import MessageType
from ..data.market_calendar import IST

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auto-trader", tags=["auto-trader"])

RISK_PROFILES = {"conservative", "moderate", "aggressive"}

DEFAULT_CONFIG: Dict[str, Any] = {
    "risk_profile": "moderate",
    "max_position_pct": 7.0,
    "daily_loss_limit_pct": 2.0,
    "max_concurrent_positions": 12,
    "allow_fno": False,
}


# ============================================================================
# Models
# ============================================================================


class AutoTraderConfig(BaseModel):
    risk_profile: str = Field("moderate")
    max_position_pct: float = Field(7.0, ge=1.0, le=25.0)
    daily_loss_limit_pct: float = Field(2.0, ge=0.5, le=10.0)
    max_concurrent_positions: int = Field(12, ge=1, le=30)
    allow_fno: bool = Field(False)

    @validator("risk_profile")
    def _risk(cls, v: str) -> str:
        v = (v or "").lower()
        if v not in RISK_PROFILES:
            raise ValueError(f"risk_profile must be one of {sorted(RISK_PROFILES)}")
        return v


class ConfigPatch(BaseModel):
    risk_profile: Optional[str] = None
    max_position_pct: Optional[float] = Field(None, ge=1.0, le=25.0)
    daily_loss_limit_pct: Optional[float] = Field(None, ge=0.5, le=10.0)
    max_concurrent_positions: Optional[int] = Field(None, ge=1, le=30)
    allow_fno: Optional[bool] = None


class ToggleRequest(BaseModel):
    enabled: bool
    # Pricing v2 2026-06-12 — optional explicit mode switch. 'live' clears
    # a paper opt-in (real money, broker + suitability gates apply);
    # 'paper' keeps it virtual. None = leave the stored mode unchanged.
    mode: Optional[str] = None


class AutoTraderStatus(BaseModel):
    enabled: bool
    paused: bool          # kill_switch_active = true
    mode: str             # 'paper' (virtual) | 'live' — pricing v2 2026-06-12
    last_run_at: Optional[str]
    broker_connected: bool
    broker_name: Optional[str]
    open_positions: int
    today_trades: int
    today_pnl_pct: float
    regime: Optional[Dict[str, Any]]
    vix_band: Optional[str]
    equity_scaler_pct: int    # VIX → equity allocation hint
    config: AutoTraderConfig


class TradeRow(BaseModel):
    id: str
    symbol: str
    direction: str
    quantity: int
    entry_price: Optional[float]
    exit_price: Optional[float]
    status: str
    net_pnl: Optional[float]
    pnl_percent: Optional[float]
    created_at: Optional[str]
    closed_at: Optional[str]
    signal_id: Optional[str]


class RebalanceRunRow(BaseModel):
    """One rebalance tick. PR 69 — backs the dashboard rebalance log
    so a quiet day still shows "engine ran, no trade fired" rather than
    a blank trades list."""
    id: str
    ran_at: str
    regime: Optional[str]
    vix: Optional[float]
    vix_band: Optional[str]
    equity_scaler_pct: Optional[int]
    actions_count: int
    trades_executed: int
    summary: Optional[str]


# ============================================================================
# Helpers
# ============================================================================


def _load_profile(user_id: str) -> Dict[str, Any]:
    sb = get_supabase_admin()
    rows = (
        sb.table("user_profiles")
        .select(
            "id, tier, auto_trader_enabled, auto_trader_config, "
            "auto_trader_last_run_at, kill_switch_active"
        )
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    if not rows.data:
        raise HTTPException(status_code=404, detail="profile_not_found")
    return rows.data[0]


def _load_broker(user_id: str) -> Dict[str, Any]:
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("broker_connections")
            .select("broker_name, status, last_synced_at")
            .eq("user_id", user_id)
            .eq("status", "connected")
            .limit(1)
            .execute()
        )
        if rows.data:
            return rows.data[0]
    except Exception as exc:
        logger.debug("broker_connections lookup failed: %s", exc)
    return {}


def _merge_config(stored: Optional[Dict[str, Any]]) -> AutoTraderConfig:
    merged = {**DEFAULT_CONFIG, **(stored or {})}
    return AutoTraderConfig(**merged)


def _vix_band_and_scaler(vix: Optional[float]) -> tuple[Optional[str], int]:
    """Step 1 §F4 deterministic VIX risk overlay."""
    if vix is None:
        return None, 100
    if vix < 15:
        return "calm", 100
    if vix < 18:
        return "normal", 85
    if vix < 22:
        return "elevated", 70
    if vix < 27:
        return "high", 50
    if vix < 35:
        return "stressed", 30
    return "panic", 15


def _today_ist_start_utc() -> datetime:
    now_ist = datetime.now(IST)
    start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_ist.astimezone(timezone.utc)


def _load_regime_and_vix() -> tuple[Optional[Dict[str, Any]], Optional[float]]:
    sb = get_supabase_admin()
    regime: Optional[Dict[str, Any]] = None
    vix: Optional[float] = None
    try:
        rows = (
            sb.table("regime_history")
            .select("regime, prob_bull, prob_sideways, prob_bear, vix, detected_at")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
        )
        if rows.data:
            row = rows.data[0]
            regime = {
                "name": row.get("regime"),
                "prob_bull": row.get("prob_bull"),
                "prob_sideways": row.get("prob_sideways"),
                "prob_bear": row.get("prob_bear"),
                "as_of": row.get("as_of"),
            }
            if row.get("vix") is not None:
                vix = float(row["vix"])
    except Exception as exc:
        logger.debug("regime_history lookup failed: %s", exc)
    return regime, vix


def _today_trade_stats(user_id: str) -> tuple[int, float]:
    sb = get_supabase_admin()
    start_iso = _today_ist_start_utc().isoformat()
    try:
        rows = (
            sb.table("trades")
            .select("id, pnl_percent, status, execution_mode")
            .eq("user_id", user_id)
            .eq("execution_mode", "live")
            .gte("created_at", start_iso)
            .limit(200)
            .execute()
        )
        data = rows.data or []
    except Exception as exc:
        logger.debug("today trade stats lookup failed: %s", exc)
        return 0, 0.0
    n = len(data)
    closed = [r for r in data if r.get("status") == "closed" and r.get("pnl_percent") is not None]
    pnl = sum(float(r["pnl_percent"] or 0.0) for r in closed)
    return n, pnl


def _open_positions_count(user_id: str) -> int:
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("positions")
            .select("id")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .limit(200)
            .execute()
        )
        return len(rows.data or [])
    except Exception as exc:
        logger.debug("positions count lookup failed: %s", exc)
        return 0


# ============================================================================
# Routes
# ============================================================================


@router.get("/status", response_model=AutoTraderStatus)
async def get_status(
    user: UserTier = Depends(current_user_tier),
) -> AutoTraderStatus:
    profile = _load_profile(user.user_id)
    broker = _load_broker(user.user_id)
    regime, vix = _load_regime_and_vix()
    today_n, today_pnl = _today_trade_stats(user.user_id)
    band, scaler = _vix_band_and_scaler(vix)

    from ..core.tiers import resolve_autopilot_mode
    last_run = profile.get("auto_trader_last_run_at")
    return AutoTraderStatus(
        enabled=bool(profile.get("auto_trader_enabled", False)),
        paused=bool(profile.get("kill_switch_active", False)),
        mode=resolve_autopilot_mode(user.tier, profile.get("auto_trader_config")),
        last_run_at=str(last_run) if last_run else None,
        broker_connected=bool(broker),
        broker_name=broker.get("broker_name"),
        open_positions=_open_positions_count(user.user_id),
        today_trades=today_n,
        today_pnl_pct=round(today_pnl, 2),
        regime=regime,
        vix_band=band,
        equity_scaler_pct=scaler,
        config=_merge_config(profile.get("auto_trader_config")),
    )


@router.get("/config", response_model=AutoTraderConfig)
async def get_config(
    user: UserTier = Depends(current_user_tier),
) -> AutoTraderConfig:
    profile = _load_profile(user.user_id)
    return _merge_config(profile.get("auto_trader_config"))


@router.patch("/config", response_model=AutoTraderConfig)
async def patch_config(
    body: ConfigPatch,
    user: UserTier = Depends(current_user_tier),
) -> AutoTraderConfig:
    profile = _load_profile(user.user_id)
    merged_dict = {**DEFAULT_CONFIG, **(profile.get("auto_trader_config") or {})}
    patch = body.dict(exclude_unset=True, exclude_none=True)
    if "risk_profile" in patch and patch["risk_profile"] not in RISK_PROFILES:
        raise HTTPException(status_code=422, detail="invalid_risk_profile")
    merged_dict.update(patch)
    new_config = AutoTraderConfig(**merged_dict)

    # Pricing v2 2026-06-12 — preserve raw non-schema keys (notably
    # ``mode='paper'`` set by the managed paper toggle). A schema-only
    # full replace here would silently flip a paper user to live money.
    raw_existing = profile.get("auto_trader_config") or {}
    stored = {
        **{k: v for k, v in raw_existing.items() if k not in AutoTraderConfig.__fields__},
        **new_config.dict(),
    }

    sb = get_supabase_admin()
    sb.table("user_profiles").update(
        {"auto_trader_config": stored}
    ).eq("id", user.user_id).execute()

    logger.info("auto_trader.config updated user=%s patch=%s", user.user_id, patch)
    return new_config


@router.post("/toggle")
async def toggle(
    body: ToggleRequest,
    user: UserTier = Depends(current_user_tier),
) -> Dict[str, Any]:
    """Enable or pause the auto-trader.

    Enabling while ``kill_switch_active=true`` first clears the kill
    switch — treat it as the user explicitly opting back in.
    """
    sb = get_supabase_admin()
    if body.mode is not None and body.mode not in {"live", "paper"}:
        raise HTTPException(status_code=422, detail="invalid_mode")

    # Pricing v2: the PAPER bot is Free (all tiers); only LIVE money requires
    # Pro+. Free users default to paper and cannot switch to live.
    can_live = bool(getattr(user, "is_admin", False)) or can_use_feature(user.tier, "auto_trader")

    # Resolve the mode this toggle results in: explicit body.mode wins, else the
    # stored mode, else live for paid tiers / paper for free.
    profile_cfg = (_load_profile(user.user_id) or {}).get("auto_trader_config") or {}
    stored_mode = "paper" if profile_cfg.get("mode") == "paper" else ("live" if can_live else "paper")
    effective_mode = body.mode or stored_mode

    # Live money is a paid feature — refuse (402 upgrade) but keep paper free.
    if effective_mode == "live" and not can_live:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "tier_gate",
                "feature": "auto_trader",
                "required_tier": required_tier("auto_trader").value,
                "current_tier": user.tier.value,
                "upgrade_url": "/pricing",
                "message": (
                    "Live AutoPilot is a Pro feature. The practice (paper) bot is "
                    "free — run it in practice mode now, upgrade to go live."
                ),
            },
        )

    broker = _load_broker(user.user_id)
    if body.enabled and effective_mode == "live" and not broker:
        raise HTTPException(
            status_code=400,
            detail="broker_not_connected",
        )

    # HIGH #6 (2026-05-31) — Suitability gate enforcement.
    # When enabling LIVE, verify the user has completed the risk-profile
    # quiz (lives at /onboarding/risk-quiz). Paper mode is virtual money —
    # no suitability obligation. Conservative profiles are additionally
    # blocked from F&O auto-execution per Terms §2.
    if body.enabled and effective_mode == "live":
        from ..core.config import settings
        if settings.REQUIRE_SUITABILITY_QUIZ_FOR_LIVE:
            prof = (
                sb.table("user_profiles")
                .select("risk_profile, onboarding_completed")
                .eq("id", user.user_id)
                .single()
                .execute()
            )
            row = prof.data or {}
            if not row.get("onboarding_completed"):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "code": "suitability_quiz_required",
                        "message": (
                            "Complete the in-app risk profile assessment at "
                            "/onboarding/risk-quiz before enabling live AutoPilot. "
                            "Required by Terms §2 + SEBI suitability obligations."
                        ),
                        "next_url": "/onboarding/risk-quiz",
                    },
                )

    update: Dict[str, Any] = {"auto_trader_enabled": bool(body.enabled)}
    if body.enabled:
        update["kill_switch_active"] = False
    if body.mode is not None:
        # Persist the explicit mode switch in the raw config blob.
        update["auto_trader_config"] = {**profile_cfg, "mode": body.mode}

    sb.table("user_profiles").update(update).eq("id", user.user_id).execute()

    try:
        event_type = (
            MessageType.AUTO_TRADE_EXECUTED if body.enabled
            else MessageType.AUTO_TRADE_BLOCKED
        )
        await emit_event(
            event_type,
            {"action": "enabled" if body.enabled else "paused"},
            user_id=user.user_id,
            broadcast=False,
        )
    except Exception:
        pass

    try:
        from ..observability import EventName, track
        track(
            EventName.AUTO_TRADE_EXECUTED if body.enabled else EventName.AUTO_TRADE_BLOCKED,
            user.user_id,
            {"source": "auto_trader_toggle", "enabled": bool(body.enabled)},
        )
    except Exception:
        pass

    return {"enabled": bool(body.enabled), "ok": True}


@router.get("/trades", response_model=List[TradeRow])
async def get_recent_trades(
    user: UserTier = Depends(current_user_tier),
    days: int = 7,
) -> List[TradeRow]:
    days = max(1, min(30, int(days)))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("trades")
            .select(
                "id, symbol, direction, quantity, entry_price, exit_price, "
                "status, net_pnl, pnl_percent, created_at, closed_at, signal_id"
            )
            .eq("user_id", user.user_id)
            .eq("execution_mode", "live")
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
    except Exception as exc:
        logger.error("auto_trader.trades lookup failed: %s", exc)
        return []
    out: List[TradeRow] = []
    for r in rows.data or []:
        out.append(
            TradeRow(
                id=str(r.get("id")),
                symbol=r.get("symbol") or "",
                direction=r.get("direction") or "LONG",
                quantity=int(r.get("quantity") or 0),
                entry_price=r.get("entry_price"),
                exit_price=r.get("exit_price"),
                status=r.get("status") or "pending",
                net_pnl=r.get("net_pnl"),
                pnl_percent=r.get("pnl_percent"),
                created_at=r.get("created_at"),
                closed_at=r.get("closed_at"),
                signal_id=r.get("signal_id"),
            )
        )
    return out


@router.get("/weekly")
async def get_weekly_summary(
    user: UserTier = Depends(current_user_tier),
) -> Dict[str, Any]:
    """Rolling 7-day summary for the weekly-report panel."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("trades")
            .select("id, status, pnl_percent, net_pnl, symbol, direction")
            .eq("user_id", user.user_id)
            .eq("execution_mode", "live")
            .gte("created_at", cutoff)
            .limit(500)
            .execute()
        )
        data = rows.data or []
    except Exception as exc:
        logger.error("auto_trader.weekly lookup failed: %s", exc)
        data = []

    closed = [r for r in data if r.get("status") == "closed"]
    wins = [r for r in closed if (r.get("pnl_percent") or 0) > 0]
    losses = [r for r in closed if (r.get("pnl_percent") or 0) <= 0]
    win_rate = (len(wins) / len(closed)) if closed else 0.0
    total_pnl_pct = sum(float(r.get("pnl_percent") or 0) for r in closed)
    net_pnl = sum(float(r.get("net_pnl") or 0) for r in closed)

    symbols_traded = sorted({r.get("symbol") for r in data if r.get("symbol")})

    return {
        "days": 7,
        "trades_executed": len(data),
        "trades_closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 3),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "net_pnl": round(net_pnl, 2),
        "symbols": symbols_traded[:25],
    }


@router.get("/runs", response_model=List[RebalanceRunRow])
async def get_recent_runs(
    user: UserTier = Depends(current_user_tier),
    limit: int = 10,
) -> List[RebalanceRunRow]:
    """Last N rebalance ticks for the dashboard log. Rows are written live by
    the supervised AutoPilot daily rebalance (trading/autopilot_service.py);
    the frontend renders an empty state until the first run lands."""
    limit = max(1, min(50, int(limit)))
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("auto_trader_runs")
            .select(
                "id, ran_at, regime, vix, vix_band, equity_scaler_pct, "
                "actions_count, trades_executed, summary"
            )
            .eq("user_id", user.user_id)
            .order("ran_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        logger.debug("auto_trader.runs lookup failed: %s", exc)
        return []
    out: List[RebalanceRunRow] = []
    for r in rows.data or []:
        out.append(
            RebalanceRunRow(
                id=str(r.get("id")),
                ran_at=str(r.get("ran_at") or ""),
                regime=r.get("regime"),
                vix=r.get("vix"),
                vix_band=r.get("vix_band"),
                equity_scaler_pct=r.get("equity_scaler_pct"),
                actions_count=int(r.get("actions_count") or 0),
                trades_executed=int(r.get("trades_executed") or 0),
                summary=r.get("summary"),
            )
        )
    return out


# ============================================================================
# PR 133 — today's AutoPilot plan + overlay diagnostics
# ============================================================================
#
# Surfaces the latest auto_trader_runs row for the calling user with:
#   • target_weights (the FinRL-X ensemble's blended action vector)
#   • diagnostics    (PR 132 VIX overlay + bear-scale + VaR cap output)
#
# The dashboard renders the diagnostic blob as the "AI moved 20% to
# cash because VIX spiked to 22" caption.


class TodayPlan(BaseModel):
    ran_at: Optional[str] = None
    regime: Optional[str] = None
    target_weights: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}
    status: Optional[str] = None


@router.get("/plan/today", response_model=TodayPlan)
async def get_today_plan(
    user: UserTier = Depends(current_user_tier),
) -> TodayPlan:
    """Latest decision row for the user. Empty payload when AutoPilot
    hasn't run yet today (dashboard renders the empty-state copy)."""
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("auto_trader_runs")
            .select("ran_at, regime, target_weights, diagnostics, status")
            .eq("user_id", user.user_id)
            .order("ran_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.debug("auto_trader.plan/today lookup failed: %s", exc)
        return TodayPlan()
    if not rows.data:
        return TodayPlan()
    r = rows.data[0]
    return TodayPlan(
        ran_at=str(r.get("ran_at") or "") or None,
        regime=r.get("regime"),
        target_weights=r.get("target_weights") or {},
        diagnostics=r.get("diagnostics") or {},
        status=r.get("status"),
    )


# ────────────────────────────────────────────────────────────────────
# CRITICAL #2 (2026-05-31) — Live track record endpoint
# ────────────────────────────────────────────────────────────────────


@router.get("/track-record")
async def get_track_record(
    window_days: int = 30,
    source: str = "paper",
    user: UserTier = Depends(current_user_tier),
):
    """User-facing realised P&L track record over `window_days`.

    Reads from `autopilot_track_record_daily` snapshot table (populated
    by the daily cron) for fast response. Falls back to live aggregation
    when no snapshot exists yet (early days).

    Surface is BRAND-SAFE per memory `project_greek_branding_2026_04_19`:
    returns OUTCOMES (Sharpe, win-rate, drawdown) but NEVER per-model
    decomposition — that stays admin-only on a separate endpoint.
    """
    if source not in ("paper", "live"):
        raise HTTPException(400, "source must be 'paper' or 'live'")
    if window_days not in (30, 60, 90):
        raise HTTPException(400, "window_days must be 30, 60, or 90")
    sb = get_supabase_admin()

    # Try snapshot first
    try:
        snap = (
            sb.table("autopilot_track_record_daily")
            .select("*")
            .eq("source", source)
            .eq("window_days", window_days)
            .is_("user_id", "null")   # system-wide aggregate
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
        )
        if snap.data:
            row = snap.data[0]
            row["surface"] = "snapshot"
            return row
    except Exception as e:
        logger.debug("track-record snapshot lookup failed: %s", e)

    # Fallback: live aggregate
    try:
        from ..services.autopilot.track_record import aggregate_track_record
        summary = aggregate_track_record(
            sb, user_id=None, window_days=window_days, source=source,
        )
        out = summary.to_dict()
        out["surface"] = "live_aggregate"
        return out
    except Exception as e:
        logger.warning("track-record live aggregate failed: %s", e)
        raise HTTPException(503, f"track record unavailable: {e}")


@router.get("/compliance")
async def get_compliance():
    """HIGH #6 — public compliance metadata for the disclaimer footer.

    Returned on every dashboard load so the SEBI RA registration number
    + suitability requirements appear with every signal/trade context.

    No-auth: legal disclosures are public-by-design.
    """
    from ..core.config import settings

    reg = (settings.SEBI_RA_REG_NUMBER or "").strip()
    # Only assert RA registration when a REAL approved number is configured —
    # never while it's the placeholder (else we'd make a false regulatory claim).
    registered = bool(reg) and reg.upper() not in ("PENDING_APPROVAL", "PENDING", "")

    if registered:
        disclaimer_short = (
            "AutoPilot signals are general-market technology research. "
            "Not personalised investment advice. Trading involves risk of capital loss. "
            f"SEBI RA Reg #: {reg}"
        )
        disclaimer_long = (
            f"Quant X is a SEBI Research Analyst (registration {reg}) operating as "
            "a technology platform. All signals are general-market research, not "
            "personalised investment advice. Each user is responsible for assessing "
            "suitability based on their own risk profile, capital, and objectives. "
            "Past performance is no guarantee of future results. Trading in equity "
            "and derivatives involves substantial risk of loss, including loss of capital."
        )
    else:
        disclaimer_short = (
            "AutoPilot signals are general-market technology research — educational "
            "only, not personalised investment advice. Trading involves risk of "
            "capital loss. SEBI Research Analyst registration is pending."
        )
        disclaimer_long = (
            "Quant X operates as a technology platform; SEBI Research Analyst "
            "registration is pending. All signals are general-market research/"
            "educational information, NOT personalised investment advice. Each user "
            "is responsible for assessing suitability based on their own risk "
            "profile, capital, and objectives. Past performance is no guarantee of "
            "future results. Trading in equity and derivatives involves substantial "
            "risk of loss, including loss of capital."
        )

    return {
        "sebi_ra_reg_number": reg if registered else None,
        "sebi_ra_registered": registered,
        "sebi_ra_valid_until": settings.SEBI_RA_VALID_UNTIL or None,
        "requires_suitability_quiz_for_live": settings.REQUIRE_SUITABILITY_QUIZ_FOR_LIVE,
        "disclaimer_short": disclaimer_short,
        "disclaimer_long": disclaimer_long,
    }


__all__ = ["router"]
