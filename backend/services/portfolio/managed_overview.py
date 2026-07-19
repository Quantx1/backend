"""
Managed-mode overview — the single aggregate behind the beginner Home.

Dual-mode 2026-06-12: users in "managed" mode get a simple dashboard
(health score, money, risk level, what AutoPilot did) instead of the pro
terminal. This service composes that page's entire payload in one call
from REAL data only — deterministic, zero LLM tokens, zero quota burn,
honest-null wherever a source is missing.

NOT tier-gated: Free/Pro users get the same payload with
``autopilot.available = False`` so the UI can show an honest upsell
instead of a 403. The auto-trader's own routes stay Elite-gated.

Health score is transparent: it starts at 100 and every deduction is
itemised in ``health.components`` so the UI can explain the number.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Deductions per risk-warning severity. Warnings come from
# services.portfolio.user_risk.check_risk (day-loss, single-name, sector, exposure).
_SEVERITY_IMPACT = {"high": -15, "medium": -10, "low": -5}
_MAX_WARNING_DEDUCTION = -45
_APPROACH_FRACTION = 0.8  # day loss ≥80% of limit → early warning deduction


def _sb() -> Any:
    from ...core.database import get_supabase_admin
    return get_supabase_admin()


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# Source reads — every one best-effort; failures return empty/None.
# ──────────────────────────────────────────────────────────────────────────


def _profile_row(sb: Any, user_id: str) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("user_profiles")
            .select(
                "capital, total_pnl, total_trades, winning_trades, "
                "risk_profile, auto_trader_enabled, kill_switch_active, "
                "auto_trader_last_run_at, auto_trader_config"
            )
            .eq("id", user_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed_overview: profile read failed: %s", exc)
        return {}


def _unrealized_pnl(sb: Any, user_id: str) -> Optional[float]:
    try:
        rows = (
            sb.table("positions")
            .select("unrealized_pnl")
            .eq("user_id", user_id)
            .eq("is_active", True)
            .limit(100)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed_overview: positions read failed: %s", exc)
        return None
    total = 0.0
    for r in rows:
        p = _num(r.get("unrealized_pnl"))
        if p is not None:
            total += p
    return round(total, 2)


def _live_trades(
    sb: Any, user_id: str, days: int = 7, mode: str = "live",
) -> List[Dict[str, Any]]:
    """Recent AutoPilot trades in the user's execution mode — paper-mode
    users (Free, or Pro/Elite trialling) see their virtual activity."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        return (
            sb.table("trades")
            .select(
                "symbol, direction, quantity, entry_price, exit_price, "
                "status, net_pnl, pnl_percent, created_at, closed_at"
            )
            .eq("user_id", user_id)
            .eq("execution_mode", mode)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(30)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed_overview: trades read failed: %s", exc)
        return []


def _latest_regime(sb: Any) -> Optional[Dict[str, Any]]:
    try:
        rows = (
            sb.table("regime_history")
            .select("regime, prob_bull, prob_sideways, prob_bear, detected_at")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed_overview: regime read failed: %s", exc)
        return None
    if not rows:
        return None
    r = rows[0]
    return {
        "name": r.get("regime"),
        "prob_bull": _num(r.get("prob_bull")),
        "prob_sideways": _num(r.get("prob_sideways")),
        "prob_bear": _num(r.get("prob_bear")),
        "as_of": r.get("detected_at"),
    }


def _latest_drawdown(sb: Any, user_id: str) -> Optional[Dict[str, Any]]:
    try:
        rows = (
            sb.table("paper_snapshots")
            .select("snapshot_date, drawdown_pct")
            .eq("user_id", user_id)
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed_overview: snapshot read failed: %s", exc)
        return None
    if not rows:
        return None
    dd = _num(rows[0].get("drawdown_pct"))
    if dd is None:
        return None
    return {"current_pct": round(dd, 2), "as_of": rows[0].get("snapshot_date")}


# ──────────────────────────────────────────────────────────────────────────
# Plain-English activity lines (deterministic — no LLM)
# ──────────────────────────────────────────────────────────────────────────


def _fmt_money(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"₹{v:,.0f}" if abs(v) >= 100 else f"₹{v:,.2f}"


def _activity_lines(trades: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if profile.get("kill_switch_active"):
        lines.append("AutoPilot is paused by your kill switch — no new trades until you resume.")
    for t in trades:
        sym = t.get("symbol") or "?"
        qty = int(_num(t.get("quantity")) or 0)
        status = (t.get("status") or "").lower()
        direction = (t.get("direction") or "LONG").upper()
        if status == "closed":
            pct = _num(t.get("pnl_percent"))
            pnl = _num(t.get("net_pnl"))
            pct_s = f"{pct:+.1f}%" if pct is not None else ""
            lines.append(f"Closed {sym} {pct_s} ({_fmt_money(pnl)})".replace("  ", " "))
        else:
            entry = _num(t.get("entry_price"))
            verb = "Bought" if direction == "LONG" else "Sold short"
            at = f" @ {_fmt_money(entry)}" if entry is not None else ""
            lines.append(f"{verb} {qty} {sym}{at}")
    return lines


# ──────────────────────────────────────────────────────────────────────────
# Health score — transparent deductions
# ──────────────────────────────────────────────────────────────────────────


def _health(risk: Dict[str, Any]) -> Dict[str, Any]:
    components: List[Dict[str, Any]] = []
    warning_total = 0
    for w in risk.get("warnings") or []:
        impact = _SEVERITY_IMPACT.get(w.get("severity"), -5)
        warning_total += impact
        components.append({
            "key": w.get("key"),
            "label": "Risk warning",
            "impact": impact,
            "detail": w.get("message"),
        })
    if warning_total < _MAX_WARNING_DEDUCTION:
        # Cap so a pile of medium flags can't zero the score on its own.
        adjust = _MAX_WARNING_DEDUCTION - warning_total
        components.append({
            "key": "warning_cap",
            "label": "Deduction cap",
            "impact": adjust,
            "detail": "Risk-warning deductions are capped at 45 points.",
        })
        warning_total = _MAX_WARNING_DEDUCTION

    score = 100 + warning_total

    # Early signal: day loss approaching (but not yet breaching) the limit.
    day_pnl = _num(risk.get("day_pnl"))
    limit_pct = _num(risk.get("daily_loss_limit_pct"))
    capital = _num(risk.get("capital"))
    has_day_loss_warning = any(
        (w.get("key") == "day_loss") for w in (risk.get("warnings") or [])
    )
    if (
        not has_day_loss_warning
        and day_pnl is not None and day_pnl < 0
        and limit_pct and capital and capital > 0
    ):
        limit_amt = capital * limit_pct / 100.0
        if limit_amt > 0 and abs(day_pnl) >= _APPROACH_FRACTION * limit_amt:
            score -= 10
            components.append({
                "key": "day_loss_approach",
                "label": "Approaching daily loss limit",
                "impact": -10,
                "detail": (
                    f"Today's P&L ({_fmt_money(day_pnl)}) is over "
                    f"{int(_APPROACH_FRACTION * 100)}% of your daily loss limit."
                ),
            })

    score = max(0, min(100, score))
    label = "Healthy" if score >= 85 else "Watch" if score >= 65 else "At risk"
    return {"score": score, "label": label, "components": components}


# ──────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────


def build_overview(user_id: str) -> Dict[str, Any]:
    """Everything the managed Home needs, in one deterministic payload."""
    from .user_risk import risk_status

    risk = risk_status(user_id)  # never raises; honest-empty inside

    sb = None
    try:
        sb = _sb()
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed_overview: supabase unavailable: %s", exc)

    # AutoPilot availability is a tier fact; mode (paper/live) combines
    # tier + the user's paper opt-in (Free is always paper).
    available = False
    ap_mode = "paper"
    user_tier = None
    try:
        from ...core.tiers import required_tier, resolve_user_tier, tier_rank
        ut = resolve_user_tier(user_id)
        user_tier = ut.tier
        available = tier_rank(ut.tier) >= tier_rank(required_tier("auto_trader"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("managed_overview: tier resolve failed: %s", exc)

    profile: Dict[str, Any] = {}
    unrealized: Optional[float] = None
    trades: List[Dict[str, Any]] = []
    regime: Optional[Dict[str, Any]] = None
    drawdown: Optional[Dict[str, Any]] = None
    if sb is not None:
        profile = _profile_row(sb, user_id)
        if user_tier is not None:
            try:
                from ...core.tiers import resolve_autopilot_mode as _ram
                ap_mode = _ram(user_tier, profile.get("auto_trader_config"))
            except Exception:  # noqa: BLE001
                ap_mode = "paper"
        unrealized = _unrealized_pnl(sb, user_id)
        trades = _live_trades(sb, user_id, mode=ap_mode)
        regime = _latest_regime(sb)
        drawdown = _latest_drawdown(sb, user_id)

    total_trades = int(_num(profile.get("total_trades")) or 0)
    winning = int(_num(profile.get("winning_trades")) or 0)
    win_rate = round(winning / total_trades * 100, 1) if total_trades > 0 else None

    return {
        "health": _health(risk),
        "pnl": {
            "capital": _num(profile.get("capital")),
            "total_pnl": _num(profile.get("total_pnl")),
            "today_pnl": _num(risk.get("day_pnl")),
            "unrealized_pnl": unrealized,
            "total_trades": total_trades,
            "win_rate": win_rate,
        },
        "risk": {
            "level": risk.get("risk_profile"),
            "flags": risk.get("warnings") or [],
            "ok": bool(risk.get("ok", True)),
            "day_pnl": _num(risk.get("day_pnl")),
            "daily_loss_limit_pct": _num(risk.get("daily_loss_limit_pct")),
            "positions_count": risk.get("positions_count"),
            "positions_value": _num(risk.get("positions_value")),
        },
        "autopilot": {
            "available": available,
            "mode": ap_mode,
            "enabled": bool(profile.get("auto_trader_enabled", False)),
            "paused": bool(profile.get("kill_switch_active", False)),
            "last_run_at": profile.get("auto_trader_last_run_at"),
            "activity": _activity_lines(trades, profile),
            "trades_7d": len(trades),
        },
        "regime": regime,
        "drawdown": drawdown,
    }


__all__ = ["build_overview"]
