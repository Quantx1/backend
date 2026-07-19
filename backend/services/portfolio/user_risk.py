"""User-level Risk Manager — deterministic pre-trade + status WARNINGS.

Warn, NEVER block: nothing here gates, sizes, or rejects an order. The
checks are pure arithmetic over the user's risk profile + today's P&L +
open positions, mirroring the P&L conventions of
``services/strategy_runner/day_loss_breaker.py`` (realized = closed-today
rows, exposure marked against capital).

Two entry points:
  ``check_risk``   — PURE over plain dicts (tested, 0 tokens, no I/O).
  ``risk_status``  — best-effort loader: user_profiles risk settings,
                     today's realized paper/live P&L, open paper + platform
                     positions (same reads the paper/positions endpoints
                     use), then delegates to ``check_risk``.

Honest-empty: missing inputs → no warnings, never a fabricated number.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default daily loss limit (% of capital) by onboarding risk profile.
# Overridden by an explicit ``daily_loss_limit_pct`` on the profile dict
# (user_profiles.daily_loss_limit — the Settings → Risk slider).
PROFILE_DAY_LOSS_PCT: Dict[str, float] = {
    "conservative": 2.0,
    "moderate": 3.0,
    "aggressive": 5.0,
}

SINGLE_NAME_CAP_PCT = 20.0   # one symbol > 20% of capital → warn
SECTOR_CAP_PCT = 40.0        # one sector > 40% of capital → warn
TOTAL_EXPOSURE_CAP_PCT = 100.0  # deployed > 100% of capital → warn


def _num(v: Any) -> Optional[float]:
    """Coerce to float, or None if missing/unparseable. Never fabricates."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def effective_day_loss_limit_pct(profile: Optional[Dict[str, Any]]) -> Optional[float]:
    """The daily loss limit (% of capital) this user is judged against.

    Explicit ``daily_loss_limit_pct`` wins; else the risk-profile default;
    else None (no profile info at all → no day-loss check, honest-empty).
    """
    p = profile or {}
    explicit = _num(p.get("daily_loss_limit_pct"))
    if explicit is not None and explicit > 0:
        return explicit
    rp = str(p.get("risk_profile") or "").strip().lower()
    return PROFILE_DAY_LOSS_PCT.get(rp)


def check_risk(
    profile: Optional[Dict[str, Any]],
    day_pnl: Optional[float],
    capital: Optional[float],
    positions: Optional[List[Dict[str, Any]]],
    proposed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """PURE deterministic risk warnings. Warn-only — callers must never
    use this to block or size an order.

    profile   — {daily_loss_limit_pct?, risk_profile?}
    day_pnl   — today's realized P&L in ₹ (negative when losing)
    capital   — the user's capital base in ₹
    positions — [{symbol, sector?, value}] open positions, value in ₹
    proposed  — {symbol, sector?, value} order being considered, or None

    Returns {"warnings": [{key, severity, message}], "ok": bool}.
    Honest-empty: capital missing/<=0 → no warnings at all.
    """
    warnings: List[Dict[str, str]] = []
    cap = _num(capital)
    if cap is None or cap <= 0:
        return {"warnings": [], "ok": True}

    # ── 1. Day-loss (mirrors the day-loss breaker comparison: <= -limit) ──
    limit_pct = effective_day_loss_limit_pct(profile)
    pnl = _num(day_pnl)
    if limit_pct is not None and limit_pct > 0 and pnl is not None:
        limit_amount = cap * limit_pct / 100.0
        if pnl <= -limit_amount:
            warnings.append({
                "key": "day_loss",
                "severity": "high",
                "message": (
                    f"Today's realized P&L ₹{pnl:,.0f} has crossed your daily "
                    f"loss limit of {limit_pct:g}% (₹{limit_amount:,.0f}). "
                    "Consider stepping back for the day."
                ),
            })

    # ── 2. Build exposure buckets (existing + proposed) ──
    prop_sym: Optional[str] = None
    prop_sector: Optional[str] = None
    prop_value = 0.0
    if proposed:
        prop_sym = str(proposed.get("symbol") or "").strip().upper() or None
        sec = proposed.get("sector")
        prop_sector = str(sec).strip() or None if sec else None
        prop_value = _num(proposed.get("value")) or 0.0

    by_symbol: Dict[str, float] = {}
    by_sector: Dict[str, float] = {}
    total_value = 0.0
    for row in positions or []:
        sym = str(row.get("symbol") or "").strip().upper()
        val = _num(row.get("value"))
        if not sym or val is None or val <= 0:
            continue
        by_symbol[sym] = by_symbol.get(sym, 0.0) + val
        total_value += val
        sec = row.get("sector")
        if sec:
            sec_name = str(sec).strip()
            if sec_name:
                by_sector[sec_name] = by_sector.get(sec_name, 0.0) + val

    if prop_sym and prop_value > 0:
        by_symbol[prop_sym] = by_symbol.get(prop_sym, 0.0) + prop_value
        total_value += prop_value
        if prop_sector:
            by_sector[prop_sector] = by_sector.get(prop_sector, 0.0) + prop_value

    # ── 3. Single-name concentration (> 20% of capital) ──
    for sym in sorted(by_symbol):
        pct = by_symbol[sym] / cap * 100.0
        if pct > SINGLE_NAME_CAP_PCT:
            incl = " including this order" if prop_sym == sym and prop_value > 0 else ""
            warnings.append({
                "key": f"single_name:{sym}",
                "severity": "medium",
                "message": (
                    f"{sym} would be {pct:.1f}% of your capital{incl} — above "
                    f"the {SINGLE_NAME_CAP_PCT:g}% single-stock guideline."
                ),
            })

    # ── 4. Sector concentration (> 40% of capital) ──
    for sec_name in sorted(by_sector):
        pct = by_sector[sec_name] / cap * 100.0
        if pct > SECTOR_CAP_PCT:
            incl = (
                " including this order"
                if prop_sector == sec_name and prop_value > 0 else ""
            )
            warnings.append({
                "key": f"sector:{sec_name}",
                "severity": "medium",
                "message": (
                    f"{sec_name} exposure would be {pct:.1f}% of your capital"
                    f"{incl} — above the {SECTOR_CAP_PCT:g}% sector guideline."
                ),
            })

    # ── 5. Total exposure (> 100% of capital) ──
    if total_value > 0:
        pct = total_value / cap * 100.0
        if pct > TOTAL_EXPOSURE_CAP_PCT:
            incl = " including this order" if prop_value > 0 else ""
            warnings.append({
                "key": "total_exposure",
                "severity": "high",
                "message": (
                    f"Total deployed ₹{total_value:,.0f} is {pct:.1f}% of your "
                    f"capital{incl} — over {TOTAL_EXPOSURE_CAP_PCT:g}%."
                ),
            })

    return {"warnings": warnings, "ok": len(warnings) == 0}


# ──────────────────────────────────────────────────────────────────────────
# Loader — best-effort reads, same patterns as the paper/positions routes
# ──────────────────────────────────────────────────────────────────────────


def _sb() -> Any:
    from ...core.database import get_supabase_admin
    return get_supabase_admin()


def risk_status(user_id: str) -> Dict[str, Any]:
    """Current risk picture for one user: profile + today's realized
    paper/live P&L + open positions → ``check_risk`` with proposed=None,
    plus the raw numbers so the UI can show them. Every read is
    best-effort; anything that fails is simply absent (honest-empty)."""
    today_iso = date.today().isoformat()
    sb = None
    try:
        sb = _sb()
    except Exception as exc:
        logger.debug("user_risk: supabase unavailable: %s", exc)

    # ── profile (user_profiles risk settings, same row Settings edits) ──
    profile_row: Dict[str, Any] = {}
    if sb is not None:
        try:
            rows = (
                sb.table("user_profiles")
                .select("risk_profile, daily_loss_limit, capital")
                .eq("id", user_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            profile_row = rows[0] if rows else {}
        except Exception as exc:
            logger.debug("user_risk: profile read failed: %s", exc)

    profile: Dict[str, Any] = {"risk_profile": profile_row.get("risk_profile")}
    explicit_limit = _num(profile_row.get("daily_loss_limit"))
    if explicit_limit is not None and explicit_limit > 0:
        profile["daily_loss_limit_pct"] = explicit_limit

    # ── today's realized P&L ──
    # paper: SELL event rows written by paper_executor (pnl set on sells)
    day_pnl = 0.0
    if sb is not None:
        try:
            rows = (
                sb.table("paper_trades")
                .select("pnl, action, executed_at")
                .eq("user_id", user_id)
                .eq("action", "sell")
                .gte("executed_at", today_iso)
                .limit(500)
                .execute()
                .data
                or []
            )
            for r in rows:
                p = _num(r.get("pnl"))
                if p is not None:
                    day_pnl += p
        except Exception as exc:
            logger.debug("user_risk: paper_trades read failed: %s", exc)

        # platform trades closed today (paper or live) — net_pnl is what
        # close_trade_record / the scheduler write on close.
        try:
            rows = (
                sb.table("trades")
                .select("net_pnl, closed_at, status")
                .eq("user_id", user_id)
                .eq("status", "closed")
                .gte("closed_at", today_iso)
                .limit(500)
                .execute()
                .data
                or []
            )
            for r in rows:
                p = _num(r.get("net_pnl"))
                if p is not None:
                    day_pnl += p
        except Exception as exc:
            logger.debug("user_risk: trades read failed: %s", exc)

    # ── open positions: paper_positions + platform positions ──
    positions: List[Dict[str, Any]] = []
    paper_value = 0.0
    paper_cash = 0.0
    has_platform_positions = False
    if sb is not None:
        try:
            rows = (
                sb.table("paper_portfolios")
                .select("cash")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            if rows:
                paper_cash = _num(rows[0].get("cash")) or 0.0
        except Exception as exc:
            logger.debug("user_risk: paper_portfolios read failed: %s", exc)

        # same read shape as paper_routes._open_positions
        try:
            rows = (
                sb.table("paper_positions")
                .select("symbol, qty, entry_price")
                .eq("user_id", user_id)
                .eq("status", "open")
                .limit(200)
                .execute()
                .data
                or []
            )
            for r in rows:
                sym = str(r.get("symbol") or "").strip().upper()
                qty = _num(r.get("qty")) or 0.0
                entry = _num(r.get("entry_price")) or 0.0
                value = qty * entry
                if sym and value > 0:
                    positions.append({"symbol": sym, "value": round(value, 2)})
                    paper_value += value
        except Exception as exc:
            logger.debug("user_risk: paper_positions read failed: %s", exc)

        # same read shape as /api/positions/open
        try:
            rows = (
                sb.table("positions")
                .select("symbol, quantity, current_price, average_price")
                .eq("user_id", user_id)
                .eq("is_active", True)
                .limit(100)
                .execute()
                .data
                or []
            )
            for r in rows:
                sym = str(r.get("symbol") or "").strip().upper()
                qty = _num(r.get("quantity")) or 0.0
                px = _num(r.get("current_price")) or _num(r.get("average_price")) or 0.0
                value = qty * px
                if sym and value > 0:
                    positions.append({"symbol": sym, "value": round(value, 2)})
                    has_platform_positions = True
        except Exception as exc:
            logger.debug("user_risk: positions read failed: %s", exc)

    # ── sector tags, best-effort (instruments table, same as why_moving) ──
    syms = sorted({p["symbol"] for p in positions})
    if sb is not None and syms:
        try:
            rows = (
                sb.table("instruments")
                .select("symbol, sector")
                .in_("symbol", syms)
                .eq("instrument_type", "EQ")
                .limit(len(syms))
                .execute()
                .data
                or []
            )
            sector_map = {
                str(r.get("symbol") or "").upper(): r.get("sector")
                for r in rows
                if r.get("sector")
            }
            for p in positions:
                sec = sector_map.get(p["symbol"])
                if sec:
                    p["sector"] = sec
        except Exception as exc:
            logger.debug("user_risk: instruments read failed: %s", exc)

    # ── capital base ──
    # Paper account → its own equity (cash + open value at entry marks);
    # platform positions are sized against user_profiles.capital, so add
    # it only when those exist. Fallback: profile capital alone.
    capital = 0.0
    if paper_cash > 0 or paper_value > 0:
        capital += paper_cash + paper_value
    profile_capital = _num(profile_row.get("capital"))
    if has_platform_positions and profile_capital:
        capital += profile_capital
    if capital <= 0 and profile_capital:
        capital = profile_capital

    result = check_risk(profile, day_pnl, capital, positions, None)
    result.update({
        "day_pnl": round(day_pnl, 2),
        "capital": round(capital, 2),
        "positions_value": round(sum(p["value"] for p in positions), 2),
        "positions_count": len(positions),
        "risk_profile": profile.get("risk_profile"),
        "daily_loss_limit_pct": effective_day_loss_limit_pct(profile),
    })
    return result
