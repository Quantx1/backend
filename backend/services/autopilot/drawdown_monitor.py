"""Drawdown alert emitter.

HIGH #5 (2026-05-31) — closes the audit gap "silence during drawdowns
kills retention faster than honesty." Runs daily after market close.
For each enrolled user, computes 30-day rolling drawdown from
`paper_snapshots` and fires the `portfolio_drawdown` alert event when
crossing thresholds (-5% warning · -10% serious · -15% critical).

Includes regime context in the alert body so users understand if the
drawdown is normal-for-the-regime or genuinely alarming.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


THRESHOLDS = [
    (-15.0, "critical", "🔴"),
    (-10.0, "serious", "🟠"),
    (-5.0, "warning", "🟡"),
]


def _regime_expected_dd_pct(regime: Optional[str]) -> str:
    """One-line expected DD band per regime for context."""
    r = (regime or "").lower()
    if r == "bull":
        return "0 to -5%"
    if r == "sideways":
        return "-5% to -10%"
    if r == "bear":
        return "-10% to -18%"
    return "regime unknown"


def emit_drawdown_alerts(supabase: Any, *, lookback_days: int = 30) -> Dict[str, Any]:
    """Daily entry point. Returns summary for cron logger."""
    out = {"users_checked": 0, "alerts_fired": 0, "errors": []}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    # Get latest regime once (system-wide)
    regime = None
    try:
        from ..regime.resolver import resolve_regime_at
        from datetime import date
        regime_row = resolve_regime_at(supabase, date.today())
        regime = (regime_row or {}).get("regime")
    except Exception:
        pass

    try:
        # `auto_trader_config` is a JSONB COLUMN on user_profiles, not a
        # separate table. Master on/off is `auto_trader_enabled` boolean
        # (per migration 2026_04_20_pr28_auto_trader_config.sql).
        users = (
            supabase.table("user_profiles")
            .select("id, auto_trader_config")
            .eq("auto_trader_enabled", True)
            .execute()
            .data or []
        )
    except Exception as e:
        out["errors"].append(f"user fetch failed: {e}")
        return out

    for u in users:
        uid = u.get("id")
        if not uid:
            continue
        out["users_checked"] += 1
        try:
            snaps = (
                supabase.table("paper_snapshots")
                .select("snapshot_date, equity, drawdown_pct")
                .eq("user_id", uid)
                .gte("snapshot_date", cutoff)
                .order("snapshot_date", desc=True)
                .limit(30)
                .execute()
                .data or []
            )
            if len(snaps) < 5:
                continue
            # Use the stored drawdown_pct if available; else recompute
            current_dd = snaps[0].get("drawdown_pct")
            if current_dd is None:
                equities = [float(s["equity"]) for s in reversed(snaps) if s.get("equity")]
                if not equities:
                    continue
                peak = max(equities)
                current_dd = (equities[-1] - peak) / peak * 100 if peak > 0 else 0.0
            current_dd = float(current_dd)

            # Find the worst threshold crossed
            crossed = None
            for thr, level, emoji in THRESHOLDS:
                if current_dd <= thr:
                    crossed = (thr, level, emoji)
                    break
            if crossed is None:
                continue

            thr, level, emoji = crossed
            expected = _regime_expected_dd_pct(regime)
            title = f"{emoji} AutoPilot drawdown alert · {level.upper()}"
            body = (
                f"Your portfolio is down {abs(current_dd):.1f}% over the last "
                f"{lookback_days} days. Current regime: {regime or 'unknown'} "
                f"(expected drawdown band: {expected}). "
                + (
                    "This is WITHIN expected bands — AutoPilot is holding course."
                    if level == "warning"
                    else "This is approaching the OUTER edge of expected drawdown for this regime."
                    if level == "serious"
                    else "This is BEYOND the expected drawdown band. Consider pausing AutoPilot to re-evaluate."
                )
            )
            # SCHEMA: type/message/data (verified via information_schema 2026-05-31)
            supabase.table("notifications").insert({
                "user_id": uid,
                "type": "portfolio_drawdown",
                "priority": level,    # 'warning' | 'serious' | 'critical'
                "title": title,
                "message": body,
                "channels": ["push", "telegram", "whatsapp", "email"],
                "data": {
                    "drawdown_pct": current_dd,
                    "level": level,
                    "regime": regime,
                    "lookback_days": lookback_days,
                    "url": "/autopilot/track-record",
                },
            }).execute()
            out["alerts_fired"] += 1
        except Exception as e:
            out["errors"].append(f"user {uid}: {str(e)[:200]}")
            continue

    return out
