"""
Admin observability endpoints — audit-log + A/B experiment summaries.

  GET /admin/audit-log            browse admin_audit_log with filters
  GET /admin/experiments/summary  per-variant exposure + conversion (last 30d)

Both are read-only reporting surfaces. The audit-log captures every
admin mutation (suspends, bans, kill-switch flips, etc.) and is the
source of truth for "who did what and when" reviews. The experiments
summary feeds the admin command-center copy-test tile.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from ._deps import AdminUser, get_admin_user, get_supabase_admin

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# AUDIT LOG (PR 49)
# ============================================================================


@router.get("/audit-log")
async def list_audit_log(
    actor_id: Optional[str] = Query(None, description="Filter by admin user_id"),
    action: Optional[str] = Query(None, description="Filter by action (e.g. 'user_ban')"),
    target_type: Optional[str] = Query(None),
    target_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    admin: AdminUser = Depends(get_admin_user),
):
    """Browse the admin_audit_log with optional filters. Most recent first."""
    client = get_supabase_admin()
    try:
        q = (
            client.table("admin_audit_log")
            .select(
                "id, actor_id, actor_email, action, target_type, target_id, "
                "payload, ip_address, user_agent, created_at"
            )
            .order("created_at", desc=True)
            .limit(limit)
        )
        if actor_id:
            q = q.eq("actor_id", actor_id)
        if action:
            q = q.eq("action", action)
        if target_type:
            q = q.eq("target_type", target_type)
        if target_id:
            q = q.eq("target_id", target_id)
        resp = q.execute()
        rows = resp.data or []
    except Exception as exc:
        logger.warning("audit-log query failed: %s", exc)
        rows = []

    # Distinct-action facet so the UI can populate its filter dropdown.
    actions_seen = sorted({r.get("action") for r in rows if r.get("action")})
    return {
        "rows": rows,
        "count": len(rows),
        "actions": actions_seen,
        "computed_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# A/B EXPERIMENT SUMMARY (PR 148)
# ============================================================================
#
# Joins EXPERIMENT_EXPOSED (denominator) with UPGRADE_INITIATED filtered
# to ``source = quiz_rec_what_changes`` (numerator) per variant. Used by
# the admin command-center to monitor whether the feature_led vs
# outcome_led copy is performing.


@router.get("/experiments/summary")
async def experiments_summary(admin: AdminUser = Depends(get_admin_user)):
    """Per-variant exposure + conversion counts over the last 30 days."""
    sb = get_supabase_admin()
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    out: List[Dict[str, Any]] = []

    # PostHog events are mirrored to Supabase ``analytics_events`` via
    # the observability layer (PR 96). The schema there: event TEXT,
    # properties JSONB, ts TIMESTAMPTZ, user_id UUID NULL.
    try:
        exposures = (
            sb.table("analytics_events")
            .select("event, properties, ts")
            .eq("event", "experiment_exposed")
            .gte("ts", cutoff)
            .limit(50_000)
            .execute()
        )
        upgrades = (
            sb.table("analytics_events")
            .select("event, properties, ts")
            .eq("event", "upgrade_initiated")
            .gte("ts", cutoff)
            .limit(50_000)
            .execute()
        )
    except Exception as exc:
        logger.warning("analytics_events query failed: %s", exc)
        return {"experiments": [], "computed_at": datetime.utcnow().isoformat()}

    # Roll up by (experiment, variant).
    counters: Dict[tuple, Dict[str, int]] = {}
    for r in exposures.data or []:
        p = r.get("properties") or {}
        key = (p.get("experiment"), p.get("experiment_variant"))
        if not key[0] or not key[1]:
            continue
        counters.setdefault(key, {"exposed": 0, "converted": 0})["exposed"] += 1
    for r in upgrades.data or []:
        p = r.get("properties") or {}
        if p.get("source") != "quiz_rec_what_changes":
            continue
        v = p.get("experiment_variant")
        if not v:
            continue
        key = ("quiz_rec_delta_copy", v)
        counters.setdefault(key, {"exposed": 0, "converted": 0})["converted"] += 1

    for (exp, variant), c in counters.items():
        rate = (c["converted"] / c["exposed"]) if c["exposed"] else 0.0
        out.append({
            "experiment": exp,
            "variant": variant,
            "exposed": c["exposed"],
            "converted": c["converted"],
            "conversion_rate": round(rate, 4),
        })

    return {
        "experiments": sorted(out, key=lambda r: (r["experiment"], r["variant"])),
        "computed_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# LLM COST DASHBOARD (PR-V)
# ============================================================================


@router.get("/llm-cost")
async def llm_cost_summary(
    hours: int = Query(24, ge=1, le=720, description="Lookback window."),
    admin: AdminUser = Depends(get_admin_user),
):
    """Per-feature + top-spender LLM cost rollup over the last ``hours``.

    Reads from ``llm_usage_events`` (PR-V migration). Each row carries
    a precomputed ``micros_usd`` from ``observability.llm_pricing``, so
    the totals never re-tokenize. Returns:

      * by_feature : [{feature, calls, input_tokens, output_tokens, usd}]
      * by_user    : top-10 by spend, [{user_id, calls, usd}]
      * by_model   : [{provider, model, calls, usd}]
      * total      : {calls, usd, window_hours, computed_at}
    """
    sb = get_supabase_admin()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    try:
        rows = (
            sb.table("llm_usage_events")
            .select(
                "user_id, feature, provider, model, input_tokens, "
                "output_tokens, micros_usd, ts",
            )
            .gte("ts", cutoff)
            .order("ts", desc=True)
            .limit(50_000)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.warning("llm_usage_events query failed: %s", exc)
        return {
            "by_feature": [],
            "by_user": [],
            "by_model": [],
            "total": {
                "calls": 0,
                "usd": 0.0,
                "window_hours": hours,
                "computed_at": datetime.utcnow().isoformat(),
                "error": str(exc),
            },
        }

    by_feature: Dict[str, Dict[str, int]] = {}
    by_user: Dict[str, Dict[str, int]] = {}
    by_model: Dict[tuple, Dict[str, int]] = {}
    total_calls = 0
    total_micros = 0

    for r in rows:
        feature = r.get("feature") or "unknown"
        user_id = r.get("user_id") or "anonymous"
        provider = r.get("provider") or "?"
        model = r.get("model") or "?"
        micros = int(r.get("micros_usd") or 0)
        in_t = int(r.get("input_tokens") or 0)
        out_t = int(r.get("output_tokens") or 0)

        f = by_feature.setdefault(
            feature,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "micros": 0},
        )
        f["calls"] += 1
        f["input_tokens"] += in_t
        f["output_tokens"] += out_t
        f["micros"] += micros

        u = by_user.setdefault(user_id, {"calls": 0, "micros": 0})
        u["calls"] += 1
        u["micros"] += micros

        mkey = (provider, model)
        m = by_model.setdefault(mkey, {"calls": 0, "micros": 0})
        m["calls"] += 1
        m["micros"] += micros

        total_calls += 1
        total_micros += micros

    def _usd(micros: int) -> float:
        return round(micros / 1_000_000, 4)

    return {
        "by_feature": [
            {
                "feature": k,
                "calls": v["calls"],
                "input_tokens": v["input_tokens"],
                "output_tokens": v["output_tokens"],
                "usd": _usd(v["micros"]),
            }
            for k, v in sorted(by_feature.items(), key=lambda x: -x[1]["micros"])
        ],
        "by_user": [
            {"user_id": k, "calls": v["calls"], "usd": _usd(v["micros"])}
            for k, v in sorted(by_user.items(), key=lambda x: -x[1]["micros"])[:10]
        ],
        "by_model": [
            {"provider": k[0], "model": k[1], "calls": v["calls"], "usd": _usd(v["micros"])}
            for k, v in sorted(by_model.items(), key=lambda x: -x[1]["micros"])
        ],
        "total": {
            "calls": total_calls,
            "usd": _usd(total_micros),
            "window_hours": hours,
            "computed_at": datetime.utcnow().isoformat(),
        },
    }
