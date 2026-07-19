"""
Signals API routes — read-only views over the ``signals`` table.

  GET /api/signals/today         today's active+triggered signals (7d fallback)
  GET /api/signals/intraday      last-N-minute intraday signals (Pro tier)
  GET /api/signals/{signal_id}   signal detail
  GET /api/signals/history       historical signals with filters
  GET /api/signals/performance   model_performance rollup over N days

Tier gating (PR-U 2026-05-28):
  * ``is_premium=false`` row filter still applies for Free visitors
    (kept for back-compat).
  * Free is now ALSO count-capped at 1 signal/day on /today + /history
    via current_user_tier — matches FEATURE_MATRIX signal_daily=FREE.
  * Pro and Elite are uncapped (signal_unlimited=PRO).
  * Admin bypass via UserTier.is_admin.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
import uuid as _uuid
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..core.tiers import Tier, UserTier
from ..middleware.tier_gate import current_user_tier

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Signals"])

# Free-tier daily signal count. Mirrors FEATURE_MATRIX["signal_daily"]
# = Tier.FREE in backend/core/tiers.py. Bumping here without
# updating that matrix will drift the gate; keep them in sync.
FREE_DAILY_SIGNAL_CAP: int = 1


def _apply_free_cap(signals: list, tier: UserTier, cap: int = FREE_DAILY_SIGNAL_CAP) -> list:
    """Truncate ``signals`` to ``cap`` for Free non-admins."""
    if tier.is_admin or tier.tier != Tier.FREE:
        return signals
    if not signals:
        return signals
    return signals[:cap]


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


def _get_user_profile_dep():
    from .app import get_user_profile
    return get_user_profile


def _get_supabase_retry():
    from .app import supabase_query_with_retry
    return supabase_query_with_retry


@router.get("/api/signals/today")
async def get_today_signals(
    segment: Optional[str] = None,
    direction: Optional[str] = None,
    profile=Depends(_get_user_profile_dep()),
    tier: UserTier = Depends(current_user_tier),
):
    """Today's trading signals (falls back to most recent 7 days if none today).

    Free users are capped to ``FREE_DAILY_SIGNAL_CAP`` rows (1/day) per
    FEATURE_MATRIX["signal_daily"]. Pro/Elite are uncapped.
    """
    today = date.today().isoformat()
    supabase_query_with_retry = _get_supabase_retry()

    def _fetch():
        sb = _get_supabase_admin()
        query = (
            sb.table("signals")
            .select("*")
            .eq("date", today)
            .in_("status", ["active", "triggered"])
        )
        if segment:
            query = query.eq("segment", segment)
        if direction:
            query = query.eq("direction", direction)
        is_premium = profile.get("subscription_status") in ["active", "trial"]
        if not is_premium:
            query = query.eq("is_premium", False)
        results = query.order("confidence", desc=True).execute().data

        # Fallback: if no signals today, show most recent signals (last 7 days)
        if not results:
            fallback_date = (date.today() - timedelta(days=7)).isoformat()
            fb_query = (
                sb.table("signals")
                .select("*")
                .gte("date", fallback_date)
                .in_("status", ["active", "triggered"])
            )
            if segment:
                fb_query = fb_query.eq("segment", segment)
            if direction:
                fb_query = fb_query.eq("direction", direction)
            if not is_premium:
                fb_query = fb_query.eq("is_premium", False)
            results = (
                fb_query.order("date", desc=True)
                .order("confidence", desc=True)
                .limit(50)
                .execute()
                .data
            )
        return results

    signals = await supabase_query_with_retry(_fetch, retries=2, timeout_fallback=[])
    capped = _apply_free_cap(signals, tier)

    return {
        "date": today,
        "total": len(capped),
        "long_signals": [s for s in capped if s.get("direction") == "LONG"],
        "short_signals": [s for s in capped if s.get("direction") == "SHORT"],
        "equity_signals": [s for s in capped if s.get("segment") == "EQUITY"],
        "futures_signals": [s for s in capped if s.get("segment") == "FUTURES"],
        "options_signals": [s for s in capped if s.get("segment") == "OPTIONS"],
        "all_signals": capped,
        "tier_cap_applied": tier.tier == Tier.FREE and not tier.is_admin and len(signals) > len(capped),
        "tier_cap": FREE_DAILY_SIGNAL_CAP if (tier.tier == Tier.FREE and not tier.is_admin) else None,
    }


@router.get("/api/signals/intraday")
async def get_intraday_signals(
    window_minutes: int = 60,
    profile=Depends(_get_user_profile_dep()),
):
    """PR 50 — F1 intraday signals (last N-minute window, Pro tier).

    Recent intraday signals (signal_type='intraday'). Fresh ones expire
    after 1 hour — the default window matches.
    """
    window_minutes = max(5, min(240, int(window_minutes)))
    cutoff = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()
    supabase_query_with_retry = _get_supabase_retry()

    def _fetch():
        sb = _get_supabase_admin()
        query = (
            sb.table("signals")
            .select("*")
            .eq("signal_type", "intraday")
            .gte("created_at", cutoff)
            .in_("status", ["active", "triggered"])
        )
        is_premium = profile.get("subscription_status") in ["active", "trial"]
        if not is_premium:
            # Non-Pro users get an empty list + upgrade hint; tier gate
            # at the frontend routes them to /pricing. Keep the route
            # unauthed-friendly at the API level though (no 402 here).
            return []
        return query.order("created_at", desc=True).limit(50).execute().data

    signals = await supabase_query_with_retry(_fetch, retries=2, timeout_fallback=[])
    return {
        "window_minutes": window_minutes,
        "total": len(signals),
        "signals": signals,
    }


@router.get("/api/signals/history")
async def get_signal_history(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    status: Optional[str] = None,
    segment: Optional[str] = None,
    direction: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    profile=Depends(_get_user_profile_dep()),
    tier: UserTier = Depends(current_user_tier),
):
    """Get historical signals with optional filters.

    Free users see the Free-cap row count (1 most-recent signal); Pro
    and Elite see up to ``limit`` rows per FEATURE_MATRIX.
    """
    try:
        supabase = _get_supabase_admin()
        query = supabase.table("signals").select("*")

        if from_date:
            query = query.gte("date", from_date)
        if to_date:
            query = query.lte("date", to_date)
        if status:
            query = query.eq("status", status)
        if segment:
            query = query.eq("segment", segment)
        if direction:
            query = query.eq("direction", direction)

        is_premium = profile.get("subscription_status") in ["active", "trial"]
        if not is_premium:
            query = query.eq("is_premium", False)

        # Apply Free count cap at the SQL level so we don't read more rows
        # than we'll return — both faster and cheaper on Supabase.
        effective_limit = limit
        if tier.tier == Tier.FREE and not tier.is_admin:
            effective_limit = min(limit, FREE_DAILY_SIGNAL_CAP)

        # Off the event loop — supabase-py is synchronous/blocking.
        result = await asyncio.to_thread(lambda: query.order("date", desc=True).limit(effective_limit).execute())
        return {
            "signals": result.data,
            "tier_cap_applied": effective_limit < limit,
            "tier_cap": FREE_DAILY_SIGNAL_CAP if (tier.tier == Tier.FREE and not tier.is_admin) else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/signals/performance")
async def get_signal_performance(days: int = 30, user=Depends(_get_current_user_dep())):
    """Get signal performance metrics from ``model_performance``."""
    try:
        supabase = _get_supabase_admin()
        start_date = (date.today() - timedelta(days=days)).isoformat()

        result = await asyncio.to_thread(
            lambda: supabase.table("model_performance")
            .select("*")
            .gte("date", start_date)
            .order("date", desc=True)
            .execute()
        )
        return {"performance": result.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Momentum engine (per-style serving) — 60s TTL cache (scanner pattern) ──
_momentum_cache: dict = {}  # key -> (ts, payload)
_MOMENTUM_TTL_S = 60


def _momentum_engine():
    from ..ai.signals.engines.momentum import MomentumEngine  # noqa: PLC0415
    return MomentumEngine()


def _compute_momentum(top_n: int = 20) -> dict:
    try:
        eng = _momentum_engine()
        sigs = eng.run(top_n=top_n)
        return {
            "signals": [s.to_dict() for s in sigs],
            "count": len(sigs),
            "status": eng.status,
            "style": "momentum",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"momentum engine failed: {exc}")


@router.get("/api/signals/momentum")
async def get_momentum_signals(
    top_n: int = Query(20, ge=1, le=100),
    profile=Depends(_get_user_profile_dep()),
    tier: UserTier = Depends(current_user_tier),
):
    """Momentum ranker — top-of-book by expected forward return (spec §5.1).
    On-demand with a 60s in-process cache (no persistence)."""
    key = f"momentum:{top_n}"
    now = _time.time()
    hit = _momentum_cache.get(key)
    if hit and now - hit[0] < _MOMENTUM_TTL_S:
        return hit[1]
    payload = _compute_momentum(top_n=top_n)
    _momentum_cache[key] = (now, payload)
    if len(_momentum_cache) > 200:
        oldest = min(_momentum_cache.items(), key=lambda kv: kv[1][0])[0]
        _momentum_cache.pop(oldest, None)
    return payload


# ── Swing engine (per-style serving) — 60s TTL cache (scanner pattern) ──
_swing_cache: dict = {}  # key -> (ts, payload)
_SWING_TTL_S = 60


def _swing_engine():
    from ..ai.signals.engines.swing import SwingEngine  # noqa: PLC0415
    return SwingEngine()


def _compute_swing(top_n: int = 20) -> dict:
    try:
        eng = _swing_engine()
        sigs = eng.run(top_n=top_n)
        return {
            "signals": [s.to_dict() for s in sigs],
            "count": len(sigs),
            "status": eng.status,
            "style": "swing",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"swing engine failed: {exc}")


@router.get("/api/signals/swing")
async def get_swing_signals(
    top_n: int = Query(20, ge=1, le=100),
    profile=Depends(_get_user_profile_dep()),
    tier: UserTier = Depends(current_user_tier),
):
    """Swing ranker — top-of-book by expected 10-day forward return (spec §5.1).
    On-demand with a 60s in-process cache (no persistence)."""
    key = f"swing:{top_n}"
    now = _time.time()
    hit = _swing_cache.get(key)
    if hit and now - hit[0] < _SWING_TTL_S:
        return hit[1]
    payload = _compute_swing(top_n=top_n)
    _swing_cache[key] = (now, payload)
    if len(_swing_cache) > 200:
        oldest = min(_swing_cache.items(), key=lambda kv: kv[1][0])[0]
        _swing_cache.pop(oldest, None)
    return payload


# ── Paper window — live matured outcomes vs frozen backtest expectations ──
# Registered BEFORE the dynamic /{signal_id} route: static paths must win.
_paper_window_cache: dict = {}  # key -> (ts, payload)
_PAPER_WINDOW_TTL_S = 60
_BASELINE_EXPECTATIONS: Optional[dict] = None  # module-level parse cache
_EXPECTED_SOURCE = "backtest 2023-07..2026-06"


def _baseline_expectations() -> dict:
    """Parse data/paper/baseline_expectations.json once per process.

    Pre-registered + frozen before the window started — never recomputed
    from live tables. Unreadable file degrades to honest-empty expectations.
    """
    global _BASELINE_EXPECTATIONS
    if _BASELINE_EXPECTATIONS is None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        path = Path(__file__).resolve().parents[2] / "data" / "paper" / "baseline_expectations.json"
        try:
            _BASELINE_EXPECTATIONS = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 — honest-empty expectations
            logger.warning("baseline expectations unreadable (%s): %s", path, exc)
            _BASELINE_EXPECTATIONS = {"engines": {}}
    return _BASELINE_EXPECTATIONS


def _compute_paper_window() -> dict:
    """Aggregate style_signal_outcomes per (engine, trade_date) and compare
    against the frozen baseline. Honest-empty: no tables/rows => days 0,
    live nulls, status 'collecting'."""
    from math import sqrt  # noqa: PLC0415
    from ..ai.signals.style_persistence import fetch_outcomes, fetch_signal_dates  # noqa: PLC0415
    from ..platform.scheduler import STYLE_HORIZONS  # noqa: PLC0415 — the single horizon map

    baseline = (_baseline_expectations().get("engines") or {})
    engines_payload: dict = {}
    window_start = None

    for engine, horizon in STYLE_HORIZONS.items():
        sig_dates = fetch_signal_dates(engine)
        if sig_dates:
            first = min(sig_dates)
            if window_start is None or first < window_start:
                window_start = first

        # Per-date aggregation: date_gross = mean of the book's fwd returns,
        # date_bench = the (per-date constant) equal-weight universe return.
        per_date: dict = {}
        for r in fetch_outcomes(engine):
            d = str(r.get("trade_date"))[:10]
            slot = per_date.setdefault(d, {"fwd": [], "bench": None})
            if r.get("fwd_return_h") is not None:
                slot["fwd"].append(float(r["fwd_return_h"]))
            if r.get("bench_fwd_return_h") is not None:
                slot["bench"] = float(r["bench_fwd_return_h"])
        date_stats = [
            (sum(v["fwd"]) / len(v["fwd"]), v["bench"])
            for v in per_date.values()
            if v["fwd"] and v["bench"] is not None
        ]
        m = len(date_stats)

        live: dict = {"hit_rate": None, "mean_excess_h": None,
                      "mean_gross_h": None, "n_dates": m}
        if m:
            live["hit_rate"] = round(sum(1.0 for g, b in date_stats if g > b) / m, 4)
            live["mean_excess_h"] = round(sum(g - b for g, b in date_stats) / m, 6)
            live["mean_gross_h"] = round(sum(g for g, _ in date_stats) / m, 6)

        exp = baseline.get(engine) or {}
        expected = {
            "hit_rate": exp.get("hit_rate_vs_universe"),
            "mean_excess_h": exp.get("mean_excess_h"),
            "source": _EXPECTED_SOURCE,
        }

        # Status: collecting until 10 matured dates, then a 2-sigma binomial
        # lower bound around the pre-registered hit rate.
        status = "collecting"
        p = expected["hit_rate"]
        if m >= 10 and p is not None and live["hit_rate"] is not None:
            bound = p - 2.0 * sqrt(p * (1.0 - p) / m)
            status = "on_track" if live["hit_rate"] >= bound else "off_track"

        engines_payload[engine] = {
            "horizon": horizon,
            "days_signaled": len(sig_dates),
            "days_matured": m,
            "live": live,
            "expected": expected,
            "status": status,
        }

    return {
        "window_start": window_start.isoformat() if window_start else None,
        "as_of": date.today().isoformat(),
        "engines": engines_payload,
    }


@router.get("/api/signals/style/paper-window")
async def get_style_paper_window(
    profile=Depends(_get_user_profile_dep()),
    tier: UserTier = Depends(current_user_tier),
):
    """Paper-window scoreboard — matured live outcomes for the style engines
    vs the frozen pre-registered backtest expectations. All tiers (trust
    surface, no gate); 60s in-process cache like /momentum."""
    now = _time.time()
    hit = _paper_window_cache.get("paper-window")
    if hit and now - hit[0] < _PAPER_WINDOW_TTL_S:
        return hit[1]
    # Off the event loop — supabase-py is synchronous/blocking.
    payload = await asyncio.to_thread(_compute_paper_window)
    _paper_window_cache["paper-window"] = (now, payload)
    return payload


# Dynamic-path route — MUST be registered AFTER all static-path routes
# (/today, /intraday, /history, /performance, /style/paper-window) so
# FastAPI's longest-prefix match doesn't capture e.g. "history" as a
# signal_id.
@router.get("/api/signals/{signal_id}")
async def get_signal(signal_id: str, profile=Depends(_get_user_profile_dep())):
    """Get signal details by id.

    Tier-gated to match the list endpoints (``/today``, ``/history``):
    free users only see ``is_premium=False`` rows. Without this gate
    a free user with a premium signal's UUID (shared via screenshot,
    social, copy-pasted URL) could read the row directly even though
    it never appears in their lists.
    """
    # Guard malformed ids: the ``id`` column is a UUID, so a non-UUID path
    # segment (e.g. "demo-signal-id") makes Postgres raise on a bad cast. We
    # 404 fast here *before* touching the DB so a junk URL can never stall the
    # request (and, since the supabase call below is synchronous, the whole
    # event loop). Same 404 shape as a missing row — no existence leak.
    try:
        _uuid.UUID(str(signal_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Signal not found")

    try:
        supabase = _get_supabase_admin()
        # Run the blocking supabase HTTP call off the event loop so one slow
        # query can't wedge every other request on this worker.
        result = await asyncio.to_thread(
            lambda: supabase.table("signals").select("*").eq("id", signal_id).single().execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Signal not found")
        is_premium_user = profile.get("subscription_status") in ["active", "trial"]
        if result.data.get("is_premium") and not is_premium_user:
            # 404 (not 403) so existence isn't leaked to free users —
            # they can't enumerate premium UUIDs by probing this route.
            raise HTTPException(status_code=404, detail="Signal not found")
        return result.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
