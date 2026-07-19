"""
Public trust-surface endpoints (Step 4 §5.1).

Three routes, all unauthenticated + RLS-safe:

    GET /api/public/regime/history      — /regime page timeline
    GET /api/public/track-record        — /track-record signals list + stats
    GET /api/public/models              — /models page (per-model WR cards)

These power the "before-signup" trust pages. They must stay cheap +
cacheable — no heavy computation, just reads from already-aggregated
tables (``regime_history``, ``signals``, ``model_rolling_performance``).

Cache headers hint 60s freshness on CDN/Vercel edge.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..core.database import get_supabase_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])


def _with_cache(payload: Dict[str, Any], *, max_age: int = 60) -> JSONResponse:
    """Attach CDN cache-control headers. Public endpoints are CDN-safe."""
    return JSONResponse(
        content=payload,
        headers={
            "Cache-Control": f"public, max-age={max_age}, s-maxage={max_age}, stale-while-revalidate=120",
        },
    )


# ============================================================================
# /regime — timeline + current state (F8 public surface)
# ============================================================================


@router.get("/regime/history")
async def regime_history(
    days: int = Query(90, ge=1, le=365),
) -> JSONResponse:
    """Return the regime_history rows + current-regime snapshot.

    Consumers:
      - /regime public page
      - RegimeBanner (client-polls this when WS hasn't pushed regime_change yet)
    """
    client = get_supabase_admin()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    try:
        history_resp = (
            client.table("regime_history")
            .select(
                "regime, prob_bull, prob_sideways, prob_bear, vix, nifty_close, "
                "persistence_prob, detected_at"
            )
            .gte("detected_at", cutoff)
            .order("detected_at", desc=False)
            .execute()
        )
        history = history_resp.data or []
    except Exception as exc:
        logger.warning("regime_history query failed: %s", exc)
        history = []

    current = history[-1] if history else None

    # Count days each regime was active in the window (for a small donut chart).
    bucket_counts = {"bull": 0, "sideways": 0, "bear": 0}
    for row in history:
        r = row.get("regime")
        if r in bucket_counts:
            bucket_counts[r] += 1

    return _with_cache(
        {
            "days": days,
            "current": current,
            "history": history,
            "counts": bucket_counts,
            "computed_at": datetime.utcnow().isoformat() + "Z",
        },
        max_age=60,
    )


# ============================================================================
# /track-record — closed signals with real P&L (N3 trust surface)
# ============================================================================

_CLOSED_STATUSES = ("target_hit", "stop_loss_hit", "sl_hit", "expired")


@router.get("/track-record")
async def track_record(
    days: int = Query(90, ge=7, le=365),
    segment: Optional[str] = Query(None, description="EQUITY / FUTURES / OPTIONS"),
    direction: Optional[str] = Query(None, description="LONG / SHORT"),
    limit: int = Query(200, ge=1, le=1000),
) -> JSONResponse:
    """Closed signals + aggregated stats. No hidden drawers — wins AND
    losses, realized P&L, holding days, which models concurred.
    """
    client = get_supabase_admin()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    query = (
        client.table("signals")
        .select(
            "id, symbol, direction, segment, entry_price, target_1, stop_loss, "
            "status, generated_at, date, confidence, strategy_names, "
            "tft_p50, lgbm_buy_prob, qlib_rank, regime_at_signal, model_agreement"
        )
        .gte("date", cutoff)
        .in_("status", list(_CLOSED_STATUSES))
    )
    if segment:
        query = query.eq("segment", segment)
    if direction:
        query = query.eq("direction", direction)
    try:
        resp = query.order("date", desc=True).limit(limit).execute()
        rows = resp.data or []
    except Exception as exc:
        logger.warning("track_record query failed: %s", exc)
        rows = []

    # Enrich each row with realized return %, result bucket, and the
    # public engine keys that concurred on the signal. Engine inference
    # uses only which internal columns are populated — we never leak
    # the underlying architecture, just the brand name that maps to it.
    enriched: List[Dict[str, Any]] = []
    wins = 0
    losses = 0
    expired_count = 0
    win_returns: List[float] = []
    loss_returns: List[float] = []
    all_returns: List[float] = []
    for r in rows:
        entry = float(r.get("entry_price") or 0)
        if entry <= 0:
            continue
        if r["status"] == "target_hit":
            exit_price = float(r.get("target_1") or entry)
            result = "target"
            wins += 1
        elif r["status"] in ("stop_loss_hit", "sl_hit"):
            exit_price = float(r.get("stop_loss") or entry)
            result = "stop"
            losses += 1
        else:
            exit_price = entry
            result = "expired"
            expired_count += 1

        ret_pct = ((exit_price - entry) / entry) * 100
        if r["direction"] == "SHORT":
            ret_pct = -ret_pct
        all_returns.append(ret_pct)
        if result == "target":
            win_returns.append(ret_pct)
        elif result == "stop":
            loss_returns.append(ret_pct)

        engines: List[str] = []
        if r.get("tft_p50") is not None:
            engines.append("swing_forecast")
        if r.get("lgbm_buy_prob") is not None:
            engines.append("signal_gate")
        if r.get("qlib_rank") is not None:
            engines.append("cross_sectional_ranker")
        if r.get("regime_at_signal"):
            engines.append("regime_detector")

        enriched.append({
            **r,
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret_pct, 2),
            "result": result,
            "engines": engines,
        })

    n = len(enriched)
    win_rate = (wins / n) if n else 0.0
    avg_return = (sum(all_returns) / n) if all_returns else 0.0
    avg_win = (sum(win_returns) / len(win_returns)) if win_returns else 0.0
    avg_loss = (sum(loss_returns) / len(loss_returns)) if loss_returns else 0.0
    gross_profit = sum(win_returns)
    gross_loss = abs(sum(loss_returns))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None  # infinite — render as "—" on the client
    else:
        profit_factor = 0.0
    best = max(enriched, key=lambda x: x.get("return_pct", 0), default=None)
    worst = min(enriched, key=lambda x: x.get("return_pct", 0), default=None)

    # Daily cumulative-return curve (oldest → newest) for the chart.
    curve: List[Dict[str, Any]] = []
    running = 0.0
    for row in sorted(enriched, key=lambda x: x.get("date") or ""):
        running += float(row.get("return_pct") or 0)
        curve.append({
            "date": row.get("date"),
            "cum_return_pct": round(running, 2),
        })

    # Current regime for top banner.
    try:
        regime_row = (
            client.table("regime_history")
            .select("regime, detected_at")
            .order("detected_at", desc=True).limit(1).execute()
        )
        current_regime = (regime_row.data or [None])[0]
    except Exception:
        current_regime = None

    return _with_cache({
        "days": days,
        "stats": {
            "n": n,
            "wins": wins,
            "losses": losses,
            "expired": expired_count,
            "win_rate": round(win_rate, 4),
            "avg_return_pct": round(avg_return, 4),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
            "best_return_pct": round(best.get("return_pct", 0), 2) if best else 0,
            "best_symbol": best.get("symbol") if best else None,
            "worst_return_pct": round(worst.get("return_pct", 0), 2) if worst else 0,
            "worst_symbol": worst.get("symbol") if worst else None,
        },
        "curve": curve,
        "current_regime": current_regime,
        "signals": enriched,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }, max_age=120)


# ============================================================================
# /models — per-model accuracy dashboard (N4 trust surface)
# ============================================================================

# BRAND FIREWALL (project_greek_branding_2026_04_19 + SEBI): internal model
# codenames (tft_swing / qlib_alpha158 / finbert_india / regime_hmm / …) must
# NEVER leave the backend in a public payload. This map mirrors the public
# engine identities in frontend/lib/engines.ts + ModelAccuracyCard so the
# public /models response carries only display names, never raw identifiers.
_PUBLIC_ENGINES: Dict[str, Dict[str, Any]] = {
    "tft_swing": {
        "name": "Forecast",
        "color": "#4FECCD",
        "description": "Swing forecast engine — 5-day quantile outlook with p10/p50/p90 bands.",
        "is_prod": True,
    },
    "qlib_alpha158": {
        "name": "Alpha",
        "color": "#5DCBD8",
        "description": "Cross-sectional alpha ranker — nightly sieve across NSE.",
        "is_prod": True,
    },
    "lgbm_signal_gate": {
        "name": "Gate",
        "color": "#FFD166",
        "description": "3-class signal gate — buy / hold / sell verdict on every candidate.",
        "is_prod": True,
    },
    "regime_hmm": {
        "name": "Regime",
        "color": "#FF9900",
        "description": "Market regime detector — bull / sideways / bear with daily updates.",
        "is_prod": True,
    },
    "finbert_india": {
        "name": "Mood",
        "color": "#05B878",
        "description": "News sentiment engine — NSE-tuned for financial text.",
        "is_prod": True,
    },
    "strategy": {
        "name": "Strategy confluence",
        "color": "#4FECCD",
        "description": "Rule-based strategy vote (breakouts, pullbacks, reversals, structure, volume).",
        "is_prod": False,
    },
    "breakout_meta_labeler": {
        "name": "PatternScope",
        "color": "#00E5CC",
        "description": "Pattern quality scorer — Scanner Lab tag.",
        "is_prod": False,
    },
}


def _public_engine(raw: Optional[str]) -> Dict[str, Any]:
    """Map an internal model codename → its public engine identity.

    Unknown/new/stale codenames collapse to a neutral "Engine" label so a raw
    identifier can never leak through the public payload.
    """
    return _PUBLIC_ENGINES.get(raw or "", {
        "name": "Engine",
        "color": "#8E8E8E",
        "description": "",
        "is_prod": False,
    })


@router.get("/models")
async def models_dashboard(
    window_days: int = Query(30, ge=7, le=365),
) -> JSONResponse:
    """Per-model rolling performance from ``model_rolling_performance``.

    The weekly ``aggregate_model_rolling_performance`` scheduler job (PR 7)
    writes these rows. Here we just sort + return them + add the sparkline
    series per model so the frontend renders each card with a 7-day trend
    line.
    """
    client = get_supabase_admin()

    try:
        resp = (
            client.table("model_rolling_performance")
            .select(
                "model_name, window_days, win_rate, avg_pnl_pct, signal_count, "
                "directional_accuracy, sharpe_ratio, max_drawdown_pct, computed_at"
            )
            .eq("window_days", window_days)
            .order("computed_at", desc=True)
            .limit(500)
            .execute()
        )
        rows = resp.data or []
    except Exception as exc:
        logger.warning("models_dashboard query failed: %s", exc)
        rows = []

    # Group by model → latest row + sparkline (last N win_rate values).
    by_model: Dict[str, Dict[str, Any]] = {}
    sparklines: Dict[str, List[float]] = {}
    for row in rows:
        name = row["model_name"]
        if name not in by_model:
            by_model[name] = row
        sparklines.setdefault(name, []).append(float(row.get("win_rate") or 0))

    # Each sparkline newest-last (flip to oldest-first for rendering).
    models = []
    for name, latest in by_model.items():
        models.append({
            **latest,
            "source": "rolling",
            "sparkline": list(reversed(sparklines.get(name, [])))[-30:],
        })

    # Fallback: no closed signals yet → surface training-time metrics from
    # model_versions so /models still shows live engines instead of an
    # empty state. The frontend renders these with a "calibration" badge.
    if not models:
        try:
            vresp = (
                client.table("model_versions")
                .select("model_name, version, is_prod, is_shadow, metrics, trained_at")
                .eq("is_prod", True)
                .eq("is_retired", False)
                .order("trained_at", desc=True)
                .limit(50)
                .execute()
            )
            # Public-friendly labels for primary metrics so the UI doesn't
            # have to know about rank_ic / log_likelihood / wape internals.
            METRIC_LABEL = {
                "directional_accuracy": "Directional accuracy",
                "rank_ic_mean": "Rank IC",
                "log_likelihood_per_obs_mean": "Log-likelihood",
                "wape": "WAPE",
                "sharpe": "Sharpe",
                "pinball_loss_mean": "Pinball loss",
                "median_mae": "MAE",
            }
            for row in vresp.data or []:
                m = row.get("metrics") or {}
                # Map the trainer's primary metric → win_rate proxy when
                # nothing better exists. Different trainers emit different
                # primaries (dir_acc, sharpe, loglik), so coerce safely.
                dir_acc = m.get("directional_accuracy") or m.get("dir_acc")
                wr_proxy = float(dir_acc) if isinstance(dir_acc, (int, float)) else None
                n_signal = (
                    m.get("n_walk_forward_windows")
                    or m.get("n_calibrations")
                    or m.get("n_trades")
                    or m.get("n_total_obs")
                    or 0
                )
                primary_metric = m.get("primary_metric")
                primary_value = m.get("primary_value")
                models.append({
                    "model_name": row["model_name"],
                    "window_days": window_days,
                    "win_rate": wr_proxy,
                    "signal_count": int(n_signal) if n_signal else 0,
                    "avg_pnl_pct": None,
                    "directional_accuracy": wr_proxy,
                    "sharpe_ratio": (
                        float(m["sharpe"]) if isinstance(m.get("sharpe"), (int, float)) else None
                    ),
                    "max_drawdown_pct": (
                        float(m["max_drawdown_pct"])
                        if isinstance(m.get("max_drawdown_pct"), (int, float))
                        else None
                    ),
                    # Surface the trainer's primary metric + a friendly label
                    # so qlib/regime_hmm/tft_swing (which don't have a
                    # natural "win_rate") still render something meaningful.
                    "primary_metric": primary_metric,
                    "primary_value": (
                        float(primary_value)
                        if isinstance(primary_value, (int, float))
                        else None
                    ),
                    "primary_label": METRIC_LABEL.get(primary_metric, primary_metric),
                    "computed_at": row.get("trained_at"),
                    "source": "calibration",
                    "version": row.get("version"),
                    "is_prod": bool(row.get("is_prod")),
                    "is_shadow": bool(row.get("is_shadow")),
                    "sparkline": [],
                })
        except Exception as exc:
            logger.warning("model_versions fallback failed: %s", exc)

    # Stable ordering: the v1 models customers see first. The dropped
    # entries (momentum_chronos / options_rl / vix_tft / chronos2_macro)
    # are intentionally absent — if a stale row appears in model_versions
    # the sort.index() fallback (999) lands it at the bottom.
    # 2026-05-23 — RL legs (finrl_x_*) removed from v1 dashboard.
    # See ml/training/discovery.py _SKIP_MODULES.
    ORDER = [
        "tft_swing", "qlib_alpha158", "lgbm_signal_gate", "regime_hmm",
        "strategy", "breakout_meta_labeler", "finbert_india",
    ]
    models.sort(key=lambda m: ORDER.index(m["model_name"]) if m["model_name"] in ORDER else 999)

    # Brand firewall: swap the internal codename for its public engine identity
    # before the payload leaves the process. `model_name` is retained (the
    # frontend keys cards on it) but populated with the PUBLIC display name so
    # no raw identifier (tft_swing / qlib_alpha158 / finbert_india / …) ships
    # to the browser. Sorting above still used the raw name.
    public_models: List[Dict[str, Any]] = []
    for m in models:
        eng = _public_engine(m.pop("model_name", None))
        m["model_name"] = eng["name"]
        m["engine"] = eng["name"]
        m["engine_color"] = eng["color"]
        m["engine_description"] = eng["description"]
        if m.get("is_prod") is None:
            m["is_prod"] = eng["is_prod"]
        m.setdefault("is_shadow", False)
        public_models.append(m)
    models = public_models

    return _with_cache({
        "window_days": window_days,
        "models": models,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }, max_age=300)


# ============================================================================
# PR 48 — Global kill-switch status (public, unauth)
# ============================================================================
# Lightweight read so the platform layout can render an ops-halt banner
# across every authenticated page without admin creds. No reason text
# leaks — we only expose active/inactive to keep internal ops details
# staff-only.


@router.get("/system/status")
async def get_system_status():
    """Public ops status — currently just the platform-wide trading halt
    flag. No admin reason text is exposed; only the boolean."""
    try:
        from ..platform.system_flags import is_globally_halted
        halted = is_globally_halted()
    except Exception:
        halted = False
    return _with_cache({
        "trading_halted": bool(halted),
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }, max_age=30)


@router.get("/models/status")
async def get_models_status():
    """PR 52 — public model-availability status so the frontend can hide
    features whose models aren't ready yet (per the no-fallback rule).
    Only booleans are returned — no internal architecture names, no
    artifact paths.

    Each key maps 1:1 to a public engine brand name. Frontend looks up
    the key and shows a "coming soon" state when false.
    """
    status: Dict[str, bool] = {}
    return _with_cache({
        "models": status,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }, max_age=30)


# ============================================================================
# /indices — live-ish ticker strip for /dashboard + /stocks (PR 66)
# ============================================================================
# Cheap polled endpoint backing the IndexTickerStrip. We hit the live
# screener engine's `_fetch_index_df` (Kite-first, yfinance fallback)
# for each tracked index and return last close + day change. The 30s
# cache bound matches the frontend SWR poll, so a stampede of clients
# hits one upstream call per half-minute.
#
# Public-safe: no per-user data, no auth required. The same data is
# already visible on every NSE site.

_TICKER_INDICES = [
    {"key": "nifty", "label": "NIFTY 50", "td": "NIFTY 50", "yf": "^NSEI"},
    {"key": "banknifty", "label": "BANK NIFTY", "td": "NIFTY BANK", "yf": "^NSEBANK"},
    {"key": "sensex", "label": "SENSEX", "td": "SENSEX", "yf": "^BSESN"},
    {"key": "vix", "label": "INDIA VIX", "td": "INDIA VIX", "yf": "^INDIAVIX"},
]


def _index_quote(td: str, yf: str) -> Optional[Dict[str, Any]]:
    """Pull the last two daily closes and compute the day change. Falls
    back across providers via the screener engine's existing helper."""
    try:
        from ..data.screener.engine import LiveScreenerEngine
        engine = LiveScreenerEngine()
        df = engine._fetch_index_df(td, yf, period="5d")
        if df is None or df.empty or "close" not in df.columns:
            return None
        closes = df["close"].dropna().tail(2)
        if len(closes) < 1:
            return None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
        change = last - prev
        change_pct = (change / prev * 100.0) if prev else 0.0
        return {
            "last": round(last, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
        }
    except Exception as exc:
        logger.debug("index quote fetch failed td=%s yf=%s: %s", td, yf, exc)
        return None


@router.get("/indices")
async def get_indices():
    """Last close + day change for the four indices that matter to a
    swing-trader on day one: NIFTY, BANKNIFTY, SENSEX, INDIA VIX. Each
    entry is `{key, label, last, change, change_pct}` or `null` if the
    upstream is unavailable. Frontend renders the row regardless and
    shows a `--` for null cells."""
    out: List[Dict[str, Any]] = []
    for idx in _TICKER_INDICES:
        quote = _index_quote(idx["td"], idx["yf"])
        out.append({
            "key": idx["key"],
            "label": idx["label"],
            **(quote or {"last": None, "change": None, "change_pct": None}),
        })
    return _with_cache(
        {"indices": out, "computed_at": datetime.utcnow().isoformat() + "Z"},
        max_age=30,
    )


# ============================================================================
# /signal-of-the-day — public hero card surface (PR 108)
# ============================================================================
# The landing hero (PR 74) currently shows the *best closed* signal of
# the last 30 days from /track-record. That works as social proof, but
# a returning visitor sees the same line every time. This endpoint
# surfaces today's *highest-confidence active* signal when one exists,
# falling back to the best closed signal of the last 7 days when the
# day is quiet — so the hero card always has something live to show.
#
# Public-safe: only the symbol + direction tag + confidence + entry
# price are exposed. Stop-loss, target, and any per-engine numerics
# stay behind the auth wall (Pro+ on /signals/{id}).


@router.get("/signal-of-the-day")
async def signal_of_the_day():
    """Best signal to feature on the public landing card.

    Selection priority:
        1. Highest-confidence active swing signal generated today
        2. Best closed signal (largest realized gain) from the last 7 days
        3. None — landing falls back to its skeleton state

    The response shape is intentionally minimal — full numerics live
    behind auth on /signals/{id}.
    """
    client = get_supabase_admin()
    today_iso = date.today().isoformat()

    # 1. Try today's highest-confidence active signal.
    try:
        active = (
            client.table("signals")
            .select(
                "id, symbol, direction, segment, confidence, entry_price, "
                "regime_at_signal, generated_at, status"
            )
            .eq("date", today_iso)
            .in_("status", ["active", "triggered"])
            .order("confidence", desc=True)
            .limit(1)
            .execute()
        )
        active_rows = active.data or []
    except Exception as exc:
        logger.warning("signal_of_the_day active query failed: %s", exc)
        active_rows = []

    if active_rows:
        row = active_rows[0]
        return _with_cache({
            "kind": "active",
            "symbol": str(row.get("symbol") or "").replace(".NS", ""),
            "direction": row.get("direction"),
            "segment": row.get("segment"),
            "confidence": int(row.get("confidence") or 0),
            "entry_price": _safe_float_or_none(row.get("entry_price")),
            "regime_at_signal": row.get("regime_at_signal"),
            "generated_at": row.get("generated_at"),
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }, max_age=60)

    # 2. Fall back to the best closed winner from the last 7 days.
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    try:
        closed = (
            client.table("signals")
            .select(
                "id, symbol, direction, segment, confidence, entry_price, "
                "target_1, generated_at, date, status"
            )
            .gte("date", cutoff)
            .eq("status", "target_hit")
            .order("date", desc=True)
            .limit(50)
            .execute()
        )
        closed_rows = closed.data or []
    except Exception as exc:
        logger.warning("signal_of_the_day closed query failed: %s", exc)
        closed_rows = []

    best: Optional[Dict[str, Any]] = None
    best_pct = -1.0
    for r in closed_rows:
        entry = float(r.get("entry_price") or 0)
        target = float(r.get("target_1") or 0)
        if entry <= 0 or target <= 0:
            continue
        pct = ((target - entry) / entry) * 100.0
        if r.get("direction") == "SHORT":
            pct = -pct
        if pct > best_pct:
            best_pct = pct
            best = r

    if best:
        return _with_cache({
            "kind": "closed_winner",
            "symbol": str(best.get("symbol") or "").replace(".NS", ""),
            "direction": best.get("direction"),
            "segment": best.get("segment"),
            "return_pct": round(best_pct, 2),
            "closed_on": best.get("date"),
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }, max_age=300)

    return _with_cache({
        "kind": "none",
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }, max_age=60)


def _safe_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
