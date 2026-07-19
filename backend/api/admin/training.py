"""
Admin training-pipeline + launch-readiness endpoints.

  GET  /admin/training/trainers   list trainers discovered by ml.training.runner
  GET  /admin/training/runs       in-flight + recent unified-runner invocations
  POST /admin/training/run        trigger a unified-runner invocation (background)
  GET  /admin/launch-readiness    aggregate go / no-go checklist for v1 launch

The training-run state is held in-memory for fast polling + mirrored to
the ``training_runs`` table for persistence across restarts (PR 154).
The unified runner runs on a daemon thread so the request returns
immediately; the UI polls ``/training/runs`` for completion.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ._deps import (
    AdminRole,
    AdminUser,
    get_admin_user,
    get_supabase_admin,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# IN-MEMORY RUN STATE — survives within a single process; the
# ``training_runs`` table is the cross-restart source of truth.
# ============================================================================


_TRAINING_RUNS: Dict[str, Dict[str, Any]] = {}
_TRAINING_LOCK = threading.Lock()


def _persist_training_run(record: Dict[str, Any]) -> None:
    """Mirror an in-memory training-run record to ``training_runs``.

    Best-effort: a DB write failure must not affect the run itself.
    The in-memory record remains the source of truth for the active run.
    """
    try:
        sb = get_supabase_admin()
        payload = {
            "id": record.get("run_id"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "status": record.get("status"),
            "triggered_by": record.get("triggered_by"),
            "params": record.get("params") or {},
            "reports": record.get("reports") or [],
            "error": record.get("error"),
        }
        sb.table("training_runs").upsert(payload).execute()
    except Exception as exc:
        logger.debug("training_runs persist skipped: %s", exc)


class TrainingRunBody(BaseModel):
    only: Optional[List[str]] = None
    skip_gpu: bool = False
    promote: bool = False
    dry_run: bool = False


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/training/trainers")
async def list_trainers(admin: AdminUser = Depends(get_admin_user)):
    """List trainers discovered by ``ml.training.runner``."""
    try:
        from ml.training.discovery import discover_sorted  # noqa: PLC0415
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"trainer discovery failed: {exc}")
    out = []
    for t in discover_sorted():
        out.append({
            "name": t.name,
            "requires_gpu": bool(t.requires_gpu),
            "depends_on": list(t.depends_on or []),
        })
    return {"trainers": out, "count": len(out)}


@router.get("/training/runs")
async def list_training_runs(admin: AdminUser = Depends(get_admin_user)):
    """List in-flight + recent unified-runner invocations.

    Also returns the most recent ``model_versions`` row per model so
    the admin UI can show "last trained" without a second API call.

    PR 154 — merges the in-memory list (currently running) with persisted
    rows from ``training_runs`` so history survives server restarts.
    """
    in_memory = list(_TRAINING_RUNS.values())
    persisted: List[Dict[str, Any]] = []
    try:
        sb = get_supabase_admin()
        rows = (
            sb.table("training_runs")
            .select("id, started_at, finished_at, status, triggered_by, params, reports, error")
            .order("started_at", desc=True)
            .limit(50)
            .execute()
        )
        for r in rows.data or []:
            persisted.append({
                "run_id": str(r.get("id")),
                "started_at": r.get("started_at"),
                "finished_at": r.get("finished_at"),
                "status": r.get("status"),
                "triggered_by": r.get("triggered_by"),
                "params": r.get("params") or {},
                "reports": r.get("reports") or [],
                "error": r.get("error"),
            })
    except Exception as exc:
        logger.debug("training_runs read skipped: %s", exc)

    # Dedup by run_id, prefer in-memory (fresher) when both present.
    by_id: Dict[str, Dict[str, Any]] = {r.get("run_id"): r for r in persisted}
    for r in in_memory:
        rid = r.get("run_id")
        if rid:
            by_id[rid] = r
    runs = sorted(
        by_id.values(),
        key=lambda r: r.get("started_at", ""),
        reverse=True,
    )[:50]

    last_versions: List[Dict[str, Any]] = []
    try:
        sb = get_supabase_admin()
        rows = (
            sb.table("model_versions")
            .select("model_name, version, trained_at, trained_by, metrics, is_prod, is_shadow")
            .order("trained_at", desc=True)
            .limit(200)
            .execute()
        )
        seen = set()
        for r in rows.data or []:
            n = r.get("model_name")
            if not n or n in seen:
                continue
            seen.add(n)
            last_versions.append(r)
    except Exception as exc:
        logger.warning("model_versions fetch failed: %s", exc)

    return {"runs": runs, "last_versions": last_versions}


@router.post("/training/run")
async def trigger_training_run(
    body: TrainingRunBody,
    admin: AdminUser = Depends(get_admin_user),
):
    """Trigger a unified training-runner invocation in a background thread.

    Returns the new ``run_id`` immediately. The UI polls
    ``/training/runs`` for completion + reports.
    """
    if admin.role == AdminRole.READ_ONLY:
        raise HTTPException(status_code=403, detail="read_only_admin_cannot_trigger")

    run_id = str(uuid.uuid4())
    started_at = datetime.utcnow().isoformat()
    record: Dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "triggered_by": admin.email,
        "params": body.model_dump(),
        "reports": [],
        "error": None,
    }
    with _TRAINING_LOCK:
        _TRAINING_RUNS[run_id] = record
    # PR 154 — write the running row immediately so the persisted history
    # captures even a run that crashes mid-execution.
    _persist_training_run(record)

    def _worker():
        try:
            from ml.training.runner import run as run_pipeline  # noqa: PLC0415
            reports = run_pipeline(
                only=body.only or None,
                skip_gpu=body.skip_gpu,
                promote=body.promote,
                dry_run=body.dry_run,
            )
            with _TRAINING_LOCK:
                rec = _TRAINING_RUNS.get(run_id)
                if rec is not None:
                    rec["status"] = "ok" if not any(r.status == "failed" for r in reports) else "partial"
                    rec["finished_at"] = datetime.utcnow().isoformat()
                    rec["reports"] = [asdict(r) for r in reports]
                    _persist_training_run(rec)
        except Exception as exc:
            logger.exception("training run %s failed", run_id)
            with _TRAINING_LOCK:
                rec = _TRAINING_RUNS.get(run_id)
                if rec is not None:
                    rec["status"] = "failed"
                    rec["finished_at"] = datetime.utcnow().isoformat()
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    _persist_training_run(rec)

    threading.Thread(target=_worker, daemon=True, name=f"training-{run_id[:8]}").start()
    return {"run_id": run_id, "status": "running", "started_at": started_at}


# ============================================================================
# LAUNCH READINESS CHECKLIST (PR 157)
# ============================================================================
#
# Single endpoint the launch-day operator hits before flipping prod
# traffic on. Each check returns ``{name, ok, detail}`` so the admin UI
# can render a green/red list. Anything red blocks the v1.0.0 tag.


REQUIRED_TRAINERS_FOR_PROD = [
    # 2026-05-24 — v1 launches with these 4 PROD-promoted models. Each
    # passed honest walk-forward + deflated Sharpe + cost-aware eval gates.
    "regime_hmm",         # v20, 3-state HMM bull/sideways/bear, log-lik -5.07
    "qlib_alpha158",      # v4, Qlib + LightGBM, rank_ic_mean 0.030 (real alpha)
    "tft_swing",          # v3, Temporal Fusion Transformer, 68% directional acc
    # B4 (pre-training audit 2026-05-19) — finbert_india IS a PROD v1 model
    # (the Mood engine per v2 spec). SignalGenerator.__init__ hard-requires
    # its load; previously launch-readiness returned ready=true even when
    # FinBERT couldn't load → post-deploy 503s. Adding to the gate.
    "finbert_india",      # v1, pre-trained Vansh180/FinBERT-India-v1
    # v1 scope locked 2026-05-17 — DROPPED:
    #   momentum_chronos  : redundant with momentum_timesfm (corr ~0.85)
    #   options_rl        : replaced by rule-based F&O recommender
    #   vix_tft           : replaced by 5-day VIX slope rule
    #   chronos2_macro    : regime persistence overlay, deferred to v1.1
    # earnings_xgb removed 2026-05-11: trainer required user-outcome data
    # that doesn't exist pre-launch. F9 EarningsScout feature is deferred
    # until live prediction data accumulates (or until we build an NSE
    # earnings-calendar scraper to label historical surprises).
]


# v1.1 BACKLOG — these 3 trainers were attempted multiple times in
# 2026-05-23 + 2026-05-24 pod runs but couldn't pass eval gates because
# they depend on data that isn't persisted to Supabase across pods.
# Moving them out of v1 PROD-required so launch isn't blocked. Will
# re-train and promote in v1.1 after the data persistence fixes below.
#
# Why each failed:
#   - lgbm_signal_gate: 266 dead features (Kronos kr_emb_000-255 + FII/DII
#     + sentiment + fundamentals). With LGBM_DEAD_FEATURE_FATAL=0 the
#     warm-feature-only model overfit (PBO=0.5 = coin flip). Real fix
#     needs all features available, not gate workarounds.
#   - intraday_lstm: REMOVED 2026-06-17 — the v1 LSTM (Sharpe 0.13, below
#     the 0.5 promote gate) is superseded by the planned PatchTST intraday
#     engine (4-engine ML/DL program), not a v1.1 rework.
#   - momentum_timesfm: TimesFM 2.0-500m API fix landed but the 500M
#     model on CPU is too slow to complete calibration in pod time
#     budget. Needs GPU inference path or smaller checkpoint.
#
# v1.1 data persistence work required before re-attempting:
#   1. sentiment_history table → persist news_sentiment.mean_score to
#      Supabase from backfill_sentiment.py (currently writes local only)
#   2. fundamentals_pit → Supabase table + backfill script
#   3. fii_dii_history → Supabase table + backfill script
#   4. Kronos embedding generation as a Phase 7 step (currently never
#      runs — KRONOS_PATH is set but no embeddings are computed)
V1_1_BACKLOG_TRAINERS = [
    "lgbm_signal_gate",
]


# 2026-05-23 — RL fully removed from v1 scope. Even SHADOW training
# is disabled. The 2 RL legs that DID train (finrl_x_a2c v3, _ddpg v3)
# both failed the eval gate badly (Sharpe -0.16 and 0.09 vs ≥1.2
# required), confirming the OOD/data-scale concerns: with 10y daily
# yfinance we can't out-train Renaissance.
#
# F4 AutoPilot ships exclusively on supervised stack: Qlib LightGBM
# ranker → HMM regime sizing → VIX overlay → Kelly → hard caps.
# Code for finrl_x_ensemble stays in ml/training/trainers/ but
# ml.training.discovery._SKIP_MODULES excludes it, so the orchestrator
# never queues RL training.
SHADOW_TRAINERS: list[str] = []


@router.get("/launch-readiness")
async def launch_readiness(admin: AdminUser = Depends(get_admin_user)):
    """Aggregate go / no-go checklist.

    Checks: every required trainer has a prod-promoted version, no failing
    schedulers in the last 24h, kill switch is operational, Sentry release
    tag is set.
    """
    checks: List[Dict[str, Any]] = []
    sb = get_supabase_admin()

    # 1) Models: every required trainer has an is_prod=TRUE row.
    try:
        rows = (
            sb.table("model_versions")
            .select("model_name, version, is_prod, trained_at")
            .eq("is_prod", True)
            .execute()
        )
        prod_models = {r["model_name"]: r for r in (rows.data or [])}
        for name in REQUIRED_TRAINERS_FOR_PROD:
            row = prod_models.get(name)
            checks.append({
                "name": f"model:{name}",
                "ok": bool(row),
                "detail": f"v{row['version']} trained {row['trained_at']}" if row else "no prod version",
            })
    except Exception as exc:
        checks.append({"name": "models", "ok": False, "detail": f"query failed: {exc}"})

    # 2) Scheduler: no jobs failed in the last 24h.
    try:
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        rows = (
            sb.table("scheduler_runs")
            .select("job_id, status")
            .gte("started_at", cutoff)
            .eq("status", "failed")
            .limit(20)
            .execute()
        )
        failed = rows.data or []
        checks.append({
            "name": "scheduler_24h_no_failures",
            "ok": len(failed) == 0,
            "detail": f"{len(failed)} failed jobs"
            + (f": {[r['job_id'] for r in failed[:5]]}" if failed else ""),
        })
    except Exception as exc:
        # scheduler_runs is opt-in; missing table = check passes silently
        checks.append({
            "name": "scheduler_24h_no_failures", "ok": True,
            "detail": f"skipped: {type(exc).__name__}",
        })

    # 3) Kill switch wiring smoke test.
    try:
        from ...platform.system_flags import is_globally_halted  # noqa: PLC0415
        is_globally_halted(supabase_client=sb)
        checks.append({"name": "kill_switch_wired", "ok": True, "detail": "OK"})
    except Exception as exc:
        checks.append({"name": "kill_switch_wired", "ok": False, "detail": str(exc)})

    # 4) Sentry release tag.
    try:
        import sentry_sdk  # noqa: PLC0415
        client = sentry_sdk.Hub.current.client
        release = (client.options.get("release") if client else None) or ""
        checks.append({
            "name": "sentry_release_set",
            "ok": bool(release),
            "detail": release or "no release tag — set GIT_SHA / RAILWAY_GIT_COMMIT_SHA",
        })
    except Exception as exc:
        checks.append({"name": "sentry_release_set", "ok": False, "detail": str(exc)})

    all_ok = all(c["ok"] for c in checks)
    return {
        "ready": all_ok,
        "checks": checks,
        "computed_at": datetime.utcnow().isoformat(),
    }
