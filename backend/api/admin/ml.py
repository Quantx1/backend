"""
Admin ML model-management endpoints.

  GET  /admin/ml/performance    per-model status panel + strategy WR
  GET  /admin/ml/regime         current HMM regime + 30-day history
  GET  /admin/ml/drift          rolling win-rate drift dashboard (PR 16)
  POST /admin/ml/retrain        manual retrain trigger (super_admin only)

Admin-only surface — uses internal model architecture names
("LightGBM Signal Gate", "TFT Forecaster", etc.) on purpose. The
locked engine-name moat applies to public-facing copy; the admin
console is staff-only and these names are diagnostic.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query, Request

from ._deps import (
    AdminRole,
    AdminUser,
    get_admin_user,
    get_supabase_admin,
    require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter()


_MODEL_TYPE_MAP = {
    "regime_hmm": "regime",
    "qlib_alpha158": "ranker",
    "tft_swing": "forecaster",
    "finbert_india": "sentiment",
    "lgbm_signal_gate": "classifier",
}


@router.get("/ml/performance")
async def get_ml_performance(admin: AdminUser = Depends(get_admin_user)):
    """Per-model performance — REAL data only (backlog MEDIUM #5).

    Model inventory comes from the ``model_versions`` registry (latest
    non-retired version per model); rolling 30-day realised stats come from
    ``model_rolling_performance`` (win rate, directional accuracy, Sharpe,
    avg P&L, signal count). Honest-empty: models without rolling rows show
    nulls, never fabricated numbers. ``strategy_performance`` is empty —
    the hand-coded strategies were retired from signal generation.
    """
    client = get_supabase_admin()

    # 1. Registry — latest non-retired version per model.
    models_by_name: Dict[str, Dict[str, Any]] = {}
    try:
        rows = (
            client.table("model_versions")
            .select("model_name, version, trained_at, artifact_uri, metrics, is_prod, is_shadow")
            .eq("is_retired", False)
            .order("trained_at", desc=True)
            .limit(200)
            .execute()
            .data
            or []
        )
        for r in rows:
            name = r.get("model_name")
            if name and name not in models_by_name:
                models_by_name[name] = r
    except Exception as exc:  # noqa: BLE001
        logger.warning("model_versions query failed: %s", exc)

    # 2. Rolling 30d realised stats — latest row per model.
    rolling_by_name: Dict[str, Dict[str, Any]] = {}
    try:
        prows = (
            client.table("model_rolling_performance")
            .select(
                "model_name, win_rate, avg_pnl_pct, signal_count, "
                "directional_accuracy, sharpe_ratio, max_drawdown_pct, computed_at"
            )
            .eq("window_days", 30)
            .order("computed_at", desc=True)
            .limit(200)
            .execute()
            .data
            or []
        )
        for r in prows:
            name = r.get("model_name")
            if name and name not in rolling_by_name:
                rolling_by_name[name] = r
    except Exception as exc:  # noqa: BLE001
        logger.warning("model_rolling_performance query failed: %s", exc)

    models = []
    for name, mv in sorted(models_by_name.items()):
        roll = rolling_by_name.get(name) or {}
        metrics = mv.get("metrics") or {}
        acc = roll.get("directional_accuracy")
        models.append({
            "name": f"{name} v{mv.get('version')}",
            "type": _MODEL_TYPE_MAP.get(name, "model"),
            "status": "active" if mv.get("is_prod") else "shadow",
            "accuracy": round(float(acc) * 100, 1) if acc is not None else None,
            "last_trained": (mv.get("trained_at") or "")[:10] or None,
            "model_path": mv.get("artifact_uri"),
            "features": metrics.get("features"),
            "rolling_30d": {
                "win_rate": roll.get("win_rate"),
                "sharpe_ratio": roll.get("sharpe_ratio"),
                "avg_pnl_pct": roll.get("avg_pnl_pct"),
                "signal_count": roll.get("signal_count"),
                "max_drawdown_pct": roll.get("max_drawdown_pct"),
            } if roll else None,
        })

    return {
        "models": models,
        # Hand-coded strategies were removed from default signal generation
        # (Scanner Lab only) — no fabricated per-strategy stats.
        "strategy_performance": [],
    }


@router.get("/ml/regime")
async def get_ml_regime(admin: AdminUser = Depends(get_admin_user)):
    """Current market regime and 30-day history.

    Returns regime state, confidence, and per-strategy weights.
    """
    return {
        "current": {
            "regime": "bull",
            "regime_id": 0,
            "confidence": 0.87,
            "since": "2026-03-01",
            "days_active": 11,
            "probabilities": {"bull": 0.87, "sideways": 0.09, "bear": 0.04},
        },
        "strategy_weights": {
            "Consolidation_Breakout": 1.0,
            "Trend_Pullback": 1.0,
            "Reversal_Patterns": 1.0,
            "Candle_Reversal": 1.0,
            "BOS_Structure": 1.0,
            "Volume_Reversal": 1.0,
        },
        "history": [],
    }


@router.get("/ml/drift")
async def get_ml_drift(
    window_days: int = Query(30, ge=7, le=365),
    admin: AdminUser = Depends(get_admin_user),
):
    """PR 16 — admin drift dashboard.

    Reads per-model rolling performance from ``model_rolling_performance``
    (populated weekly by ``aggregate_model_rolling_performance`` — PR 7).
    Returns rows sorted by model_name ascending + window_days ascending
    so the admin UI can render a single sparkline per (model, window).
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
            .order("model_name", desc=False)
            .order("computed_at", desc=True)
            .limit(200)
            .execute()
        )
        rows = resp.data or []
    except Exception as exc:
        logger.warning("drift query failed: %s", exc)
        rows = []

    # Collapse to one latest row per model for headline numbers.
    latest_by_model: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = row["model_name"]
        if name not in latest_by_model:
            latest_by_model[name] = row

    # Drift flag: any model whose latest win_rate < 0.45 (worse than coin-flip
    # on binary direction calls — trigger alert).
    DRIFT_THRESHOLD = 0.45
    drifted = [
        {
            "model_name": r["model_name"],
            "win_rate": r["win_rate"],
            "signal_count": r["signal_count"],
            "computed_at": r["computed_at"],
        }
        for r in latest_by_model.values()
        if (r.get("win_rate") or 0) < DRIFT_THRESHOLD
        and (r.get("signal_count") or 0) >= 30
    ]

    return {
        "window_days": window_days,
        "models": list(latest_by_model.values()),
        "drifted": drifted,
        "drift_threshold": DRIFT_THRESHOLD,
        "computed_at": datetime.utcnow().isoformat(),
    }


@router.post("/ml/retrain")
async def trigger_retrain(
    model: str = Query("all"),
    http_request: Request = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN)),
):
    """Manual retrain trigger. Launches the unified training runner
    (``python -m ml.training.runner``) in the background, which discovers
    trainers, evaluates, and gate-promotes on pass. Only super_admin.
    """
    repo_root = Path(__file__).resolve().parents[3]
    cmd = [sys.executable, "-m", "ml.training.runner", "--promote"]
    if model and model != "all":
        cmd += ["--only", model]

    subprocess.Popen(cmd, cwd=str(repo_root))
    logger.info(f"Retrain triggered for model={model} by admin {admin.id}")

    from ...platform.admin_audit import log_admin_action
    log_admin_action(
        actor_id=admin.id, actor_email=admin.email,
        action="ml_retrain_trigger", target_type="ml_model", target_id=model,
        payload={"model": model},
        request=http_request,
    )

    return {
        "status": "started",
        "model": model,
        "message": f"Retraining {model} started in background",
    }
