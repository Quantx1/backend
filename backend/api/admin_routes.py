"""
================================================================================
QUANT X - ADMIN CONSOLE API ROUTES
================================================================================
Top-level admin router. Sub-domains live in the ``admin/`` package and
get included into this router (see end of file). Shared types + the
``get_admin_user`` dependency live in ``admin/_deps.py`` and are
re-exported here for back-compat with anything that does
``from .admin_routes import get_admin_user`` (e.g. payment_routes lazy import).
================================================================================
"""

from .admin.observability import router as _observability_router
from .admin.training import router as _training_router
from .admin.eod import router as _eod_router
from .admin.ml import router as _ml_router
from .admin.payments import router as _payments_router
from .admin.system import router as _system_router
from .admin.users import router as _users_router
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends

# Shared admin types + auth gate. Re-exported below for back-compat with
# anything that does ``from .admin_routes import get_admin_user`` (e.g.
# payment_routes lazy import).
from .admin._deps import (
    AdminUser,
    get_admin_user,
    get_supabase_admin,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/verify")
async def verify_admin(admin: AdminUser = Depends(get_admin_user)) -> Dict[str, Any]:
    """Lightweight check used by the frontend admin layout to decide whether
    to render the admin shell. Returns 200 + {is_admin: true} for admins,
    403 for everyone else (thrown by get_admin_user dependency)."""
    return {"is_admin": True, "role": admin.role.value, "email": admin.email}


@router.get("/model-performance")
async def admin_model_performance(
    admin: AdminUser = Depends(get_admin_user),
) -> Dict[str, Any]:
    """MED #5 (2026-05-31) — Per-model performance dashboard.

    Admin-only. Returns live IC + backtest IC + rolling Sharpe per
    PROD model from `model_rolling_performance` + `model_versions`,
    so admins can decompose "Qlib + HMM + TFT + FinBERT contributed
    X / Y / Z to this week's P&L" without exposing IP to end users.
    """
    sb = get_supabase_admin()
    out: Dict[str, Any] = {"models": [], "errors": []}
    try:
        prod = (
            sb.table("model_versions")
            .select("model_name, version, is_prod, metrics, trained_at")
            .eq("is_prod", True)
            .eq("is_retired", False)
            .execute()
            .data
            or []
        )
    except Exception as e:
        out["errors"].append(f"model_versions read failed: {e}")
        return out

    for m in prod:
        name = m.get("model_name")
        rolling = []
        try:
            rolling = (
                sb.table("model_rolling_performance")
                .select(
                    "window_days, win_rate, avg_pnl_pct, signal_count, "
                    "directional_accuracy, sharpe_ratio, max_drawdown_pct, "
                    "computed_at"
                )
                .eq("model_name", name)
                .order("computed_at", desc=True)
                .limit(10)
                .execute()
                .data
                or []
            )
        except Exception as e:
            out["errors"].append(f"rolling read for {name}: {e}")

        metrics = m.get("metrics") or {}
        # Backtest Sharpe is the gate the model passed at promotion
        backtest_sharpe = (
            metrics.get("sharpe_ratio") or metrics.get("sharpe") or None
        )
        live_30d = next((r for r in rolling if r.get("window_days") == 30), None)
        live_sharpe = live_30d.get("sharpe_ratio") if live_30d else None
        ratio = None
        if backtest_sharpe and live_sharpe:
            try:
                ratio = float(live_sharpe) / float(backtest_sharpe)
            except Exception:
                pass

        out["models"].append({
            "model_name": name,
            "version": m.get("version"),
            "trained_at": m.get("trained_at"),
            "backtest_sharpe": backtest_sharpe,
            "live_sharpe_30d": live_sharpe,
            "drift_ratio": ratio,                  # <1.0 means live < backtest
            "rolling": rolling,
        })

    return out


# ============================================================================
# Sub-router includes — extracted domains live under admin/<domain>.py
# ============================================================================

router.include_router(_users_router)
router.include_router(_system_router)
router.include_router(_payments_router)
router.include_router(_ml_router)
router.include_router(_eod_router)
router.include_router(_training_router)
router.include_router(_observability_router)


# ============================================================================
# REGISTER ROUTES
# ============================================================================


def register_admin_routes(app):
    """Register the combined admin router with the FastAPI app.

    Exposed by ``app.py`` as ``register_admin_routes(app)`` — the public
    entry point. All sub-routers were already mounted into ``router``
    above via ``include_router``, so this just attaches the top-level
    router to the FastAPI instance under the ``/api`` prefix.
    """
    app.include_router(router, prefix="/api")
    logger.info("✅ Admin routes registered")
