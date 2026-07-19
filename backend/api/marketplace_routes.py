"""
================================================================================
STRATEGY MARKETPLACE API ROUTES
================================================================================
Browse, deploy, manage algo strategies.
Sprint 1: Read-only catalog + detail + backtest data
Sprint 2: Deploy, update, deactivate
================================================================================
"""

import logging
from typing import Optional
from fastapi import APIRouter, Query, HTTPException, Depends, Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/marketplace", tags=["Strategy Marketplace"])


# ============================================================================
# HELPERS
# ============================================================================

def _get_supabase():
    from .app import get_supabase_admin
    return get_supabase_admin()


async def _retry_query(fn, fallback=None):
    from .app import supabase_query_with_retry
    return await supabase_query_with_retry(fn, retries=2, timeout_fallback=fallback)


def _get_current_user(credentials=None):
    """Lazy import to avoid circular deps"""
    from .app import get_current_user
    return get_current_user


def _get_user_profile(user=None):
    from .app import get_user_profile
    return get_user_profile


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class StrategyListItem(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    category: str
    segment: str
    template_slug: str
    min_capital: int
    risk_level: str
    tier_required: str
    icon: Optional[str] = None
    tags: list = []
    is_featured: bool = False
    backtest_total_return: Optional[float] = None
    backtest_win_rate: Optional[float] = None
    backtest_profit_factor: Optional[float] = None
    backtest_sharpe: Optional[float] = None
    backtest_max_drawdown: Optional[float] = None
    backtest_total_trades: Optional[int] = None


# ============================================================================
# CATALOG ENDPOINTS (Sprint 1 — read-only)
# ============================================================================

@router.get("/strategies")
async def list_strategies(
    category: Optional[str] = Query(None, description="Filter by category"),
    segment: Optional[str] = Query(None, description="Filter by EQUITY or OPTIONS"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    tier: Optional[str] = Query(None, description="Filter by required tier"),
    search: Optional[str] = Query(None, description="Search by name or tags"),
    sort_by: str = Query("sort_order", description="Sort field"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all available strategies in the marketplace."""
    try:
        supabase = _get_supabase()

        def _fetch_strategies():
            query = supabase.table("strategy_catalog").select("*").eq("is_active", True)
            if category:
                query = query.eq("category", category)
            if segment:
                query = query.eq("segment", segment)
            if risk_level:
                query = query.eq("risk_level", risk_level)
            if tier:
                query = query.eq("tier_required", tier)
            if search:
                query = query.or_(f"name.ilike.%{search}%,description.ilike.%{search}%")
            desc_order = sort_by.startswith("-")
            field = sort_by.lstrip("-")
            valid_sorts = ["sort_order", "name", "backtest_total_return", "backtest_win_rate",
                           "backtest_profit_factor", "backtest_sharpe", "min_capital"]
            if field not in valid_sorts:
                field = "sort_order"
            query = query.order(field, desc=desc_order)
            query = query.range(offset, offset + limit - 1)
            return query.execute().data or []

        strategies = await _retry_query(_fetch_strategies, fallback=[])

        # Category counts from the fetched data (avoids second query)
        def _fetch_cats():
            return supabase.table("strategy_catalog").select("category").eq("is_active", True).execute().data or []

        all_cats = await _retry_query(_fetch_cats, fallback=[])
        category_counts = {}
        for row in all_cats:
            cat = row.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "success": True,
            "strategies": strategies,
            "total": len(all_cats) or len(strategies),
            "category_counts": category_counts,
        }

    except Exception as e:
        logger.error(f"List strategies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{slug}")
async def get_strategy_detail(slug: str = Path(..., description="Strategy slug")):
    """Get full strategy details including configurable params."""
    try:
        supabase = _get_supabase()
        # maybe_single() returns data=None for 0 rows instead of raising
        # PGRST116 (which .single() does and would bubble up as 500 —
        # discovered by the API contract audit 2026-05-18). Unknown slugs
        # are a normal not-found case, not an internal error.
        result = (
            supabase.table("strategy_catalog")
            .select("*")
            .eq("slug", slug)
            .eq("is_active", True)
            .maybe_single()
            .execute()
        )

        # maybe_single().execute() returns None (not an object with
        # data=None) when 0 rows match — must null-check before
        # dereferencing .data.
        if result is None or not getattr(result, "data", None):
            raise HTTPException(status_code=404, detail="Strategy not found")

        return {
            "success": True,
            "strategy": result.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{slug}/backtest")
async def get_strategy_backtest(slug: str = Path(..., description="Strategy slug")):
    """Get backtest results for a strategy (equity curve, heatmap, trade log)."""
    try:
        supabase = _get_supabase()

        # Get strategy ID
        strategy = (
            supabase.table("strategy_catalog")
            .select("id")
            .eq("slug", slug)
            .single()
            .execute()
        )
        if not strategy.data:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy_id = strategy.data["id"]

        # Get latest backtest
        backtest = (
            supabase.table("strategy_backtests")
            .select("*")
            .eq("strategy_id", strategy_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if not backtest.data:
            # Return summary from catalog as fallback
            catalog = (
                supabase.table("strategy_catalog")
                .select("backtest_total_return,backtest_cagr,backtest_win_rate,backtest_profit_factor,backtest_sharpe,backtest_max_drawdown,backtest_total_trades")
                .eq("slug", slug)
                .single()
                .execute()
            )
            return {
                "success": True,
                "backtest": None,
                "summary": catalog.data if catalog.data else None,
            }

        return {
            "success": True,
            "backtest": backtest.data[0],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DEPLOYMENT ENDPOINTS (Sprint 2)
# ============================================================================

class DeployStrategyRequest(BaseModel):
    strategy_slug: str
    allocated_capital: float = 100000
    max_positions: int = 2
    trade_mode: str = "signal_only"
    custom_params: dict = {}


class UpdateDeploymentRequest(BaseModel):
    allocated_capital: Optional[float] = None
    max_positions: Optional[int] = None
    trade_mode: Optional[str] = None
    custom_params: Optional[dict] = None
    is_paused: Optional[bool] = None


@router.post("/deploy")
async def deploy_strategy(req: DeployStrategyRequest, user=Depends(_get_current_user())):
    """Deploy a strategy for the current user."""
    try:
        supabase = _get_supabase()

        # Get strategy
        strategy = (
            supabase.table("strategy_catalog")
            .select("*")
            .eq("slug", req.strategy_slug)
            .eq("is_active", True)
            .single()
            .execute()
        )
        if not strategy.data:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strat = strategy.data

        # Get user profile for subscription checks
        profile = (
            supabase.table("user_profiles")
            .select("*, subscription_plans(*)")
            .eq("id", user.id)
            .single()
            .execute()
        )
        if not profile.data:
            raise HTTPException(status_code=404, detail="Profile not found")

        plan = profile.data.get("subscription_plans", {})

        # Tier gate. elite is highest; legacy 'starter' maps to pro's rank.
        tier_order = {"free": 0, "starter": 1, "pro": 1, "elite": 2}
        user_tier = plan.get("name", "free")
        required_tier = strat.get("tier_required", "free")
        if tier_order.get(user_tier, 0) < tier_order.get(required_tier, 0):
            raise HTTPException(
                status_code=403,
                detail=f"Upgrade to {required_tier.title()} plan to deploy this strategy",
            )

        # Count existing deployments
        existing = (
            supabase.table("user_strategy_deployments")
            .select("id", count="exact")
            .eq("user_id", user.id)
            .eq("is_active", True)
            .execute()
        )
        max_strategies = plan.get("max_strategies", 6)
        if len(existing.data or []) >= max_strategies:
            raise HTTPException(
                status_code=403,
                detail=f"Your plan allows {max_strategies} strategies. Upgrade to add more.",
            )

        # Mode gate
        auto_trade_mode = plan.get("auto_trade_mode", "signal_only")
        mode_order = {"signal_only": 0, "semi_auto": 1, "full_auto": 2}
        if mode_order.get(req.trade_mode, 0) > mode_order.get(auto_trade_mode, 0):
            raise HTTPException(
                status_code=403,
                detail=f"Upgrade your plan to use {req.trade_mode} mode",
            )

        # F&O gate
        if strat.get("requires_fo_enabled") and not profile.data.get("fo_enabled"):
            raise HTTPException(status_code=403, detail="Enable F&O trading in settings first")

        # Capital gate
        if req.allocated_capital < strat.get("min_capital", 0):
            raise HTTPException(
                status_code=400,
                detail=f"Minimum capital for this strategy is ₹{strat['min_capital']:,}",
            )

        # Create deployment (upsert)
        deployment = (
            supabase.table("user_strategy_deployments")
            .upsert({
                "user_id": user.id,
                "strategy_id": strat["id"],
                "custom_params": req.custom_params,
                "allocated_capital": req.allocated_capital,
                "max_positions": req.max_positions,
                "trade_mode": req.trade_mode,
                "is_active": True,
                "is_paused": False,
            }, on_conflict="user_id,strategy_id")
            .execute()
        )

        return {
            "success": True,
            "deployment": deployment.data[0] if deployment.data else None,
            "message": f"Strategy '{strat['name']}' deployed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deploy strategy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my-strategies")
async def get_my_strategies(user=Depends(_get_current_user())):
    """Get all strategies deployed by the current user."""
    try:
        supabase = _get_supabase()
        result = (
            supabase.table("user_strategy_deployments")
            .select("*, strategy_catalog(*)")
            .eq("user_id", user.id)
            .eq("is_active", True)
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "success": True,
            "deployments": result.data or [],
            "total": len(result.data or []),
        }

    except Exception as e:
        logger.error(f"My strategies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/deployments/{deployment_id}")
async def update_deployment(
    deployment_id: str,
    req: UpdateDeploymentRequest,
    user=Depends(_get_current_user()),
):
    """Update a deployment's configuration."""
    try:
        supabase = _get_supabase()

        # Verify ownership
        existing = (
            supabase.table("user_strategy_deployments")
            .select("*")
            .eq("id", deployment_id)
            .eq("user_id", user.id)
            .single()
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Deployment not found")

        update_data = {}
        if req.allocated_capital is not None:
            update_data["allocated_capital"] = req.allocated_capital
        if req.max_positions is not None:
            update_data["max_positions"] = req.max_positions
        if req.trade_mode is not None:
            update_data["trade_mode"] = req.trade_mode
        if req.custom_params is not None:
            update_data["custom_params"] = req.custom_params
        if req.is_paused is not None:
            update_data["is_paused"] = req.is_paused

        if not update_data:
            return {"success": True, "deployment": existing.data}

        # Re-apply the user_id filter on the UPDATE itself (defense in
        # depth — the SELECT above already gated ownership, but matching
        # the DELETE's filter pattern means a future refactor can't drop
        # the SELECT and silently turn this into a write-anywhere bug).
        result = (
            supabase.table("user_strategy_deployments")
            .update(update_data)
            .eq("id", deployment_id)
            .eq("user_id", user.id)
            .execute()
        )

        return {
            "success": True,
            "deployment": result.data[0] if result.data else existing.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/deployments/{deployment_id}")
async def deactivate_deployment(deployment_id: str, user=Depends(_get_current_user())):
    """Deactivate (soft-delete) a deployment."""
    try:
        supabase = _get_supabase()

        result = (
            supabase.table("user_strategy_deployments")
            .update({"is_active": False})
            .eq("id", deployment_id)
            .eq("user_id", user.id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Deployment not found")

        return {"success": True, "message": "Strategy deactivated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
