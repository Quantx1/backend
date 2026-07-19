"""
Admin payments + signals stats endpoints.

  GET /admin/payments          paginated payment list with filters
  GET /admin/payments/stats    revenue / failures / refunds rollup
  GET /admin/signals/stats     signal-generation stats (count, accuracy)

Read-only reporting surfaces for the admin command-center revenue tile
and the model-quality strip. No mutations — both gated on
``Depends(get_admin_user)`` (any admin role can view).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ._deps import AdminUser, get_admin_user, get_supabase_admin

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# PAYMENTS
# ============================================================================


@router.get("/payments")
async def list_payments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    admin: AdminUser = Depends(get_admin_user),
):
    """List all payments with filters."""
    supabase = get_supabase_admin()

    query = supabase.table("payments").select(
        "*, user_profiles(email, full_name), subscription_plans(display_name)",
        count="exact",
    )

    if status:
        query = query.eq("status", status)
    if user_id:
        query = query.eq("user_id", user_id)

    offset = (page - 1) * page_size
    result = (
        query.order("created_at", desc=True)
        .range(offset, offset + page_size - 1)
        .execute()
    )

    return {
        "payments": result.data or [],
        "total": result.count or 0,
        "page": page,
        "page_size": page_size,
    }


@router.get("/payments/stats")
async def get_payment_stats(
    days: int = Query(30, ge=1, le=365),
    admin: AdminUser = Depends(get_admin_user),
):
    """Get payment statistics over the last ``days`` window."""
    supabase = get_supabase_admin()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    # Completed payments in period
    completed = (
        supabase.table("payments")
        .select("amount", count="exact")
        .eq("status", "completed")
        .gte("completed_at", start_date)
        .execute()
    )
    total_revenue = sum(p.get("amount", 0) for p in completed.data or [])

    # Failed payments
    failed = (
        supabase.table("payments")
        .select("id", count="exact")
        .eq("status", "failed")
        .gte("created_at", start_date)
        .execute()
    )

    # Refunds (synced from Razorpay webhook — refund initiation surface
    # was removed 2026-05-07 per the no-refund-surface rule).
    refunds = (
        supabase.table("payments")
        .select("amount", count="exact")
        .eq("status", "refunded")
        .gte("created_at", start_date)
        .execute()
    )
    total_refunds = sum(p.get("amount", 0) for p in refunds.data or [])

    return {
        "period_days": days,
        "total_revenue": total_revenue / 100,  # paise → INR
        "completed_payments": completed.count or 0,
        "failed_payments": failed.count or 0,
        "refunds_count": refunds.count or 0,
        "refunds_amount": total_refunds / 100,
        "net_revenue": (total_revenue - total_refunds) / 100,
    }


# ============================================================================
# SIGNALS STATS — admin model-quality strip
# ============================================================================


@router.get("/signals/stats")
async def get_signals_stats(
    days: int = Query(30, ge=1, le=365),
    admin: AdminUser = Depends(get_admin_user),
):
    """Signal generation statistics (volume + hit rate)."""
    supabase = get_supabase_admin()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    total = (
        supabase.table("signals")
        .select("id", count="exact")
        .gte("date", start_date)
        .execute()
    )
    target_hit = (
        supabase.table("signals")
        .select("id", count="exact")
        .eq("status", "target_hit")
        .gte("date", start_date)
        .execute()
    )
    sl_hit = (
        supabase.table("signals")
        .select("id", count="exact")
        .eq("status", "sl_hit")
        .gte("date", start_date)
        .execute()
    )

    total_resolved = (target_hit.count or 0) + (sl_hit.count or 0)
    accuracy = ((target_hit.count or 0) / total_resolved * 100) if total_resolved > 0 else 0

    return {
        "period_days": days,
        "total_signals": total.count or 0,
        "target_hit": target_hit.count or 0,
        "sl_hit": sl_hit.count or 0,
        "accuracy": round(accuracy, 2),
        "avg_per_day": round((total.count or 0) / days, 1),
    }
