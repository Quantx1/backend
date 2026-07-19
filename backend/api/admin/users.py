"""
Admin user-management endpoints.

  GET  /admin/users                               list with paging + filter
  GET  /admin/users/{user_id}                     detail panel (trades, positions, payments)
  POST /admin/users/{user_id}/suspend             flag is_suspended=true
  POST /admin/users/{user_id}/unsuspend           clear suspension
  POST /admin/users/{user_id}/ban                 permanent (super_admin only)
  POST /admin/users/{user_id}/reset-subscription  switch tier or reset to free
  GET  /admin/users/export/csv                    full CSV export (super_admin only)

All endpoints are gated by ``Depends(get_admin_user)`` (role check in
the body). Mutating endpoints write a ``user_profiles`` update + an
``audit_log`` row + a structured admin-audit event.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ._deps import (
    AdminRole,
    AdminUser,
    SubscriptionResetRequest,
    UserActionRequest,
    UserDetailResponse,
    UserListItem,
    UserListResponse,
    get_admin_user,
    get_supabase_admin,
    require_role,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    subscription_status: Optional[str] = None,
    is_suspended: Optional[bool] = None,
    is_banned: Optional[bool] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    admin: AdminUser = Depends(get_admin_user),
):
    """List all users with pagination, search, and filters."""
    supabase = get_supabase_admin()

    query = supabase.table("user_profiles").select(
        "*, subscription_plans(name, display_name)",
        count="exact",
    )

    if search:
        query = query.or_(
            f"email.ilike.%{search}%,full_name.ilike.%{search}%,phone.ilike.%{search}%"
        )
    if subscription_status:
        query = query.eq("subscription_status", subscription_status)
    # Note: is_suspended and is_banned filters skipped if columns don't exist yet.

    query = query.order(sort_by, desc=(sort_order == "desc"))
    offset = (page - 1) * page_size
    query = query.range(offset, offset + page_size - 1)

    result = query.execute()
    total = result.count or 0

    users = []
    for row in result.data or []:
        plan_data = row.get("subscription_plans") or {}
        users.append(UserListItem(
            id=row["id"],
            email=row["email"],
            full_name=row.get("full_name"),
            phone=row.get("phone"),
            capital=row.get("capital", 0),
            trading_mode=row.get("trading_mode", "signal_only"),
            subscription_status=row.get("subscription_status", "free"),
            subscription_plan=plan_data.get("display_name"),
            broker_connected=row.get("broker_connected", False),
            broker_name=row.get("broker_name"),
            total_trades=row.get("total_trades", 0),
            winning_trades=row.get("winning_trades", 0),
            total_pnl=row.get("total_pnl", 0),
            created_at=row.get("created_at", ""),
            last_login=row.get("last_login"),
            last_active=row.get("last_active"),
            is_suspended=row.get("is_suspended", False),
            is_banned=row.get("is_banned", False),
        ))

    return UserListResponse(
        users=users,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(
    user_id: str,
    admin: AdminUser = Depends(get_admin_user),
):
    """Get detailed user info: profile, trading settings, trades, positions, payments, activity."""
    supabase = get_supabase_admin()

    profile = (
        supabase.table("user_profiles")
        .select("*, subscription_plans(name, display_name)")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not profile.data:
        raise HTTPException(status_code=404, detail="User not found")

    row = profile.data
    plan_data = row.get("subscription_plans") or {}

    user = UserListItem(
        id=row["id"],
        email=row["email"],
        full_name=row.get("full_name"),
        phone=row.get("phone"),
        capital=row.get("capital", 0),
        trading_mode=row.get("trading_mode", "signal_only"),
        subscription_status=row.get("subscription_status", "free"),
        subscription_plan=plan_data.get("display_name"),
        broker_connected=row.get("broker_connected", False),
        broker_name=row.get("broker_name"),
        total_trades=row.get("total_trades", 0),
        winning_trades=row.get("winning_trades", 0),
        total_pnl=row.get("total_pnl", 0),
        created_at=row.get("created_at", ""),
        last_login=row.get("last_login"),
        last_active=row.get("last_active"),
        is_suspended=row.get("is_suspended", False),
        is_banned=row.get("is_banned", False),
    )

    trading_settings = {
        "risk_profile": row.get("risk_profile"),
        "risk_per_trade": row.get("risk_per_trade"),
        "max_positions": row.get("max_positions"),
        "fo_enabled": row.get("fo_enabled"),
        "preferred_option_type": row.get("preferred_option_type"),
        "daily_loss_limit": row.get("daily_loss_limit"),
        "weekly_loss_limit": row.get("weekly_loss_limit"),
        "monthly_loss_limit": row.get("monthly_loss_limit"),
        "trailing_sl_enabled": row.get("trailing_sl_enabled"),
    }

    trades = (
        supabase.table("trades")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    positions = (
        supabase.table("positions")
        .select("*")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .execute()
    )
    payments = (
        supabase.table("payments")
        .select("*, subscription_plans(display_name)")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    activity = (
        supabase.table("audit_log")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )

    return UserDetailResponse(
        user=user,
        trading_settings=trading_settings,
        recent_activity=activity.data or [],
        payment_history=payments.data or [],
        positions=positions.data or [],
        trades=trades.data or [],
    )


@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    request: UserActionRequest,
    http_request: Request = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_ADMIN)),
):
    """Suspend a user account. User cannot login or trade."""
    supabase = get_supabase_admin()

    result = (
        supabase.table("user_profiles")
        .update({
            "is_suspended": True,
            "suspended_at": datetime.utcnow().isoformat(),
            "suspended_by": admin.id,
            "suspension_reason": request.reason,
        })
        .eq("id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    supabase.table("audit_log").insert({
        "user_id": user_id,
        "action": "user_suspended",
        "entity_type": "user_profile",
        "entity_id": user_id,
        "new_value": {"is_suspended": True, "reason": request.reason},
    }).execute()

    logger.info(f"User {user_id} suspended by admin {admin.id}")

    from ...platform.admin_audit import log_admin_action
    log_admin_action(
        actor_id=admin.id, actor_email=admin.email,
        action="user_suspend", target_type="user", target_id=user_id,
        payload={"reason": request.reason},
        request=http_request, supabase_client=supabase,
    )

    return {"success": True, "message": "User suspended"}


@router.post("/users/{user_id}/unsuspend")
async def unsuspend_user(
    user_id: str,
    http_request: Request = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_ADMIN)),
):
    """Remove suspension from a user account."""
    supabase = get_supabase_admin()

    result = (
        supabase.table("user_profiles")
        .update({
            "is_suspended": False,
            "suspended_at": None,
            "suspended_by": None,
            "suspension_reason": None,
        })
        .eq("id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    supabase.table("audit_log").insert({
        "user_id": user_id,
        "action": "user_unsuspended",
        "entity_type": "user_profile",
        "entity_id": user_id,
    }).execute()

    from ...platform.admin_audit import log_admin_action
    log_admin_action(
        actor_id=admin.id, actor_email=admin.email,
        action="user_unsuspend", target_type="user", target_id=user_id,
        request=http_request, supabase_client=supabase,
    )

    return {"success": True, "message": "User unsuspended"}


@router.post("/users/{user_id}/ban")
async def ban_user(
    user_id: str,
    request: UserActionRequest,
    http_request: Request = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN)),
):
    """Permanently ban a user. Only super_admin can ban."""
    supabase = get_supabase_admin()

    result = (
        supabase.table("user_profiles")
        .update({
            "is_banned": True,
            "banned_at": datetime.utcnow().isoformat(),
            "banned_by": admin.id,
            "ban_reason": request.reason,
        })
        .eq("id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    supabase.table("audit_log").insert({
        "user_id": user_id,
        "action": "user_banned",
        "entity_type": "user_profile",
        "entity_id": user_id,
        "new_value": {"is_banned": True, "reason": request.reason},
    }).execute()

    logger.warning(f"User {user_id} BANNED by admin {admin.id}")

    from ...platform.admin_audit import log_admin_action
    log_admin_action(
        actor_id=admin.id, actor_email=admin.email,
        action="user_ban", target_type="user", target_id=user_id,
        payload={"reason": request.reason},
        request=http_request, supabase_client=supabase,
    )

    return {"success": True, "message": "User banned"}


@router.post("/users/{user_id}/reset-subscription")
async def reset_subscription(
    user_id: str,
    request: SubscriptionResetRequest,
    http_request: Request = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_ADMIN)),
):
    """Reset user's subscription status."""
    supabase = get_supabase_admin()

    update_data = {
        "subscription_status": request.new_status,
        "subscription_end": None if request.new_status == "free" else None,
    }

    if request.new_plan_id:
        plan = (
            supabase.table("subscription_plans")
            .select("id")
            .eq("id", request.new_plan_id)
            .single()
            .execute()
        )
        if not plan.data:
            raise HTTPException(status_code=400, detail="Invalid plan ID")
        update_data["subscription_plan_id"] = request.new_plan_id

    result = (
        supabase.table("user_profiles")
        .update(update_data)
        .eq("id", user_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")

    supabase.table("audit_log").insert({
        "user_id": user_id,
        "action": "subscription_reset",
        "entity_type": "user_profile",
        "entity_id": user_id,
        "new_value": update_data,
    }).execute()

    from ...platform.admin_audit import log_admin_action
    log_admin_action(
        actor_id=admin.id, actor_email=admin.email,
        action="subscription_reset", target_type="tier", target_id=user_id,
        payload=update_data,
        request=http_request, supabase_client=supabase,
    )

    return {"success": True, "message": "Subscription reset"}


@router.get("/users/export/csv")
async def export_users_csv(
    subscription_status: Optional[str] = None,
    admin: AdminUser = Depends(require_role(AdminRole.SUPER_ADMIN)),
):
    """Export user data as CSV. Only super_admin can export."""
    supabase = get_supabase_admin()

    query = supabase.table("user_profiles").select(
        "id, email, full_name, phone, capital, trading_mode, subscription_status, "
        "total_trades, winning_trades, total_pnl, broker_connected, broker_name, "
        "created_at, last_login"
    )
    if subscription_status:
        query = query.eq("subscription_status", subscription_status)

    result = query.order("created_at", desc=True).execute()

    output = io.StringIO()
    if result.data:
        writer = csv.DictWriter(output, fieldnames=result.data[0].keys())
        writer.writeheader()
        writer.writerows(result.data)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=users_export_{date.today().isoformat()}.csv"
        },
    )
