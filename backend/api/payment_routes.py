"""
================================================================================
QUANT X - RAZORPAY PAYMENT ROUTES
================================================================================
Complete payment integration with:
- Order creation
- Payment verification
- Webhook handlers (success/fail/refund)
- Idempotent subscription lifecycle
================================================================================
"""

import hmac
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel
from supabase import Client
import razorpay

from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["Payments"])


def _get_current_user_dep():
    """Lazy import to avoid circular dependency with app.py."""
    from ..api.app import get_current_user
    return get_current_user


# ============================================================================
# SCHEMAS
# ============================================================================

class CreateSubscriptionOrder(BaseModel):
    plan_id: str
    billing_period: str  # monthly, quarterly, yearly


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_razorpay_client() -> razorpay.Client:
    """Get Razorpay client instance"""
    if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Payment service not configured"
        )
    return razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))


def get_supabase_admin() -> Client:
    """Get Supabase admin client"""
    from ..api.app import get_supabase_admin as _get_admin
    return _get_admin()


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify Razorpay webhook signature.

    Args:
        payload: Raw request body bytes
        signature: X-Razorpay-Signature header value
        secret: Razorpay webhook secret

    Returns:
        True if signature is valid
    """
    expected = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def calculate_subscription_end(billing_period: str, start_date: datetime = None) -> datetime:
    """Calculate subscription end date based on billing period"""
    start = start_date or datetime.utcnow()

    if billing_period == "monthly":
        return start + timedelta(days=30)
    elif billing_period == "quarterly":
        return start + timedelta(days=90)
    elif billing_period == "yearly":
        return start + timedelta(days=365)
    else:
        return start + timedelta(days=30)  # Default to monthly

# ============================================================================
# IDEMPOTENCY HELPERS
# ============================================================================


async def get_or_create_payment(
    supabase: Client,
    user_id: str,
    razorpay_order_id: str,
    plan_id: str,
    billing_period: str,
    amount: int
) -> Dict:
    """
    Idempotent payment creation — uses upsert with razorpay_order_id UNIQUE
    constraint to prevent race conditions from concurrent requests.
    """
    payment = {
        "user_id": user_id,
        "razorpay_order_id": razorpay_order_id,
        "amount": amount,
        "plan_id": plan_id,
        "billing_period": billing_period,
        "status": "pending"
    }

    # Upsert: insert or return existing row (relies on UNIQUE index on razorpay_order_id)
    result = supabase.table("payments").upsert(
        payment, on_conflict="razorpay_order_id"
    ).execute()
    return result.data[0] if result.data else payment


async def process_successful_payment(
    supabase: Client,
    payment_id: str,
    razorpay_payment_id: str,
    razorpay_signature: str
) -> bool:
    """
    Idempotent payment success processing.
    Returns False if already processed, True if newly processed.
    """
    # Get payment record
    payment = supabase.table("payments").select("*").eq(
        "id", payment_id
    ).single().execute()

    if not payment.data:
        logger.error(f"Payment not found: {payment_id}")
        return False

    # Check if already processed (idempotency)
    if payment.data["status"] == "completed":
        logger.info(f"Payment {payment_id} already completed - skipping")
        return False

    # Update payment record
    supabase.table("payments").update({
        "razorpay_payment_id": razorpay_payment_id,
        "razorpay_signature": razorpay_signature,
        "status": "completed",
        "completed_at": datetime.utcnow().isoformat()
    }).eq("id", payment_id).execute()

    # Activate subscription
    user_id = payment.data["user_id"]
    plan_id = payment.data["plan_id"]
    billing_period = payment.data["billing_period"]

    subscription_end = calculate_subscription_end(billing_period)

    # PR 26 — resolve plan name → tier string before the profile write so we
    # only do one user_profiles update instead of two.
    previous_tier = _current_user_tier(supabase, user_id)
    new_tier = _resolve_tier_from_plan(supabase, plan_id) or previous_tier

    # PR 44 — consume accumulated referral credit (N12). Extends
    # subscription_end by 30 days per credited month. The consume helper
    # zeroes the column, so a retry of this payment won't re-apply.
    referral_bonus_days = 0
    try:
        from ..platform.referrals import consume_referral_credit
        bonus_months = consume_referral_credit(user_id, supabase_client=supabase)
        if bonus_months > 0:
            referral_bonus_days = bonus_months * 30
            subscription_end = subscription_end + timedelta(days=referral_bonus_days)
            logger.info(
                "Payment %s — referral credit consumed: user=%s +%d months (+%d days)",
                payment_id, user_id, bonus_months, referral_bonus_days,
            )
    except Exception as ref_exc:
        logger.warning("referral credit consume skipped for %s: %s", user_id, ref_exc)

    supabase.table("user_profiles").update({
        "subscription_plan_id": plan_id,
        "subscription_status": "active",
        "subscription_start": datetime.utcnow().isoformat(),
        "subscription_end": subscription_end.isoformat(),
        "tier": new_tier,
    }).eq("id", user_id).execute()

    logger.info(
        "Payment %s processed — subscription activated for user %s (%s → %s, +%d bonus days)",
        payment_id, user_id, previous_tier, new_tier, referral_bonus_days,
    )

    # PR 26 — notify the rest of the stack that the tier changed.
    if new_tier != previous_tier:
        await _emit_tier_change(
            supabase=supabase,
            user_id=user_id,
            previous=previous_tier,
            new=new_tier,
            plan_id=plan_id,
            payment_id=payment_id,
            direction="upgrade",
        )

        # PR 42 — N12 referral reward. On first paid upgrade (free → pro/elite)
        # both the referrer and this user get +1 month Pro credit. Fire once;
        # the resolver guards against double-reward via status=rewarded.
        if previous_tier == "free" and new_tier in {"pro", "elite"}:
            try:
                from ..platform.referrals import credit_referral_on_first_paid
                credit_referral_on_first_paid(user_id, supabase_client=supabase)
            except Exception as ref_exc:
                logger.warning("referral credit skipped for %s: %s", user_id, ref_exc)

    return True


async def process_failed_payment(
    supabase: Client,
    razorpay_order_id: str,
    failure_reason: str
) -> bool:
    """
    Idempotent payment failure processing.
    """
    # Get payment by order ID
    payment = supabase.table("payments").select("*").eq(
        "razorpay_order_id", razorpay_order_id
    ).single().execute()

    if not payment.data:
        logger.error(f"Payment not found for order: {razorpay_order_id}")
        return False

    # Check if already processed
    if payment.data["status"] in ["completed", "failed", "refunded"]:
        logger.info(f"Payment {payment.data['id']} already in final state - skipping")
        return False

    # Update payment record
    supabase.table("payments").update({
        "status": "failed",
        "failure_reason": failure_reason
    }).eq("id", payment.data["id"]).execute()

    logger.info(f"Payment {payment.data['id']} marked as failed: {failure_reason}")

    return True


async def process_refund(
    supabase: Client,
    razorpay_payment_id: str,
    refund_amount: int,
    refund_id: str
) -> bool:
    """
    Idempotent refund processing.
    """
    # Get payment by Razorpay payment ID
    payment = supabase.table("payments").select("*").eq(
        "razorpay_payment_id", razorpay_payment_id
    ).single().execute()

    if not payment.data:
        logger.error(f"Payment not found: {razorpay_payment_id}")
        return False

    # Check if already refunded
    if payment.data["status"] == "refunded":
        logger.info(f"Payment {payment.data['id']} already refunded - skipping")
        return False

    # Update payment record
    supabase.table("payments").update({
        "status": "refunded",
        "refund_id": refund_id,
        "refund_amount": refund_amount
    }).eq("id", payment.data["id"]).execute()

    # Revert subscription if full refund
    if refund_amount >= payment.data["amount"]:
        user_id = payment.data["user_id"]

        # Get free plan
        free_plan = supabase.table("subscription_plans").select("id").eq(
            "name", "free"
        ).single().execute()

        previous_tier = _current_user_tier(supabase, user_id)

        supabase.table("user_profiles").update({
            "subscription_plan_id": free_plan.data["id"] if free_plan.data else None,
            "subscription_status": "free",
            "subscription_end": None,
            "tier": "free",
        }).eq("id", user_id).execute()

        logger.info(f"Full refund processed - subscription reverted for user {user_id}")

        # PR 26 — emit tier_downgraded so Copilot caps + feature gates flip
        # back to Free immediately.
        if previous_tier != "free":
            await _emit_tier_change(
                supabase=supabase,
                user_id=user_id,
                previous=previous_tier,
                new="free",
                plan_id=free_plan.data["id"] if free_plan.data else None,
                payment_id=payment.data["id"],
                direction="downgrade",
            )

    return True


# ============================================================================
# PR 26 — tier resolver + change emitter (shared by success + refund paths)
# ============================================================================


def _resolve_tier_from_plan(supabase: Client, plan_id: str) -> Optional[str]:
    """Map ``subscription_plans.name`` → Tier enum value.

    Canonical plan names in the DB: ``free`` / ``pro`` / ``elite``. Returns
    None when the plan lookup fails (caller falls back to previous tier
    so we never accidentally downgrade a paying user on a DB blip).
    """
    try:
        row = (
            supabase.table("subscription_plans")
            .select("name")
            .eq("id", plan_id)
            .limit(1)
            .execute()
        )
        rows = row.data or []
        if not rows:
            return None
        name = str(rows[0].get("name") or "").strip().lower()
        if name in ("free", "pro", "elite"):
            return name
        # Legacy plan names — normalize.
        if "elite" in name or "premium" in name:
            return "elite"
        if "pro" in name or "plus" in name:
            return "pro"
        return "free"
    except Exception as exc:
        logger.warning("plan→tier lookup failed for %s: %s", plan_id, exc)
        return None


def _current_user_tier(supabase: Client, user_id: str) -> str:
    """Read the user's current tier column; default to ``free`` on any error."""
    try:
        row = (
            supabase.table("user_profiles")
            .select("tier")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        rows = row.data or []
        return str((rows[0] or {}).get("tier") or "free").strip().lower() or "free"
    except Exception:
        return "free"


async def _emit_tier_change(
    *,
    supabase: Client,
    user_id: str,
    previous: str,
    new: str,
    plan_id: Optional[str],
    payment_id: str,
    direction: str,
) -> None:
    """Fan-out side-effects when a user's tier flips.

    - Invalidate the in-process tier cache (PR 14) so the next gate check
      sees the new tier without the 60s TTL wait.
    - Emit a ``tier_upgraded`` / ``tier_downgraded`` PostHog event with
      the payment_id + plan_id so conversion-funnel dashboards stay honest.
    - Refresh PostHog user context so distinct_id properties carry the
      new tier for every subsequent event.
    - Broadcast the ``tier_upgraded`` WS event via PR 13 event_bus so
      any active frontend session re-fetches ``/api/user/tier`` and
      unlocks the newly-available features live.
    """
    # 1) cache bust
    try:
        from ..core.tiers import invalidate_user_tier_cache
        invalidate_user_tier_cache(user_id)
    except Exception as exc:
        logger.debug("tier cache invalidate skipped: %s", exc)

    # 2) PostHog analytics + identify
    try:
        from ..observability import EventName, set_user_context, track
        track(
            EventName.TIER_UPGRADED if direction == "upgrade" else EventName.TIER_DOWNGRADED,
            user_id,
            {
                "previous_tier": previous,
                "new_tier": new,
                "plan_id": plan_id,
                "payment_id": payment_id,
            },
        )
        set_user_context(user_id, {"tier": new})
    except Exception as exc:
        logger.debug("PostHog tier-change emit skipped: %s", exc)

    # 3) Real-time event bus so frontend hides tier-gate locks immediately
    try:
        from ..platform.events import MessageType, emit_event
        await emit_event(
            MessageType.NOTIFICATION,
            {
                "type": f"tier_{direction}d",
                "title": f"Tier {direction}d to {new.title()}",
                "message": (
                    "Your plan just upgraded. New features unlocked — reload the page to see them."
                    if direction == "upgrade"
                    else "Your plan downgraded to Free after refund."
                ),
                "previous_tier": previous,
                "new_tier": new,
                "priority": "high",
            },
            user_id=user_id,
        )
    except Exception as exc:
        logger.debug("EventBus tier-change emit skipped: %s", exc)

# ============================================================================
# API ENDPOINTS
# ============================================================================


@router.post("/create-order")
async def create_order(
    data: CreateSubscriptionOrder,
    user=Depends(_get_current_user_dep()),
):
    """
    Create a Razorpay order for subscription payment.
    """

    supabase = get_supabase_admin()
    rzp = get_razorpay_client()

    # Get plan details
    plan = supabase.table("subscription_plans").select("*").eq(
        "id", data.plan_id
    ).single().execute()

    if not plan.data:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Calculate amount based on billing period
    if data.billing_period == "monthly":
        amount = plan.data["price_monthly"]
    elif data.billing_period == "quarterly":
        amount = plan.data["price_quarterly"] or plan.data["price_monthly"] * 3
    elif data.billing_period == "yearly":
        amount = plan.data["price_yearly"] or plan.data["price_monthly"] * 12
    else:
        raise HTTPException(status_code=400, detail="Invalid billing period")

    if amount <= 0:
        raise HTTPException(status_code=400, detail="Cannot create order for free plan")

    # Create Razorpay order
    order_data = {
        "amount": amount,
        "currency": "INR",
        "receipt": f"order_{user.id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "notes": {
            "user_id": user.id,
            "plan_id": data.plan_id,
            "plan_name": plan.data["name"],
            "billing_period": data.billing_period
        }
    }

    try:
        rzp_order = rzp.order.create(order_data)
    except Exception as e:
        logger.error(f"Razorpay order creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create payment order")

    # Save payment record (idempotent)
    await get_or_create_payment(
        supabase,
        user.id,
        rzp_order["id"],
        data.plan_id,
        data.billing_period,
        amount
    )

    return {
        "order_id": rzp_order["id"],
        "amount": amount,
        "currency": "INR",
        "key_id": settings.RAZORPAY_KEY_ID,
        "plan_name": plan.data["display_name"],
        "billing_period": data.billing_period
    }


@router.post("/verify")
async def verify_payment(
    data: VerifyPaymentRequest,
    user=Depends(_get_current_user_dep()),
):
    """
    Verify Razorpay payment signature and activate subscription.
    """

    supabase = get_supabase_admin()

    # Verify signature
    message = f"{data.razorpay_order_id}|{data.razorpay_payment_id}"
    expected_signature = hmac.new(
        settings.RAZORPAY_KEY_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, data.razorpay_signature):
        logger.warning(f"Invalid payment signature for order {data.razorpay_order_id}")
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    # Get payment record
    payment = supabase.table("payments").select("*").eq(
        "razorpay_order_id", data.razorpay_order_id
    ).single().execute()

    if not payment.data:
        raise HTTPException(status_code=404, detail="Payment not found")

    # Verify user owns this payment
    if payment.data["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Process payment (idempotent)
    was_processed = await process_successful_payment(
        supabase,
        payment.data["id"],
        data.razorpay_payment_id,
        data.razorpay_signature
    )

    if was_processed:
        return {
            "success": True,
            "message": "Payment verified and subscription activated"
        }
    else:
        return {
            "success": True,
            "message": "Payment already processed"
        }


@router.get("/history")
async def get_payment_history(user=Depends(_get_current_user_dep())):
    """Get the current user's payment history with plan display names."""
    try:
        supabase = get_supabase_admin()
        result = (
            supabase.table("payments")
            .select("*, subscription_plans(display_name)")
            .eq("user_id", user.id)
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        return {"payments": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBHOOK HANDLERS
# ============================================================================

@router.post("/webhook")
async def handle_webhook(
    request: Request,
    x_razorpay_signature: str = Header(None)
):
    """
    Handle Razorpay webhook events.

    Supported events:
    - payment.captured: Payment successful
    - payment.failed: Payment failed
    - refund.created: Refund initiated
    - refund.processed: Refund completed
    """
    # Get raw body for signature verification
    body = await request.body()

    # Phase 1.7 audit fix #1.7 — RAZORPAY_WEBHOOK_SECRET is a SEPARATE
    # value configured in the Razorpay dashboard for webhook signing.
    # It is NOT the same as RAZORPAY_KEY_SECRET (the API key secret
    # used for order creation/refunds). The legacy fallback to
    # KEY_SECRET would (a) silently fail signature verification in
    # production because Razorpay signs with WEBHOOK_SECRET, or worse
    # (b) accept attacker-crafted webhooks if the attacker ever learned
    # the API key secret. Refuse to accept any webhook until
    # WEBHOOK_SECRET is set.
    webhook_secret = settings.RAZORPAY_WEBHOOK_SECRET
    if not webhook_secret:
        logger.error(
            "RAZORPAY_WEBHOOK_SECRET is not configured — refusing to accept "
            "webhook deliveries. Set this env var to the value shown in the "
            "Razorpay dashboard at Settings → Webhooks (do NOT reuse KEY_SECRET)."
        )
        raise HTTPException(status_code=503, detail="webhook_secret_not_configured")

    if not x_razorpay_signature:
        logger.warning("Webhook request missing X-Razorpay-Signature header")
        raise HTTPException(status_code=400, detail="Missing signature")

    if not verify_webhook_signature(body, x_razorpay_signature, webhook_secret):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Parse payload
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = payload.get("event")
    event_data = payload.get("payload", {}).get("payment", {}).get("entity", {})

    logger.info(f"Webhook received: {event}")

    supabase = get_supabase_admin()

    # Handle different event types
    if event == "payment.captured":
        # Payment was successful
        order_id = event_data.get("order_id")
        payment_id = event_data.get("id")

        if order_id and payment_id:
            # Get payment record
            payment = supabase.table("payments").select("*").eq(
                "razorpay_order_id", order_id
            ).single().execute()

            if payment.data:
                await process_successful_payment(
                    supabase,
                    payment.data["id"],
                    payment_id,
                    "webhook_verified"
                )

    elif event == "payment.failed":
        # Payment failed
        order_id = event_data.get("order_id")
        error_reason = event_data.get("error_description", "Payment failed")

        if order_id:
            await process_failed_payment(supabase, order_id, error_reason)

    elif event in ["refund.created", "refund.processed"]:
        # Refund event
        refund_data = payload.get("payload", {}).get("refund", {}).get("entity", {})
        payment_id = refund_data.get("payment_id")
        refund_id = refund_data.get("id")
        refund_amount = refund_data.get("amount", 0)

        if payment_id and refund_id:
            await process_refund(supabase, payment_id, refund_amount, refund_id)

    else:
        logger.info(f"Unhandled webhook event: {event}")

    # Always return 200 to acknowledge receipt
    return {"status": "ok"}

# ============================================================================
# REFUNDS — there is no programmatic refund endpoint.
# ============================================================================
# Locked product rule (2026-04-19): no user-facing refund surface, no admin
# refund tool. The webhook handler above STILL processes ``refund.created`` /
# ``refund.processed`` events from Razorpay so that if a refund is manually
# issued from the Razorpay dashboard for legal/regulatory reasons, our DB
# stays in sync. But there is no API path to *initiate* a refund from this
# service — the only valid initiation is the Razorpay dashboard, with full
# audit trail on Razorpay's side.

# ============================================================================
# SUBSCRIPTION STATUS
# ============================================================================


@router.get("/subscription-status")
async def get_subscription_status(user=Depends(_get_current_user_dep())):
    """
    Get current user's subscription status.
    """

    supabase = get_supabase_admin()

    profile = supabase.table("user_profiles").select(
        "subscription_status, subscription_start, subscription_end, "
        "subscription_plans(name, display_name, price_monthly)"
    ).eq("id", user.id).single().execute()

    if not profile.data:
        raise HTTPException(status_code=404, detail="Profile not found")

    data = profile.data
    plan = data.get("subscription_plans") or {}

    # Calculate days remaining
    days_remaining = None
    if data.get("subscription_end"):
        end_date = datetime.fromisoformat(data["subscription_end"].replace("Z", "+00:00"))
        days_remaining = max(0, (end_date - datetime.utcnow()).days)

    return {
        "status": data.get("subscription_status", "free"),
        "plan_name": plan.get("display_name", "Free"),
        "plan_code": plan.get("name", "free"),
        "start_date": data.get("subscription_start"),
        "end_date": data.get("subscription_end"),
        "days_remaining": days_remaining,
        "is_active": data.get("subscription_status") in ["active", "trial"]
    }
