"""
================================================================================
ALERTS STUDIO ROUTES — N11 per-event channel routing (PR 40)
================================================================================
Endpoints (all gated by ``RequireFeature("alert_studio")`` = Pro):

    GET   /api/alerts/preferences     — event×channel matrix + channel status
    PATCH /api/alerts/preferences     — toggle one cell or bulk-update
    POST  /api/alerts/test            — fire a test notification on one channel

The realtime bus (``services/realtime.py``) and any future feature
emitter should consult ``channels_for_event(user_id, event)`` from
``backend.services.alert_prefs`` before calling the underlying push /
telegram / whatsapp / email client.
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from ..core.database import get_supabase_admin
from ..core.tiers import UserTier
from ..middleware.tier_gate import RequireFeature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


# ============================================================================
# Event / channel catalogue — source of truth
# ============================================================================


EVENTS: List[Dict[str, Any]] = [
    {"key": "new_signal", "label": "New signal",
     "description": "A fresh Forecast or Intraday signal is published"},
    {"key": "signal_triggered", "label": "Signal triggered",
     "description": "Live price crosses the signal entry level"},
    {"key": "target_hit", "label": "Target hit",
     "description": "Your open position closes at target"},
    {"key": "sl_hit", "label": "Stop-loss hit",
     "description": "Your open position closes at stop-loss"},
    {"key": "regime_change", "label": "Regime change",
     "description": "Regime transitions bull / sideways / bear"},
    {"key": "debate_completed", "label": "Counterpoint ready",
     "description": "Bull/Bear debate finishes on an active signal"},
    {"key": "earnings_upcoming", "label": "Earnings ahead",
     "description": "EarningsScout flags an announcement in the next 14 days"},
    {"key": "weekly_review", "label": "Weekly review",
     "description": "Your Sunday personal review is ready"},
    {"key": "auto_trade_executed", "label": "Auto-trade fired",
     "description": "AutoPilot executed an order on your broker"},
    {"key": "price_alert", "label": "Price alert",
     "description": "A price crosses a level you configured in watchlist"},
    # O.7 (2026-05-31) — F&O-specific alert events
    {"key": "fno_max_pain_shift", "label": "F&O · Max Pain shifted",
     "description": "Today's Max Pain moved >1% — pull-toward-MP signal active"},
    {"key": "fno_oi_spike", "label": "F&O · OI Spike",
     "description": "Single F&O symbol's OI changed ≥20% — institutional footprint"},
    {"key": "fno_position_unprotected", "label": "F&O · Position unprotected",
     "description": "Broker did not accept the SL — your position has NO broker-side stop"},
    {"key": "fno_adjustment_recommended", "label": "F&O · Adjustment recommended",
     "description": "Open multi-leg position needs a roll / hedge / defend action"},
    {"key": "fno_vix_regime_change", "label": "F&O · VIX regime change",
     "description": "India VIX crossed a regime band (complacent/normal/elevated/stressed)"},
    {"key": "fno_pcr_extreme", "label": "F&O · PCR extreme",
     "description": "Index PCR hit contrarian-bullish (>1.4) or contrarian-bearish (<0.4)"},
    # Tier 1 catastrophic-loss alerts
    {"key": "portfolio_drawdown", "label": "Portfolio drawdown alert",
     "description": "Account drawdown crossed -5%/-10% — regime context included"},
    {"key": "cron_failed", "label": "AutoPilot cron failed (admin)",
     "description": "Daily 15:50 rebalance hit an error — admin notice"},
]

CHANNELS = ("push", "telegram", "whatsapp", "email")

EVENT_KEYS = {e["key"] for e in EVENTS}


# ============================================================================
# Defaults — mirror the migration default so freshly-loaded rows match
# ============================================================================


DEFAULT_PREFS: Dict[str, Dict[str, bool]] = {
    "new_signal": {"push": True, "telegram": True, "whatsapp": False, "email": False},
    "signal_triggered": {"push": True, "telegram": False, "whatsapp": False, "email": False},
    "target_hit": {"push": True, "telegram": True, "whatsapp": False, "email": True},
    "sl_hit": {"push": True, "telegram": True, "whatsapp": False, "email": True},
    "regime_change": {"push": True, "telegram": True, "whatsapp": False, "email": False},
    "debate_completed": {"push": True, "telegram": False, "whatsapp": False, "email": False},
    "earnings_upcoming": {"push": False, "telegram": True, "whatsapp": False, "email": False},
    "weekly_review": {"push": False, "telegram": False, "whatsapp": True, "email": True},
    "auto_trade_executed": {"push": True, "telegram": True, "whatsapp": False, "email": False},
    "price_alert": {"push": True, "telegram": True, "whatsapp": False, "email": False},
    # O.7 F&O alert defaults — push on by default for the urgent ones
    "fno_max_pain_shift": {"push": True, "telegram": True, "whatsapp": False, "email": False},
    "fno_oi_spike": {"push": True, "telegram": True, "whatsapp": False, "email": False},
    "fno_position_unprotected": {"push": True, "telegram": True, "whatsapp": True, "email": True},
    "fno_adjustment_recommended": {"push": True, "telegram": True, "whatsapp": False, "email": False},
    "fno_vix_regime_change": {"push": True, "telegram": False, "whatsapp": False, "email": False},
    "fno_pcr_extreme": {"push": False, "telegram": True, "whatsapp": False, "email": False},
    "portfolio_drawdown": {"push": True, "telegram": True, "whatsapp": True, "email": True},
    "cron_failed": {"push": False, "telegram": True, "whatsapp": False, "email": True},
}


# ============================================================================
# Pydantic
# ============================================================================


class ChannelStatus(BaseModel):
    channel: str
    connected: bool
    detail: Optional[str] = None          # e.g. telegram handle / whatsapp number


class PreferencesResponse(BaseModel):
    preferences: Dict[str, Dict[str, bool]]
    events: List[Dict[str, Any]]
    channels: List[ChannelStatus]


class CellToggle(BaseModel):
    event: str
    channel: str
    enabled: bool

    @validator("event")
    def _event(cls, v: str) -> str:
        if v not in EVENT_KEYS:
            raise ValueError(f"unknown event: {v}")
        return v

    @validator("channel")
    def _channel(cls, v: str) -> str:
        if v not in CHANNELS:
            raise ValueError(f"unknown channel: {v}")
        return v


class BulkUpdate(BaseModel):
    preferences: Dict[str, Dict[str, bool]] = Field(..., description="Full or partial matrix")

    @validator("preferences")
    def _validate(cls, v: Dict[str, Dict[str, bool]]) -> Dict[str, Dict[str, bool]]:
        for k, row in v.items():
            if k not in EVENT_KEYS:
                raise ValueError(f"unknown event: {k}")
            for ch, val in row.items():
                if ch not in CHANNELS:
                    raise ValueError(f"unknown channel: {ch}")
                if not isinstance(val, bool):
                    raise ValueError(f"{k}.{ch} must be boolean")
        return v


class PatchRequest(BaseModel):
    """Either a single cell toggle OR a bulk matrix update."""
    toggle: Optional[CellToggle] = None
    bulk: Optional[BulkUpdate] = None


class TestRequest(BaseModel):
    channel: str

    @validator("channel")
    def _channel(cls, v: str) -> str:
        if v not in CHANNELS:
            raise ValueError(f"unknown channel: {v}")
        return v


# ============================================================================
# Helpers
# ============================================================================


def _load_profile(user_id: str) -> Dict[str, Any]:
    sb = get_supabase_admin()
    rows = (
        sb.table("user_profiles")
        .select(
            "id, alert_preferences, telegram_chat_id, telegram_connected, "
            "whatsapp_phone, whatsapp_verified, email, push_notifications"
        )
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    if not rows.data:
        raise HTTPException(status_code=404, detail="profile_not_found")
    return rows.data[0]


def _merge_with_defaults(stored: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
    """Fill missing events/channels from DEFAULT_PREFS so the UI always
    renders the full matrix even for rows written pre-migration."""
    stored = stored or {}
    out: Dict[str, Dict[str, bool]] = {}
    for event_key, default_row in DEFAULT_PREFS.items():
        row = dict(default_row)
        if event_key in stored and isinstance(stored[event_key], dict):
            for ch in CHANNELS:
                if ch in stored[event_key]:
                    row[ch] = bool(stored[event_key][ch])
        out[event_key] = row
    return out


def _channel_statuses(profile: Dict[str, Any]) -> List[ChannelStatus]:
    # Rely on client-side subscription check; we don't
    # store push subscription IDs on the profile directly.
    tg_connected = bool(profile.get("telegram_connected") and profile.get("telegram_chat_id"))
    wa_verified = bool(profile.get("whatsapp_verified") and profile.get("whatsapp_phone"))
    email_present = bool(profile.get("email"))

    return [
        ChannelStatus(channel="push", connected=True,
                      detail="Web Push (browser)"),
        ChannelStatus(channel="telegram", connected=tg_connected,
                      detail=("Connected" if tg_connected else "Connect in Settings → Notifications")),
        ChannelStatus(channel="whatsapp", connected=wa_verified,
                      detail=(profile.get("whatsapp_phone") or "Verify your number in Settings")),
        ChannelStatus(channel="email", connected=email_present,
                      detail=(profile.get("email") or "No email on file")),
    ]


# ============================================================================
# Routes
# ============================================================================


@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences(
    user: UserTier = Depends(RequireFeature("alert_studio")),
) -> PreferencesResponse:
    profile = _load_profile(user.user_id)
    return PreferencesResponse(
        preferences=_merge_with_defaults(profile.get("alert_preferences")),
        events=EVENTS,
        channels=_channel_statuses(profile),
    )


@router.patch("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    body: PatchRequest,
    user: UserTier = Depends(RequireFeature("alert_studio")),
) -> PreferencesResponse:
    if body.toggle is None and body.bulk is None:
        raise HTTPException(status_code=422, detail="toggle_or_bulk_required")

    profile = _load_profile(user.user_id)
    current = _merge_with_defaults(profile.get("alert_preferences"))

    if body.toggle:
        current[body.toggle.event][body.toggle.channel] = bool(body.toggle.enabled)
    if body.bulk:
        for event_key, row in body.bulk.preferences.items():
            if event_key not in current:
                continue
            for ch in CHANNELS:
                if ch in row:
                    current[event_key][ch] = bool(row[ch])

    sb = get_supabase_admin()
    try:
        sb.table("user_profiles").update({
            "alert_preferences": current,
        }).eq("id", user.user_id).execute()
    except Exception as exc:
        logger.error("alert_preferences persist failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")

    logger.info(
        "alert_preferences updated user=%s toggle=%s bulk_keys=%s",
        user.user_id,
        body.toggle.dict() if body.toggle else None,
        list(body.bulk.preferences.keys()) if body.bulk else None,
    )

    # Re-read to include any channel-status changes triggered elsewhere.
    return PreferencesResponse(
        preferences=current,
        events=EVENTS,
        channels=_channel_statuses(_load_profile(user.user_id)),
    )


@router.post("/test")
async def test_channel(
    body: TestRequest,
    user: UserTier = Depends(RequireFeature("alert_studio")),
) -> Dict[str, Any]:
    """Fire a best-effort test notification on one channel.
    Returns {delivered, channel, detail} — never raises on delivery
    failure; instead flags the channel as not-delivered."""
    profile = _load_profile(user.user_id)
    title = "Quant X — alert test"
    body_text = (
        f"This is a test alert on your {body.channel} channel. "
        "If you can read this, delivery is working end-to-end."
    )

    delivered = False
    detail = ""

    try:
        if body.channel == "telegram":
            chat_id = profile.get("telegram_chat_id")
            if not chat_id or not profile.get("telegram_connected"):
                detail = "telegram_not_connected"
            else:
                from ..platform.push import PushService
                ps = PushService()
                sent = await ps.send_telegram(chat_id, f"{title}\n{body_text}") if hasattr(
                    ps, "send_telegram"
                ) else False
                delivered = bool(sent)
                detail = "sent" if sent else "send_failed"

        elif body.channel == "push":
            try:
                from ..platform.push import PushService
                ps = PushService()
                sent = await ps.send_web_push_to_user(user.user_id, title, body_text) if hasattr(
                    ps, "send_web_push_to_user"
                ) else False
                delivered = bool(sent)
                detail = "sent" if sent else "push_service_unavailable"
            except Exception as exc:
                detail = f"push_unavailable: {exc}"

        elif body.channel == "email":
            email = profile.get("email")
            if not email:
                detail = "no_email_on_file"
            else:
                try:
                    from ..platform.push import PushService
                    ps = PushService()
                    sent = await ps.send_email(email, title, body_text) if hasattr(
                        ps, "send_email"
                    ) else False
                    delivered = bool(sent)
                    detail = "sent" if sent else "email_send_failed"
                except Exception as exc:
                    detail = f"email_unavailable: {exc}"

        elif body.channel == "whatsapp":
            if not profile.get("whatsapp_verified") or not profile.get("whatsapp_phone"):
                detail = "whatsapp_not_verified"
            else:
                # PR 64 — route through the provider-agnostic WhatsApp
                # service that backs the F12 digest. `is_configured()`
                # returns False when Gupshup/Meta creds aren't set yet,
                # and `send_text` never raises — so this branch is safe
                # to call even before business-verification is complete.
                from ..platform import whatsapp as whatsapp_service
                if not whatsapp_service.is_configured():
                    detail = "whatsapp_provider_pending"
                else:
                    sent = await whatsapp_service.send_text(
                        profile["whatsapp_phone"],
                        f"{title}\n{body_text}",
                        template=None,
                    )
                    delivered = bool(sent)
                    detail = "sent" if sent else "whatsapp_send_failed"

    except Exception as exc:
        logger.warning("alert test failed ch=%s: %s", body.channel, exc)
        detail = f"error: {exc}"

    return {"delivered": delivered, "channel": body.channel, "detail": detail}


__all__ = ["router"]
