"""
================================================================================
TELEGRAM ROUTES — onboarding connect flow + bot webhook (PR 55)
================================================================================
Lets a signed-in user link their Telegram account so they can receive
the free daily digest and per-event alerts. Three surfaces:

    POST /api/telegram/link/start      — authed, issues link token
    GET  /api/telegram/link/status     — authed, polled by onboarding page
    POST /api/telegram/link/disconnect — authed, un-links chat_id
    POST /api/telegram/webhook/{secret} — unauthed, bot-to-server callback

Linking protocol:
    1. Frontend hits ``/link/start`` → we generate a 16-char URL-safe
       token (``secrets.token_urlsafe(12)``) with 15-minute expiry and
       stash it on ``user_profiles.telegram_link_token``.
    2. Response includes ``deep_link``:
           ``https://t.me/<bot_username>?start=<token>``
       Frontend renders the "Open Telegram" button using that URL.
    3. User taps → Telegram opens the bot chat → user taps Start →
       Telegram sends us a webhook with ``/start <token>`` as the text.
    4. Webhook handler looks up the user by token, writes
       ``telegram_chat_id`` + flips ``telegram_connected = true``, clears
       the token, and replies to the chat with a confirmation message.
    5. Onboarding page polls ``/link/status`` every 2s and advances when
       ``connected`` flips to true.

The webhook URL includes ``TELEGRAM_WEBHOOK_SECRET`` in the path AND we
also verify the ``X-Telegram-Bot-Api-Secret-Token`` header (Telegram
echoes the secret we pass to ``setWebhook``). Both layers must match
before we even parse the body.
================================================================================
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Request
from pydantic import BaseModel

from ..core.config import settings
from ..core.database import get_supabase_admin
from ..core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])


# ============================================================================
# Models
# ============================================================================


class LinkStartResponse(BaseModel):
    token: str
    bot_username: Optional[str]
    deep_link: Optional[str]
    expires_at: str


class LinkStatusResponse(BaseModel):
    connected: bool
    chat_id: Optional[str] = None
    linked_at: Optional[str] = None


class DisconnectResponse(BaseModel):
    connected: bool


class WebhookAck(BaseModel):
    ok: bool = True


# ============================================================================
# Helpers
# ============================================================================


_TOKEN_TTL_MINUTES = 15


def _user_id_from(user: Any) -> str:
    uid = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    if not uid:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return str(uid)


def _deep_link(token: str) -> Optional[str]:
    bot = settings.TELEGRAM_BOT_USERNAME
    if not bot:
        return None
    bot_handle = bot.lstrip("@")
    return f"https://t.me/{bot_handle}?start={token}"


async def _reply_to_chat(chat_id: int | str, text: str) -> None:
    """Best-effort sendMessage — never raises."""
    if not settings.TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as exc:
        logger.debug("telegram reply failed: %s", exc)


# ============================================================================
# Authed endpoints — start / status / disconnect
# ============================================================================


@router.post("/link/start", response_model=LinkStartResponse)
async def link_start(user: Any = Depends(get_current_user)) -> LinkStartResponse:
    """Mint (or rotate) a link token for the current user. Overwrites any
    prior pending token for the same user so the onboarding page can be
    retried safely."""
    uid = _user_id_from(user)

    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_BOT_USERNAME:
        raise HTTPException(status_code=503, detail="telegram_bot_not_configured")

    token = secrets.token_urlsafe(12)
    expires = datetime.now(timezone.utc) + timedelta(minutes=_TOKEN_TTL_MINUTES)

    sb = get_supabase_admin()
    try:
        sb.table("user_profiles").update({
            "telegram_link_token": token,
            "telegram_link_expires_at": expires.isoformat(),
        }).eq("id", uid).execute()
    except Exception as exc:
        logger.error("telegram link token persist failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")

    return LinkStartResponse(
        token=token,
        bot_username=settings.TELEGRAM_BOT_USERNAME,
        deep_link=_deep_link(token),
        expires_at=expires.isoformat(),
    )


@router.get("/link/status", response_model=LinkStatusResponse)
async def link_status(user: Any = Depends(get_current_user)) -> LinkStatusResponse:
    """Polled by the onboarding page. Returns whether the webhook has
    already consumed the token and written a chat_id."""
    uid = _user_id_from(user)
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_profiles")
            .select("telegram_chat_id, telegram_connected, telegram_linked_at")
            .eq("id", uid)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        # ``telegram_linked_at`` is optional — older deployments may not
        # have the column. Fall back to a narrower select in that case.
        if "column" in str(exc).lower() and "telegram_linked_at" in str(exc):
            rows = (
                sb.table("user_profiles")
                .select("telegram_chat_id, telegram_connected")
                .eq("id", uid)
                .limit(1)
                .execute()
            )
        else:
            logger.error("telegram status lookup failed: %s", exc)
            raise HTTPException(status_code=500, detail="lookup_failed")

    row = (rows.data or [None])[0] or {}
    connected = bool(row.get("telegram_connected") and row.get("telegram_chat_id"))
    return LinkStatusResponse(
        connected=connected,
        chat_id=str(row["telegram_chat_id"]) if connected else None,
        linked_at=str(row["telegram_linked_at"]) if row.get("telegram_linked_at") else None,
    )


@router.post("/link/disconnect", response_model=DisconnectResponse)
async def link_disconnect(user: Any = Depends(get_current_user)) -> DisconnectResponse:
    """Un-link the user's Telegram chat. Clears chat_id + connected flag
    + any pending link token. Does NOT revoke the bot on Telegram's side —
    the user can type /stop to mute the bot anytime."""
    uid = _user_id_from(user)
    sb = get_supabase_admin()
    try:
        sb.table("user_profiles").update({
            "telegram_chat_id": None,
            "telegram_connected": False,
            "telegram_link_token": None,
            "telegram_link_expires_at": None,
        }).eq("id", uid).execute()
    except Exception as exc:
        logger.error("telegram disconnect failed: %s", exc)
        raise HTTPException(status_code=500, detail="persist_failed")
    return DisconnectResponse(connected=False)


# ============================================================================
# Bot-facing webhook
# ============================================================================


@router.post("/webhook/{secret}", response_model=WebhookAck)
async def telegram_webhook(
    secret: str = Path(..., min_length=8, max_length=128),
    request: Request = None,  # type: ignore[assignment]
    x_telegram_bot_api_secret_token: Optional[str] = Header(
        default=None, alias="X-Telegram-Bot-Api-Secret-Token"
    ),
) -> WebhookAck:
    """Telegram → us. Two-layer secret check: URL path segment AND
    X-Telegram-Bot-Api-Secret-Token header (both must match
    ``TELEGRAM_WEBHOOK_SECRET``). Always returns 200 OK once auth
    passes; Telegram retries aggressively on non-2xx."""
    expected = settings.TELEGRAM_WEBHOOK_SECRET
    if not expected:
        # Webhook not configured — reject everything. Don't 500; that
        # would invite Telegram retries. 404 hides the endpoint.
        raise HTTPException(status_code=404, detail="not_found")
    if not secrets.compare_digest(secret, expected):
        raise HTTPException(status_code=404, detail="not_found")
    if x_telegram_bot_api_secret_token is not None and not secrets.compare_digest(
        x_telegram_bot_api_secret_token, expected
    ):
        # Header was sent but doesn't match — treat as fake.
        raise HTTPException(status_code=404, detail="not_found")

    try:
        body = await request.json()
    except Exception:
        return WebhookAck()

    await _handle_update(body)
    return WebhookAck()


async def _handle_update(update: Dict[str, Any]) -> None:
    """Dispatch a single Telegram update. Only /start <token> is a
    link trigger; other commands get a polite no-op reply so the user
    isn't confused by silence."""
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    text = (msg.get("text") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return

    if text.startswith("/start"):
        parts = text.split(maxsplit=1)
        token = parts[1].strip() if len(parts) == 2 else ""
        if not token:
            await _reply_to_chat(
                chat_id,
                "Welcome to Quant X. Finish the link by tapping the "
                "'Connect Telegram' button from the onboarding page — "
                "that'll open this chat with the right code."
            )
            return
        await _consume_token(token, chat_id, msg.get("from") or {})
        return

    if text.startswith("/stop"):
        await _reply_to_chat(
            chat_id,
            "You can pause alerts anytime from Settings → Notifications "
            "on the app. To fully un-link, use the Disconnect button there."
        )
        return

    # Unknown — gentle nudge so the chat doesn't feel broken.
    await _reply_to_chat(
        chat_id,
        "Quant X bot — I'm here for signal briefs and alert delivery. "
        "Manage preferences from Settings → Notifications on the app."
    )


async def _consume_token(token: str, chat_id: int | str, from_user: Dict[str, Any]) -> None:
    """Look up a pending link token, flip the profile to connected, and
    reply to the chat. Handles re-link collisions by un-linking any
    prior owner of the same chat_id first."""
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("user_profiles")
            .select("id, telegram_link_expires_at")
            .eq("telegram_link_token", token)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("telegram token lookup failed: %s", exc)
        await _reply_to_chat(chat_id, "Something went wrong on our side. Try the link button again.")
        return

    row = (rows.data or [None])[0]
    if not row:
        await _reply_to_chat(
            chat_id,
            "This link has already been used or doesn't belong to a "
            "pending connection. Re-open the onboarding page and tap "
            "'Connect Telegram' to try again."
        )
        return

    expires_at = row.get("telegram_link_expires_at")
    if expires_at:
        try:
            # Supabase returns ISO string; parse defensively.
            exp_dt = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if exp_dt < datetime.now(timezone.utc):
                # Expired — null the stale token so the user can retry.
                try:
                    sb.table("user_profiles").update({
                        "telegram_link_token": None,
                        "telegram_link_expires_at": None,
                    }).eq("id", row["id"]).execute()
                except Exception:
                    pass
                await _reply_to_chat(
                    chat_id,
                    "This connection code has expired. Head back to the "
                    "app, tap 'Connect Telegram' again, and you'll get a "
                    "fresh code good for 15 minutes."
                )
                return
        except Exception:
            pass  # Bad timestamp → be lenient, accept the link.

    user_id = str(row["id"])
    chat_id_str = str(chat_id)

    # If this chat_id is already linked to a different user, un-link that
    # user first. Re-linking on a new device is a common flow; don't let
    # the old profile silently keep receiving the new user's alerts.
    try:
        dup = (
            sb.table("user_profiles")
            .select("id")
            .eq("telegram_chat_id", chat_id_str)
            .neq("id", user_id)
            .execute()
        )
        for other in dup.data or []:
            sb.table("user_profiles").update({
                "telegram_chat_id": None,
                "telegram_connected": False,
            }).eq("id", other["id"]).execute()
    except Exception as exc:
        logger.debug("telegram re-link cleanup skipped: %s", exc)

    # Flip the profile to connected. ``telegram_linked_at`` is optional —
    # omit it from the payload if we've never added that column; the
    # surrounding update works regardless.
    payload: Dict[str, Any] = {
        "telegram_chat_id": chat_id_str,
        "telegram_connected": True,
        "telegram_link_token": None,
        "telegram_link_expires_at": None,
    }
    try:
        sb.table("user_profiles").update(payload).eq("id", user_id).execute()
    except Exception as exc:
        logger.error("telegram link persist failed: %s", exc)
        await _reply_to_chat(chat_id, "Couldn't save the link. Please try again in a minute.")
        return

    # Telemetry — activation-funnel metric.
    try:
        from ..observability import EventName, track
        track(EventName.TELEGRAM_CONNECTED, user_id, {
            "chat_id_hashed": _hash_chat_id(chat_id_str),
            "from_username": (from_user.get("username") or "")[:32],
        })
    except Exception:
        pass

    first_name = (from_user.get("first_name") or "").strip()
    greeting = f"Hey {first_name}, " if first_name else "Hey, "
    await _reply_to_chat(
        chat_id,
        greeting + "your Quant X account is now linked. You'll get the "
        "daily digest + any alerts you enable from Settings → Notifications. "
        "Type /stop anytime to pause."
    )


def _hash_chat_id(chat_id: str) -> str:
    """Short non-reversible digest — we never ship raw chat_ids to
    analytics."""
    import hashlib
    return hashlib.sha256(chat_id.encode()).hexdigest()[:16]


__all__ = ["router"]
