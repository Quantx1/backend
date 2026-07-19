"""
WhatsApp delivery service — F12 Pro digest channel (PR 60).

Two providers are wired: **Gupshup** (default — India-native BSP, faster
onboarding for NSE-focused products) and **Meta Cloud API** (fallback).
Select via ``WHATSAPP_PROVIDER=gupshup|meta``.

When no provider is configured, ``is_configured()`` returns False and
every send path becomes a no-op logged at DEBUG. This is the safe
default — delivery stays dormant until ops has finished the business-
verification dance with Meta/Gupshup, but the rest of the feature
(phone capture, OTP flow, opt-in toggles, scheduler job wiring) can
ship and be tested end-to-end with real users in a controlled manner.

The service exposes two high-level calls:

    send_otp(phone, code, ttl_minutes) — kickoff verification.
    send_text(phone, body, template?)  — generic business-initiated send.

Both accept an E.164 phone. Both always return a bool (True = accepted
by the BSP / API, False = anything else) and never raise — callers
should never have a delivery failure cascade into a user-facing error.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 10.0


def _provider() -> Optional[str]:
    p = (settings.WHATSAPP_PROVIDER or "").strip().lower() or None
    if p and p not in {"gupshup", "meta"}:
        logger.warning("WHATSAPP_PROVIDER=%s is not recognised — ignoring", p)
        return None
    return p


def is_configured() -> bool:
    """True when a provider + its required creds are all set."""
    p = _provider()
    if p == "gupshup":
        return bool(
            settings.GUPSHUP_API_KEY
            and settings.GUPSHUP_APP_NAME
            and settings.GUPSHUP_SOURCE_NUMBER
        )
    if p == "meta":
        return bool(
            settings.META_WHATSAPP_ACCESS_TOKEN
            and settings.META_WHATSAPP_PHONE_NUMBER_ID
        )
    return False


def _normalize_phone(phone: str) -> str:
    """Strip non-digits. We store + validate E.164 at capture time in
    the route handler; this is the transport-layer normalizer. Most
    BSPs want just the digits (no leading '+')."""
    return "".join(ch for ch in phone or "" if ch.isdigit())


# ============================================================================
# Public API
# ============================================================================


async def send_text(phone: str, body: str, *, template: Optional[str] = None) -> bool:
    """Business-initiated text send. Returns True when the BSP accepted
    the request (202/200), False on any error or when unconfigured."""
    to = _normalize_phone(phone)
    if not to or not body:
        return False
    if not is_configured():
        logger.debug("whatsapp send skipped — provider not configured")
        return False

    p = _provider()
    try:
        if p == "gupshup":
            return await _send_gupshup(to, body, template=template)
        if p == "meta":
            return await _send_meta(to, body, template=template)
    except Exception as exc:
        logger.warning("whatsapp send failed (provider=%s): %s", p, exc)
    return False


async def send_otp(phone: str, code: str, ttl_minutes: int = 10) -> bool:
    """Send a verification OTP. Uses the configured OTP template; falls
    back to a plain text message for providers that permit it inside a
    24h session window."""
    body = (
        f"Your Quant X verification code is {code}. "
        f"Expires in {ttl_minutes} minutes. Don't share this with anyone."
    )
    return await send_text(phone, body, template=settings.WHATSAPP_OTP_TEMPLATE)


# ============================================================================
# Gupshup — https://docs.gupshup.io/reference/send-message
# ============================================================================


async def _send_gupshup(to: str, body: str, *, template: Optional[str]) -> bool:
    """POST to Gupshup's /wa/api/v1/msg endpoint. Template parameter is
    the approved-template name; when None we send a 'session' message
    (only valid within a 24h window of the user replying)."""
    url = "https://api.gupshup.io/wa/api/v1/msg"
    data = {
        "channel": "whatsapp",
        "source": settings.GUPSHUP_SOURCE_NUMBER,
        "destination": to,
        "src.name": settings.GUPSHUP_APP_NAME,
    }
    if template:
        data["message"] = f'{{"type":"text","text":{_json(body)}}}'
        # Template-driven sends use the template API endpoint instead —
        # simplified here: BSP routes based on the `src.name` mapping.
    else:
        data["message"] = f'{{"type":"text","text":{_json(body)}}}'

    headers = {
        "apikey": settings.GUPSHUP_API_KEY or "",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, data=data)
    ok = 200 <= resp.status_code < 300
    if not ok:
        logger.warning("gupshup send non-2xx: status=%d body=%s", resp.status_code, resp.text[:200])
    return ok


# ============================================================================
# Meta Cloud API — https://developers.facebook.com/docs/whatsapp/cloud-api/reference
# ============================================================================


async def _send_meta(to: str, body: str, *, template: Optional[str]) -> bool:
    """POST to the Meta Cloud API /messages endpoint. Uses a text message
    when no template is specified — valid inside a 24h session window.
    Outside that window you must use the template path with approved
    variables."""
    phone_id = settings.META_WHATSAPP_PHONE_NUMBER_ID
    url = f"https://graph.facebook.com/v20.0/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.META_WHATSAPP_ACCESS_TOKEN or ''}",
        "Content-Type": "application/json",
    }
    if template:
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template,
                "language": {"code": "en"},
                "components": [{
                    "type": "body",
                    "parameters": [{"type": "text", "text": body}],
                }],
            },
        }
    else:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body},
        }
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
    ok = 200 <= resp.status_code < 300
    if not ok:
        logger.warning("meta send non-2xx: status=%d body=%s", resp.status_code, resp.text[:200])
    return ok


# ============================================================================
# Helpers
# ============================================================================


def _json(s: str) -> str:
    """Minimal JSON-string escape — avoids a json.dumps import in the
    Gupshup form-data path (we embed the text in a larger urlencoded
    body)."""
    import json
    return json.dumps(s, ensure_ascii=False)


__all__ = [
    "is_configured",
    "send_otp",
    "send_text",
]
