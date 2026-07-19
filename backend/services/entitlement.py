"""
Data-licensing entitlement gate — SEBI **Path-A** enforcement.

Path-A (LOCKED product decision, 2026-06-07): paying / unauthenticated users
may only be shown *raw NSE market-data DISPLAY* when **either**

  (a) the platform holds a redistribution licence for that data class — the
      operator asserts this via a config flag (``NSE_REALTIME_LICENSED`` /
      ``NSE_EOD_LICENSED``); **or**
  (b) the request is served on behalf of a user who has connected their **own**
      broker (per-user OAuth = their own licensed terminal feed).

Otherwise the gate **fails closed**: the caller returns its honest-empty shape
plus an ``entitlement_required`` marker, and the frontend shows the frosted-glass
"connect your broker to get live data" lock (see ``BrokerLock`` / ``OptionChainPreview``).

What is **NOT** gated here: derived / educational products — ML signals, the
regime label, sentiment "Mood", backtests, screeners' pass/fail verdicts. Those
are research output, not raw-quote redistribution, and are governed by the
RA/IA question, not the data licence.

Default flags are **False** so a fresh deploy is compliant-by-default: no raw NSE
display leaves the building unless the operator has flipped a licence flag or the
user has their own broker connected. The moment a licence is signed, flip the flag
— no code change, no redeploy of logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from ..core.config import settings

logger = logging.getLogger(__name__)


class DataClass(str, Enum):
    """Raw NSE market-data display classes subject to the licence gate."""

    LIVE_QUOTE = "live_quote"          # real-time LTP for a symbol
    INTRADAY = "intraday"             # intraday bars / ticks
    INDEX = "index"                   # index levels (Nifty / BankNifty / VIX)
    OHLC = "ohlc"                     # historical candles
    MARKET_DEPTH = "market_depth"     # L1/L2 order-book depth
    FNO_CHAIN = "fno_chain"           # option chain / OI / greeks / max-pain
    FII_DII = "fii_dii"               # FII/DII cash + derivative flows
    DELIVERY = "delivery"             # delivery %, bulk/block deals
    PARTICIPANT_OI = "participant_oi" # participant-wise OI, F&O ban list


# Per-class policy:  (licence_flag_attr, broker_feed_grants)
#   licence_flag_attr  — the Settings bool that, when True, means the operator
#                        holds a central redistribution licence for this class.
#   broker_feed_grants — True if a user's OWN connected broker terminal provides
#                        this data (so per-user OAuth entitles them). NSE-published
#                        aggregates (FII/DII, participant OI, bulk deals) are NOT in
#                        a broker feed, so a broker connection does NOT entitle them —
#                        only the EOD licence does.
_POLICY: Dict[DataClass, Tuple[str, bool]] = {
    DataClass.LIVE_QUOTE:     ("NSE_REALTIME_LICENSED", True),
    DataClass.INTRADAY:       ("NSE_REALTIME_LICENSED", True),
    DataClass.INDEX:          ("NSE_REALTIME_LICENSED", True),
    DataClass.MARKET_DEPTH:   ("NSE_REALTIME_LICENSED", True),
    DataClass.FNO_CHAIN:      ("NSE_REALTIME_LICENSED", True),
    DataClass.OHLC:           ("NSE_EOD_LICENSED", True),
    DataClass.FII_DII:        ("NSE_EOD_LICENSED", False),
    DataClass.DELIVERY:       ("NSE_EOD_LICENSED", False),
    DataClass.PARTICIPANT_OI: ("NSE_EOD_LICENSED", False),
}


@dataclass
class Entitlement:
    allowed: bool
    data_class: str
    source: Optional[str] = None   # "licence" | "broker" | None
    reason: Optional[str] = None   # "licence_required" | "broker_required"


# ── per-user broker-connection cache (30s) ────────────────────────────────
# Market endpoints can fire many times per page; avoid a DB round-trip on each.
_broker_cache: Dict[str, Tuple[float, bool]] = {}
_BROKER_TTL = 30.0


def _has_live_broker(user_id: str) -> bool:
    """True when the user has at least one broker in ``connected`` state.

    A connected ``broker_connections`` row means per-user OAuth is live — that
    is their own licensed terminal feed, which entitles them to the live data
    classes (per Path-A). Cached 30s. Fail-closed (False) on any error.
    """
    if not user_id:
        return False
    hit = _broker_cache.get(user_id)
    now = time.monotonic()
    if hit and now - hit[0] < _BROKER_TTL:
        return hit[1]
    connected = False
    try:
        from ..api.app import get_supabase_admin  # lazy — avoid import cycle

        res = (
            get_supabase_admin()
            .table("broker_connections")
            .select("broker_name", count="exact")
            .eq("user_id", user_id)
            .eq("status", "connected")
            .execute()
        )
        connected = bool(res.count or (res.data or []))
    except Exception as e:  # DB down → fail closed (no entitlement)
        logger.warning("broker entitlement check failed for user: %s", e)
        connected = False
    _broker_cache[user_id] = (now, connected)
    return connected


def _licence_held(flag_attr: str) -> bool:
    return bool(getattr(settings, flag_attr, False))


def check_entitlement(data_class: DataClass, user_id: Optional[str]) -> Entitlement:
    """Decide whether *this* caller may be shown *this* raw-data class."""
    flag_attr, broker_grants = _POLICY[data_class]

    if _licence_held(flag_attr):
        return Entitlement(True, data_class.value, source="licence")

    if broker_grants and user_id and _has_live_broker(user_id):
        return Entitlement(True, data_class.value, source="broker")

    # Development / test convenience: outside production, serve data so local
    # dev and the e2e suite work without a signed licence or a live broker.
    # PRODUCTION fails closed (below) — this is the compliant deploy posture.
    if (settings.APP_ENV or "").lower() not in ("production", "prod"):
        return Entitlement(True, data_class.value, source="dev")

    reason = "broker_required" if broker_grants else "licence_required"
    return Entitlement(False, data_class.value, source=None, reason=reason)


def resolve_optional_user_id(request: Any) -> Optional[str]:
    """Best-effort user id from the Authorization bearer — never raises, never 401s.

    Mirrors ``api.app.get_current_user`` verification: signature-verified when
    ``SUPABASE_JWT_SECRET`` is set (production), unverified decode only in dev
    (no secret). Honours the dev auth bypass. Returns ``None`` for anonymous /
    invalid / expired tokens so public landing widgets keep working.
    """
    # Dev-only bypass (hard-gated off in production inside core.security).
    try:
        from ..core.security import _dev_auth_enabled, _DEV_USER

        if _dev_auth_enabled():
            return getattr(_DEV_USER, "id", None)
    except Exception:
        pass

    try:
        auth = (request.headers.get("authorization") or "").strip()
    except Exception:
        return None
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None

    try:
        import jwt as pyjwt  # lazy

        secret = settings.SUPABASE_JWT_SECRET
        if secret:
            payload = pyjwt.decode(
                token,
                key=secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_signature": True, "verify_exp": True, "verify_aud": True},
            )
        else:
            payload = pyjwt.decode(token, options={"verify_signature": False}, algorithms=["HS256", "ES256"])
        if not isinstance(payload, dict):
            return None
        if payload.get("role") != "authenticated":
            return None
        return payload.get("sub")
    except Exception:
        return None


def entitlement_marker(ent: Entitlement, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The honest-empty marker to merge into a gated endpoint's response.

    The frontend reads ``entitlement_required`` to render the frosted-glass
    broker lock instead of an error. ``reason`` distinguishes "connect your
    broker" (``broker_required``) from "not available on this plan"
    (``licence_required``).
    """
    if ent.reason == "broker_required":
        message = "Connect your broker to view live NSE market data."
        cta = "connect_broker"
    else:
        message = "Live market data is not available yet."
        cta = "unavailable"
    payload: Dict[str, Any] = {
        "entitlement_required": True,
        "data_class": ent.data_class,
        "reason": ent.reason,
        "cta": cta,
        "message": message,
    }
    if extra:
        payload.update(extra)
    return payload


def broker_lock(data_class: DataClass, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The broker-lock ("connect your broker") marker — for when a user was
    entitled via their broker but the broker session is unavailable/expired."""
    return entitlement_marker(
        Entitlement(False, data_class.value, reason="broker_required"), extra
    )


def entitlement_for(request: Any, data_class: DataClass) -> Entitlement:
    """Resolve the caller from ``request`` and check entitlement in one step."""
    return check_entitlement(data_class, resolve_optional_user_id(request))


def entitlement_and_user(request: Any, data_class: DataClass) -> Tuple[Entitlement, Optional[str]]:
    """Like :func:`entitlement_for` but also returns the resolved user id, so a
    caller entitled via ``source == "broker"`` can source the data from that
    user's own broker feed (see services.user_broker_data)."""
    uid = resolve_optional_user_id(request)
    return check_entitlement(data_class, uid), uid
