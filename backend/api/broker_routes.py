"""
================================================================================
QUANT X - BROKER AUTHENTICATION ROUTES
================================================================================
OAuth2 integration for:
- Zerodha (KiteConnect)
- Angel One (SmartAPI)
- Upstox (v2 API)

Each broker has:
1. /auth/initiate - Start OAuth flow, returns auth URL
2. /auth/callback - Handle OAuth callback, store credentials
3. Credentials are encrypted before storage
================================================================================
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from ..core.config import settings
from ..core.database import supabase_admin
from ..core.security import get_current_user
from ..core.tiers import Tier, UserTier
from ..data.brokers.credentials import encrypt_credentials, decrypt_credentials
from ..data.brokers.integration import (
    Order,
    OrderStatus,
    OrderType,
    ProductType,
    TransactionType,
)
from ..middleware.tier_gate import RequireTier
from ..platform.system_flags import global_halt_reason, is_globally_halted
from backend.platform import oauth_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/broker", tags=["broker"])


def _compute_expiry(broker: str, creds: dict | None = None) -> str | None:
    """Best-effort token expiry as an ISO8601 UTC string, or None if unknown.

    - upstox/angelone: use an explicit expiry if the stored creds carry one
      (e.g. ``expires_at``/``expires_in``); otherwise None.
    - zerodha (Kite): access tokens are valid until ~06:00 IST the next day.
    """
    from datetime import datetime, timedelta, timezone
    creds = creds or {}
    if broker in ("upstox", "angelone"):
        iso = creds.get("expires_at")
        if iso:
            return iso
        secs = creds.get("expires_in")
        if secs:
            try:
                return (datetime.now(timezone.utc) + timedelta(seconds=int(secs))).isoformat()
            except (TypeError, ValueError):
                return None
        return None
    if broker == "zerodha":
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        nxt = (now_ist + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
        return nxt.astimezone(timezone.utc).isoformat()
    return None


def _load_authenticated_broker(user: Any):
    """Load the connected broker, refresh its session if stale, and log in.

    Shared by the data endpoints. Applies refresh-before-use: if the stored
    token is near/past expiry and a silent refresh is possible, refresh and
    persist the new token+expiry; otherwise mark the connection ``expired`` and
    raise 409 so the UI prompts a reconnect.
    """
    from ..data.brokers import freshness
    from ..data.brokers.integration import BrokerFactory

    conn = supabase_admin.table("broker_connections").select(
        "broker_name, access_token, expires_at"
    ).eq("user_id", user.id).eq("status", "connected").single().execute()

    if not conn.data:
        raise HTTPException(status_code=400, detail="No broker connected")

    broker_name = conn.data['broker_name']
    credentials = decrypt_credentials(conn.data['access_token'])
    broker = BrokerFactory.create(broker_name, credentials)

    # Refresh-before-use: only when expiry is known and near/past.
    if freshness.needs_refresh(conn.data.get("expires_at")):
        new_creds = None
        if (
            broker_name == "zerodha"
            and _credential_login_allowed()
            and freshness.refreshable("zerodha", credentials)
        ):
            # DEV-ONLY (credential login disabled in prod): re-mint the enctoken
            # from stored Kite credentials. In production this branch is off and
            # an expired Zerodha token forces a fresh OAuth reconnect below.
            fresh_enctoken = _zerodha_auto_login(
                credentials.get("kite_user_id", ""),
                credentials.get("kite_password", ""),
                totp_secret=credentials.get("totp_secret", ""),
            )
            if fresh_enctoken:
                new_creds = {**credentials, "enctoken": fresh_enctoken}
        elif freshness.refreshable(broker_name, credentials) and getattr(
            broker, "refresh_session", lambda: False
        )():
            new_creds = {
                **credentials,
                "access_token": getattr(broker, "access_token", credentials.get("access_token")),
            }

        if new_creds is not None:
            supabase_admin.table("broker_connections").update({
                "access_token": encrypt_credentials(new_creds),
                "expires_at": _compute_expiry(broker_name, new_creds),
            }).eq("user_id", user.id).eq("broker_name", broker_name).execute()
            credentials = new_creds
            broker = BrokerFactory.create(broker_name, credentials)  # rebuild with the fresh token
        else:
            supabase_admin.table("broker_connections").update({"status": "expired"}).eq(
                "user_id", user.id).eq("broker_name", broker_name).execute()
            raise HTTPException(status_code=409, detail="broker_token_expired")

    if not broker.login():
        raise HTTPException(status_code=400, detail="Failed to login to broker")

    return broker


# ============================================================================
# CENTRALIZED CONFIG (from core.config.Settings)
# ============================================================================

ZERODHA_API_KEY = settings.ZERODHA_API_KEY
ZERODHA_API_SECRET = settings.ZERODHA_API_SECRET
ZERODHA_REDIRECT_URI = settings.ZERODHA_REDIRECT_URI

ANGEL_API_KEY = settings.ANGEL_API_KEY
ANGEL_REDIRECT_URI = settings.ANGEL_REDIRECT_URI

UPSTOX_API_KEY = settings.UPSTOX_API_KEY
UPSTOX_API_SECRET = settings.UPSTOX_API_SECRET
UPSTOX_REDIRECT_URI = settings.UPSTOX_REDIRECT_URI

FRONTEND_URL = settings.FRONTEND_URL


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class BrokerAuthInitiate(BaseModel):
    broker: str  # zerodha, angelone, upstox


class BrokerAuthCallback(BaseModel):
    code: str
    state: str


class BrokerStatus(BaseModel):
    connected: bool
    broker_name: Optional[str] = None
    last_synced: Optional[str] = None
    account_id: Optional[str] = None


class BrokerCredentialsUpdate(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    client_id: Optional[str] = None
    password: Optional[str] = None
    totp_secret: Optional[str] = None


class QuickConnectRequest(BaseModel):
    broker_name: str  # zerodha, angelone
    user_id: Optional[str] = None  # Zerodha user ID (e.g. AB1234)
    password: Optional[str] = None
    totp_secret: Optional[str] = None  # Base32 key for permanent auto-refresh
    totp_value: Optional[str] = None   # 6-digit OTP for one-time connect
    api_key: Optional[str] = None  # Angel One / Alice Blue API key
    client_id: Optional[str] = None  # Angel One / Dhan / Kotak / Alice Blue client id
    access_token: Optional[str] = None  # Dhan / Kotak / Alice Blue pasted token
    session_token: Optional[str] = None  # Kotak Neo "sid" session id (pasted)


class AdHocOrderRequest(BaseModel):
    symbol: str
    transaction_type: str          # BUY | SELL
    quantity: int
    order_type: str = "MARKET"     # MARKET | LIMIT
    price: float = 0               # required for LIMIT
    product: str = "CNC"           # CNC | MIS | NRML
    exchange: str = "NSE"


# ============================================================================
# AUTO-LOGIN HELPERS
# ============================================================================

def _credential_login_allowed() -> bool:
    """Broker credential/scrape login is a security + licensing liability.

    Allowed ONLY when explicitly enabled AND not in production. Production always
    uses the broker's licensed OAuth flow (Kite Connect / Upstox OAuth). See
    settings.ALLOW_BROKER_CREDENTIAL_LOGIN.
    """
    return bool(
        settings.ALLOW_BROKER_CREDENTIAL_LOGIN
        and (settings.APP_ENV or "").lower() not in ("production", "prod")
    )


def _zerodha_auto_login(
    kite_user_id: str,
    kite_password: str,
    totp_secret: str = "",
    totp_value: str = "",
) -> Optional[str]:
    """
    Auto-login to Zerodha via web session and return enctoken.
    Accepts either totp_secret (base32 key) or totp_value (6-digit code).
    """
    import requests as http_requests

    # Resolve TOTP code
    otp_code = ""
    if totp_value and len(totp_value) == 6 and totp_value.isdigit():
        otp_code = totp_value
    elif totp_secret:
        try:
            import pyotp
            otp_code = pyotp.TOTP(totp_secret).now()
        except Exception as e:
            logger.error(f"Invalid TOTP secret: {e}")
            return None
    else:
        logger.error("Either totp_secret or totp_value is required")
        return None

    try:
        session = http_requests.Session()

        # Step 1: GET login page
        session.get("https://kite.zerodha.com/", allow_redirects=True, timeout=15)

        # Step 2: POST credentials
        resp = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": kite_user_id, "password": kite_password},
            timeout=15,
        )
        resp.raise_for_status()
        login_data = resp.json()
        if login_data.get("status") != "success":
            msg = login_data.get("message", "Login failed")
            logger.error(f"Zerodha login failed: {msg}")
            return None

        request_id = login_data["data"]["request_id"]

        # Step 3: POST TOTP for 2FA
        resp = session.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "user_id": kite_user_id,
                "request_id": request_id,
                "twofa_value": otp_code,
                "twofa_type": "totp",
            },
            timeout=15,
        )
        resp.raise_for_status()
        twofa_data = resp.json()
        if twofa_data.get("status") != "success":
            msg = twofa_data.get("message", "2FA failed")
            logger.error(f"Zerodha 2FA failed: {msg}")
            return None

        # Step 4: Extract enctoken from session cookies
        enctoken = None
        for cookie in session.cookies:
            if cookie.name == "enctoken":
                enctoken = cookie.value
                break

        if not enctoken:
            logger.error("No enctoken in cookies after login")
            return None

        # Verify enctoken works
        verify = http_requests.get(
            "https://kite.zerodha.com/oms/user/profile",
            headers={"Authorization": f"enctoken {enctoken}"},
            timeout=10,
        )
        if verify.status_code != 200:
            logger.error(f"Enctoken verification failed: {verify.status_code}")
            return None

        logger.info(f"Zerodha auto-login OK for {kite_user_id}")
        return enctoken

    except Exception as e:
        logger.error(f"Zerodha auto-login error: {e}")
        return None


# ============================================================================
# COMMON ENDPOINTS
# ============================================================================

@router.get("/status")
async def get_broker_status(user: Any = Depends(get_current_user)) -> BrokerStatus:
    """
    Get current broker connection status for user (single connected row —
    preserved for backward compatibility). Frontend should prefer
    `/broker/connections` which returns per-broker status for the 3 tiles.
    """
    try:
        result = supabase_admin.table("broker_connections").select(
            "broker_name, status, last_synced_at, account_id"
        ).eq("user_id", user.id).eq("status", "connected").single().execute()

        if result.data:
            return BrokerStatus(
                connected=True,
                broker_name=result.data['broker_name'],
                last_synced=result.data.get('last_synced_at'),
                account_id=result.data.get('account_id')
            )

        return BrokerStatus(connected=False)

    except Exception as e:
        logger.error(f"Error getting broker status: {e}")
        return BrokerStatus(connected=False)


# ============================================================================
# Per-broker status list — drives the 3 broker tiles in /settings (PR 6)
# ============================================================================

SUPPORTED_BROKERS = ("zerodha", "upstox", "angelone", "fyers", "dhan", "kotakneo", "aliceblue")


@router.get("/connections")
async def get_broker_connections(user: Any = Depends(get_current_user)) -> Dict[str, Any]:
    """Return status for every supported broker, one row per broker.

    Shape: ``{"brokers": [{"broker_name": "zerodha", "status": "connected",
    "account_id": "AB1234", "last_synced_at": "...", "expires_at": "..."},
    ...]}``. Brokers the user has never connected appear with
    ``status="not_connected"``.
    """
    try:
        result = (
            supabase_admin.table("broker_connections")
            .select("broker_name, status, last_synced_at, account_id, expires_at, disconnected_at")
            .eq("user_id", user.id)
            .in_("broker_name", list(SUPPORTED_BROKERS))
            .execute()
        )
        rows_by_broker = {row["broker_name"]: row for row in (result.data or [])}
    except Exception as e:
        logger.error("broker/connections query failed: %s", e)
        rows_by_broker = {}

    brokers = []
    for name in SUPPORTED_BROKERS:
        row = rows_by_broker.get(name)
        if row is None:
            brokers.append({
                "broker_name": name,
                "status": "not_connected",
                "account_id": None,
                "last_synced_at": None,
                "expires_at": None,
            })
        else:
            brokers.append({
                "broker_name": name,
                "status": row.get("status") or "not_connected",
                "account_id": row.get("account_id"),
                "last_synced_at": row.get("last_synced_at"),
                "expires_at": row.get("expires_at"),
            })
    return {"brokers": brokers}


@router.post("/disconnect")
async def disconnect_broker(
        broker: Optional[str] = Query(
            None,
            description="Broker to disconnect (zerodha/upstox/angelone). Omit to disconnect all."),
        user: Any = Depends(get_current_user),
):
    """Disconnect a specific broker or all connected brokers.

    With `broker` omitted, disconnects every currently-connected broker
    for the user. Used by the global kill-switch flow.
    """
    try:
        query = (
            supabase_admin.table("broker_connections")
            .update({
                "status": "disconnected",
                "access_token": None,
                "refresh_token": None,
                "disconnected_at": datetime.utcnow().isoformat(),
            })
            .eq("user_id", user.id)
            .eq("status", "connected")
        )
        if broker:
            broker = broker.lower()
            if broker not in SUPPORTED_BROKERS:
                raise HTTPException(status_code=400, detail=f"Unsupported broker: {broker}")
            query = query.eq("broker_name", broker)
        query.execute()

        # Only flip the profile-level broker_connected flag if no other
        # broker still sits in `connected` state.
        remaining = (
            supabase_admin.table("broker_connections")
            .select("broker_name", count="exact")
            .eq("user_id", user.id)
            .eq("status", "connected")
            .execute()
        )
        if not (remaining.data or []):
            supabase_admin.table("user_profiles").update({
                "broker_connected": False,
                "broker_name": None,
            }).eq("id", user.id).execute()

        return {"success": True, "disconnected": broker or "all"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disconnecting broker: {e}")
        raise HTTPException(status_code=500, detail="Failed to disconnect broker")


# ============================================================================
# ONE-CLICK CONNECT (enctoken / credential-based)
# ============================================================================

@router.post("/connect")
async def quick_connect_broker(body: QuickConnectRequest, user: Any = Depends(get_current_user)):
    """
    One-click broker connection.
    Zerodha: auto-login via web session → enctoken (no API key needed).
    Angel One: SmartAPI credential auth → JWT token.
    """
    broker = body.broker_name.lower()

    if broker == "zerodha":
        # SECURITY/LICENSING: the Zerodha "credential" connect scraped the Kite
        # web login and stored the user's raw password + TOTP seed, bypassing the
        # licensed Kite Connect OAuth API. Disabled by default and always off in
        # production — users connect via the OAuth flow instead.
        if not _credential_login_allowed():
            raise HTTPException(
                status_code=400,
                detail=(
                    "For your security, Zerodha now connects via the official Kite "
                    "Connect login. Start it at /api/broker/zerodha/auth/initiate — "
                    "no password or TOTP is stored."
                ),
            )

        if not body.user_id or not body.password:
            raise HTTPException(status_code=400, detail="User ID and password are required")
        if not body.totp_secret and not body.totp_value:
            raise HTTPException(status_code=400, detail="Enter your 6-digit OTP or TOTP secret key")

        enctoken = _zerodha_auto_login(
            body.user_id, body.password,
            totp_secret=body.totp_secret or "",
            totp_value=body.totp_value or "",
        )
        if not enctoken:
            raise HTTPException(
                status_code=401,
                detail="Login failed. Please check your User ID, password, and OTP.",
            )

        # Never persist the raw broker password. Store the session enctoken only;
        # totp_secret is kept solely for dev-mode auto-refresh (credential login
        # is off in production).
        cred_payload = {
            "enctoken": enctoken,
            "kite_user_id": body.user_id,
        }
        auto_refresh = False
        if body.totp_secret and len(body.totp_secret) > 10:
            cred_payload["totp_secret"] = body.totp_secret
            auto_refresh = True

        encrypted_creds = encrypt_credentials(cred_payload)

        supabase_admin.table("broker_connections").upsert({
            "user_id": user.id,
            "broker_name": "zerodha",
            "status": "connected",
            "account_id": body.user_id,
            "access_token": encrypted_creds,
            "expires_at": _compute_expiry("zerodha", cred_payload),
            "connected_at": datetime.utcnow().isoformat(),
            "last_synced_at": datetime.utcnow().isoformat(),
        }, on_conflict="user_id,broker_name").execute()

        supabase_admin.table("user_profiles").update({
            "broker_connected": True,
            "broker_name": "zerodha",
        }).eq("id", user.id).execute()

        return {
            "success": True,
            "broker": "zerodha",
            "account_id": body.user_id,
            "auto_refresh": auto_refresh,
        }

    elif broker == "angelone":
        if not body.api_key or not body.client_id or not body.password or not body.totp_secret:
            raise HTTPException(status_code=400,
                                detail="Angel One requires api_key, client_id, password, and totp_secret")

        try:
            from SmartApi import SmartConnect
            import pyotp

            smart = SmartConnect(api_key=body.api_key)
            totp_val = pyotp.TOTP(body.totp_secret).now()

            data = smart.generateSession(body.client_id, body.password, totp_val)
            if not data or data.get("status") is False:
                raise HTTPException(status_code=401, detail="Angel One login failed — check credentials")

            jwt_token = data["data"]["jwtToken"]
            refresh_token = data["data"].get("refreshToken", "")

            angel_creds = {
                "api_key": body.api_key,
                "client_id": body.client_id,
                "password": body.password,
                "totp_secret": body.totp_secret,
                "access_token": jwt_token,
                "refresh_token": refresh_token,
            }
            encrypted_creds = encrypt_credentials(angel_creds)

            supabase_admin.table("broker_connections").upsert({
                "user_id": user.id,
                "broker_name": "angelone",
                "status": "connected",
                "account_id": body.client_id,
                "access_token": encrypted_creds,
                "expires_at": _compute_expiry("angelone", angel_creds),
                "connected_at": datetime.utcnow().isoformat(),
                "last_synced_at": datetime.utcnow().isoformat(),
            }, on_conflict="user_id,broker_name").execute()

            supabase_admin.table("user_profiles").update({
                "broker_connected": True,
                "broker_name": "angelone",
            }).eq("id", user.id).execute()

            return {"success": True, "broker": "angelone", "account_id": body.client_id}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Angel One connect error: {e}")
            raise HTTPException(status_code=500, detail=f"Angel One connection failed: {str(e)}")

    elif broker == "dhan":
        # BETA — Dhan connects by pasting a Client ID + Access Token (generated
        # at web.dhan.co). We validate the token by loading fund limits, then
        # store it encrypted. No password is ever involved.
        if not body.client_id or not body.access_token:
            raise HTTPException(status_code=400, detail="Dhan needs a Client ID and Access Token")
        try:
            from ..data.brokers.integration import BrokerFactory as _BF

            dhan_creds = {"client_id": body.client_id, "access_token": body.access_token}
            adapter = _BF.create("dhan", dhan_creds)
            if not adapter.login():
                raise HTTPException(status_code=401, detail="Dhan login failed — check your Client ID and Access Token")

            encrypted_creds = encrypt_credentials(dhan_creds)
            supabase_admin.table("broker_connections").upsert({
                "user_id": user.id,
                "broker_name": "dhan",
                "status": "connected",
                "account_id": body.client_id,
                "access_token": encrypted_creds,
                "expires_at": _compute_expiry("dhan", dhan_creds),
                "connected_at": datetime.utcnow().isoformat(),
                "last_synced_at": datetime.utcnow().isoformat(),
            }, on_conflict="user_id,broker_name").execute()
            supabase_admin.table("user_profiles").update({
                "broker_connected": True,
                "broker_name": "dhan",
            }).eq("id", user.id).execute()
            return {"success": True, "broker": "dhan", "account_id": body.client_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Dhan connect error: {e}")
            raise HTTPException(status_code=500, detail=f"Dhan connection failed: {str(e)}")

    elif broker in ("kotakneo", "aliceblue"):
        # BETA token-paste brokers — the user pastes their session token(s).
        if not body.client_id or not body.access_token:
            raise HTTPException(status_code=400, detail=f"{broker} needs a Client ID and Access Token")
        try:
            from ..data.brokers.integration import BrokerFactory as _BF

            creds = {"client_id": body.client_id, "access_token": body.access_token}
            if body.session_token:      # Kotak Neo "sid"
                creds["session_token"] = body.session_token
            if body.api_key:            # Alice Blue api key (optional)
                creds["api_key"] = body.api_key
            adapter = _BF.create(broker, creds)
            if not adapter.login():
                raise HTTPException(status_code=401, detail=f"{broker} login failed — check your token")

            supabase_admin.table("broker_connections").upsert({
                "user_id": user.id,
                "broker_name": broker,
                "status": "connected",
                "account_id": body.client_id,
                "access_token": encrypt_credentials(creds),
                "expires_at": _compute_expiry(broker, creds),
                "connected_at": datetime.utcnow().isoformat(),
                "last_synced_at": datetime.utcnow().isoformat(),
            }, on_conflict="user_id,broker_name").execute()
            supabase_admin.table("user_profiles").update({
                "broker_connected": True,
                "broker_name": broker,
            }).eq("id", user.id).execute()
            return {"success": True, "broker": broker, "account_id": body.client_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"{broker} connect error: {e}")
            raise HTTPException(status_code=500, detail=f"{broker} connection failed: {str(e)}")

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported broker: {broker}. Use zerodha, angelone, dhan, "
                "kotakneo, or aliceblue."
            ),
        )


# ============================================================================
# ZERODHA (KITE CONNECT) — legacy OAuth
# ============================================================================

@router.post("/zerodha/auth/initiate")
async def zerodha_auth_initiate(
    return_to: str = Query("settings"),
    user: Any = Depends(get_current_user),
):
    """
    Initiate Zerodha OAuth flow.
    Returns the Kite Connect login URL.
    """
    if not settings.ZERODHA_API_KEY:
        raise HTTPException(status_code=409, detail="broker_not_configured")

    state = await oauth_state.store_state(user.id, "zerodha", return_to=return_to)

    # Kite Connect login URL
    auth_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={ZERODHA_API_KEY}"

    return {
        "auth_url": auth_url,
        "state": state,
        "instructions": "Complete login on Zerodha, then call the callback endpoint with the request_token"
    }


@router.post("/zerodha/auth/callback")
async def zerodha_auth_callback(
    request_token: str = Query(...),
    state: str = Query(...),
    user: Any = Depends(get_current_user)
):
    """
    Handle Zerodha OAuth callback.
    Exchange request_token for access_token.
    """
    # Verify state
    state_data = await oauth_state.consume_state(state)
    if not state_data:
        raise HTTPException(status_code=400, detail="invalid_or_expired_state")
    if state_data['user_id'] != user.id:
        raise HTTPException(status_code=400, detail="invalid_or_expired_state")

    if not ZERODHA_API_KEY or not ZERODHA_API_SECRET:
        raise HTTPException(status_code=400, detail="Zerodha API credentials not configured")

    try:
        # Generate checksum
        checksum = hashlib.sha256(
            f"{ZERODHA_API_KEY}{request_token}{ZERODHA_API_SECRET}".encode()
        ).hexdigest()

        # Exchange for access token
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.kite.trade/session/token",
                data={
                    "api_key": ZERODHA_API_KEY,
                    "request_token": request_token,
                    "checksum": checksum
                }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to exchange token")

            data = response.json().get("data", {})

        access_token = data.get("access_token")
        user_id_broker = data.get("user_id")

        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")

        # Encrypt and store credentials
        zerodha_creds = {
            "api_key": ZERODHA_API_KEY,
            "access_token": access_token,
            "user_id": user_id_broker
        }
        encrypted_creds = encrypt_credentials(zerodha_creds)

        # Upsert broker connection
        supabase_admin.table("broker_connections").upsert({
            "user_id": user.id,
            "broker_name": "zerodha",
            "status": "connected",
            "account_id": user_id_broker,
            "access_token": encrypted_creds,
            "expires_at": _compute_expiry("zerodha", zerodha_creds),
            "connected_at": datetime.utcnow().isoformat(),
            "last_synced_at": datetime.utcnow().isoformat()
        }, on_conflict="user_id,broker_name").execute()

        # Update user profile
        supabase_admin.table("user_profiles").update({
            "broker_connected": True,
            "broker_name": "zerodha"
        }).eq("id", user.id).execute()

        return {
            "success": True,
            "broker": "zerodha",
            "account_id": user_id_broker,
            "return_to": state_data.get("return_to", "settings"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Zerodha auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


# ============================================================================
# ANGEL ONE (SMART API)
# ============================================================================

@router.post("/angelone/auth/initiate")
async def angelone_auth_initiate(
    return_to: str = Query("settings"),
    user: Any = Depends(get_current_user),
):
    """
    Initiate Angel One authentication.
    Angel One uses API key + client credentials (no OAuth redirect).
    Returns instructions for credential entry.
    """
    state = await oauth_state.store_state(user.id, "angelone", return_to=return_to)

    return {
        "auth_type": "credentials",
        "state": state,
        "required_fields": [
            {"field": "client_id", "label": "Client ID", "type": "text"},
            {"field": "password", "label": "PIN/Password", "type": "password"},
            {"field": "totp_secret", "label": "TOTP Secret (from SmartAPI)", "type": "password"}
        ],
        "instructions": "Enter your Angel One SmartAPI credentials. Get API key and TOTP secret from smartapi.angelbroking.com"
    }


@router.post("/angelone/auth/credentials")
async def angelone_auth_credentials(
    credentials: BrokerCredentialsUpdate,
    state: str = Query(...),
    user: Any = Depends(get_current_user)
):
    """
    Authenticate with Angel One using credentials.
    """
    # Verify state
    state_data = await oauth_state.consume_state(state)
    if not state_data:
        raise HTTPException(status_code=400, detail="invalid_or_expired_state")
    if state_data['user_id'] != user.id:
        raise HTTPException(status_code=400, detail="invalid_or_expired_state")

    if not credentials.client_id or not credentials.password or not credentials.totp_secret:
        raise HTTPException(status_code=400, detail="Missing required credentials")

    try:
        # Attempt login with SmartAPI
        import pyotp

        totp = pyotp.TOTP(credentials.totp_secret).now()

        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://apiconnect.angelbroking.com/rest/auth/angelbroking/user/v1/loginByPassword",
                json={
                    "clientcode": credentials.client_id,
                    "password": credentials.password,
                    "totp": totp
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-UserType": "USER",
                    "X-SourceID": "WEB",
                    "X-ClientLocalIP": "0.0.0.0",  # nosec B104 - broker API header value, not a socket bind
                    "X-ClientPublicIP": "0.0.0.0",  # nosec B104 - broker API header value, not a socket bind
                    "X-MACAddress": "00:00:00:00:00:00",
                    "X-PrivateKey": ANGEL_API_KEY
                }
            )

            data = response.json()

        if not data.get("status"):
            raise HTTPException(status_code=400, detail=data.get("message", "Login failed"))

        jwt_token = data.get("data", {}).get("jwtToken")
        refresh_token = data.get("data", {}).get("refreshToken")

        # Encrypt and store credentials
        angel_creds = {
            "api_key": ANGEL_API_KEY,
            "client_id": credentials.client_id,
            "password": credentials.password,
            "totp_secret": credentials.totp_secret,
            "jwt_token": jwt_token,
            "refresh_token": refresh_token
        }
        encrypted_creds = encrypt_credentials(angel_creds)

        # Upsert broker connection
        supabase_admin.table("broker_connections").upsert({
            "user_id": user.id,
            "broker_name": "angelone",
            "status": "connected",
            "account_id": credentials.client_id,
            "access_token": encrypted_creds,
            "expires_at": _compute_expiry("angelone", angel_creds),
            "connected_at": datetime.utcnow().isoformat(),
            "last_synced_at": datetime.utcnow().isoformat()
        }, on_conflict="user_id,broker_name").execute()

        # Update user profile
        supabase_admin.table("user_profiles").update({
            "broker_connected": True,
            "broker_name": "angelone"
        }).eq("id", user.id).execute()

        return {
            "success": True,
            "broker": "angelone",
            "account_id": credentials.client_id,
            "return_to": state_data.get("return_to", "settings"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Angel One auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


# ============================================================================
# UPSTOX (V2 API)
# ============================================================================

@router.post("/upstox/auth/initiate")
async def upstox_auth_initiate(
    return_to: str = Query("settings"),
    user: Any = Depends(get_current_user),
):
    """
    Initiate Upstox OAuth2 flow.
    Returns the Upstox authorization URL.
    """
    if not settings.UPSTOX_API_KEY:
        raise HTTPException(status_code=409, detail="broker_not_configured")

    state = await oauth_state.store_state(user.id, "upstox", return_to=return_to)

    # Upstox OAuth2 URL
    params = {
        "client_id": UPSTOX_API_KEY,
        "redirect_uri": UPSTOX_REDIRECT_URI or f"{FRONTEND_URL}/broker/callback",
        "response_type": "code",
        "state": state
    }

    auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?{urlencode(params)}"

    return {
        "auth_url": auth_url,
        "state": state
    }


@router.post("/upstox/auth/callback")
async def upstox_auth_callback(
    code: str = Query(...),
    state: str = Query(...),
    user: Any = Depends(get_current_user)
):
    """
    Handle Upstox OAuth2 callback.
    Exchange code for access token.
    """
    # Verify state
    state_data = await oauth_state.consume_state(state)
    if not state_data:
        raise HTTPException(status_code=400, detail="invalid_or_expired_state")
    if state_data['user_id'] != user.id:
        raise HTTPException(status_code=400, detail="invalid_or_expired_state")

    if not UPSTOX_API_KEY or not UPSTOX_API_SECRET:
        raise HTTPException(status_code=400, detail="Upstox API credentials not configured")

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.upstox.com/v2/login/authorization/token",
                data={
                    "code": code,
                    "client_id": UPSTOX_API_KEY,
                    "client_secret": UPSTOX_API_SECRET,
                    "redirect_uri": UPSTOX_REDIRECT_URI or f"{FRONTEND_URL}/broker/callback",
                    "grant_type": "authorization_code"
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to exchange token")

            data = response.json()

        access_token = data.get("access_token")

        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")

        # Get user profile from Upstox
        async with httpx.AsyncClient() as client:
            profile_response = await client.get(
                "https://api.upstox.com/v2/user/profile",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            profile_data = profile_response.json().get("data", {})

        user_id_broker = profile_data.get("user_id", "")

        # Encrypt and store credentials
        upstox_creds = {
            "api_key": UPSTOX_API_KEY,
            "api_secret": UPSTOX_API_SECRET,
            "access_token": access_token,
            "refresh_token": data.get("refresh_token"),
            "user_id": user_id_broker
        }
        encrypted_creds = encrypt_credentials(upstox_creds)

        # Upsert broker connection
        supabase_admin.table("broker_connections").upsert({
            "user_id": user.id,
            "broker_name": "upstox",
            "status": "connected",
            "account_id": user_id_broker,
            "access_token": encrypted_creds,
            "expires_at": _compute_expiry("upstox", upstox_creds),
            "connected_at": datetime.utcnow().isoformat(),
            "last_synced_at": datetime.utcnow().isoformat()
        }, on_conflict="user_id,broker_name").execute()

        # Update user profile
        supabase_admin.table("user_profiles").update({
            "broker_connected": True,
            "broker_name": "upstox"
        }).eq("id", user.id).execute()

        return {
            "success": True,
            "broker": "upstox",
            "account_id": user_id_broker,
            "return_to": state_data.get("return_to", "settings"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upstox auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


# ============================================================================
# FYERS (API v3) — OAuth2. BETA (2026-07-12). Verify with a live account.
# ============================================================================

@router.post("/fyers/auth/initiate")
async def fyers_auth_initiate(
    return_to: str = Query("settings"),
    user: Any = Depends(get_current_user),
):
    """Initiate Fyers OAuth2 flow — returns the Fyers auth-code login URL."""
    if not settings.FYERS_API_KEY:
        raise HTTPException(status_code=409, detail="broker_not_configured")

    state = await oauth_state.store_state(user.id, "fyers", return_to=return_to)
    params = {
        "client_id": settings.FYERS_API_KEY,
        "redirect_uri": settings.FYERS_REDIRECT_URI or f"{FRONTEND_URL}/broker/callback",
        "response_type": "code",
        "state": state,
    }
    auth_url = f"https://api-t1.fyers.in/api/v3/generate-authcode?{urlencode(params)}"
    return {"auth_url": auth_url, "state": state}


@router.post("/fyers/auth/callback")
async def fyers_auth_callback(
    state: str = Query(...),
    code: Optional[str] = Query(None),
    auth_code: Optional[str] = Query(None),
    user: Any = Depends(get_current_user),
):
    """Handle the Fyers OAuth2 callback. Fyers returns ``auth_code`` (accepted
    as ``code`` too), which we exchange for an access token."""
    state_data = await oauth_state.consume_state(state)
    if not state_data or state_data.get("user_id") != user.id:
        raise HTTPException(status_code=400, detail="invalid_or_expired_state")

    the_code = auth_code or code
    if not the_code:
        raise HTTPException(status_code=400, detail="missing_auth_code")
    if not settings.FYERS_API_KEY or not settings.FYERS_API_SECRET:
        raise HTTPException(status_code=400, detail="Fyers API credentials not configured")

    try:
        app_hash = hashlib.sha256(
            f"{settings.FYERS_API_KEY}:{settings.FYERS_API_SECRET}".encode()
        ).hexdigest()
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api-t1.fyers.in/api/v3/validate-authcode",
                json={"grant_type": "authorization_code", "appIdHash": app_hash, "code": the_code},
                timeout=15,
            )
        data = resp.json() if resp.status_code == 200 else {}
        access_token = data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to exchange Fyers auth code")

        fyers_creds = {"app_id": settings.FYERS_API_KEY, "access_token": access_token}
        supabase_admin.table("broker_connections").upsert({
            "user_id": user.id,
            "broker_name": "fyers",
            "status": "connected",
            "account_id": settings.FYERS_API_KEY,
            "access_token": encrypt_credentials(fyers_creds),
            "expires_at": _compute_expiry("fyers", fyers_creds),
            "connected_at": datetime.utcnow().isoformat(),
            "last_synced_at": datetime.utcnow().isoformat(),
        }, on_conflict="user_id,broker_name").execute()
        supabase_admin.table("user_profiles").update({
            "broker_connected": True,
            "broker_name": "fyers",
        }).eq("id", user.id).execute()

        return {
            "success": True,
            "broker": "fyers",
            "return_to": state_data.get("return_to", "settings"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fyers auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


# ============================================================================
# BROKER DATA ENDPOINTS
# ============================================================================

@router.get("/positions")
async def get_broker_positions(user: Any = Depends(get_current_user)):
    """
    Get positions from connected broker.
    """
    try:
        broker = _load_authenticated_broker(user)

        positions = broker.get_positions()

        return {
            "positions": [{
                "symbol": p.symbol,
                "exchange": p.exchange,
                "quantity": p.quantity,
                "average_price": p.average_price,
                "current_price": p.current_price,
                "pnl": p.pnl,
                "pnl_percent": p.pnl_percent
            } for p in positions]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch positions")


@router.get("/orders")
async def get_broker_orders(user: Any = Depends(get_current_user)):
    """
    Get the user's live order book from their connected broker (read-only).
    """
    try:
        broker = _load_authenticated_broker(user)

        return {"orders": broker.get_orders()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        return {"orders": []}


@router.get("/holdings")
async def get_broker_holdings(user: Any = Depends(get_current_user)):
    """
    Get holdings from connected broker.
    """
    try:
        broker = _load_authenticated_broker(user)

        holdings = broker.get_holdings()

        return {"holdings": holdings}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch holdings")


@router.get("/margin")
async def get_broker_margin(user: Any = Depends(get_current_user)):
    """
    Get available margin from connected broker.
    """
    try:
        broker = _load_authenticated_broker(user)

        available_margin = broker.get_available_margin()

        return {
            "available_margin": available_margin,
            "used_margin": 0  # Would need additional API call
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching margin: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch margin")


# ============================================================================
# MANUAL AD-HOC ORDER
# ============================================================================

def _validate_adhoc_order(body: "AdHocOrderRequest"):
    """Validate + parse ad-hoc order params into broker enums.

    Pure (no DB / no broker) so it's unit-testable. Raises
    ``HTTPException(422)`` on any bad parameter. Returns
    ``(transaction_type, order_type, product)`` enum triple on success.
    Only MARKET/LIMIT order types are accepted here.
    """
    try:
        txn = TransactionType(body.transaction_type.upper())
        otype = OrderType(body.order_type.upper())
        product = ProductType(body.product.upper())
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid order parameters")
    if otype not in (OrderType.MARKET, OrderType.LIMIT):
        raise HTTPException(status_code=422, detail="Only MARKET and LIMIT orders are supported here")
    if body.quantity < 1:
        raise HTTPException(status_code=422, detail="Quantity must be at least 1")
    if otype == OrderType.LIMIT and (not body.price or body.price <= 0):
        raise HTTPException(status_code=422, detail="Limit orders require a price")
    return txn, otype, product


@router.post("/order")
async def place_manual_order(
    body: AdHocOrderRequest,
    user: Any = Depends(get_current_user),
    tier: UserTier = Depends(RequireTier(Tier.PRO)),   # live trading is Pro+; 402 for Free
):
    """Place a manual ad-hoc order on the user's connected broker.

    Safety gates (in order): global kill-switch -> per-user kill-switch ->
    tier (Pro+) -> connected+fresh broker. Manual orders are intentionally
    exempt from the auto-trader paper-first whitelist (it's the user's own
    deliberate order). Order types limited to MARKET/LIMIT.
    """
    # 1) Global halt
    if is_globally_halted(supabase_client=supabase_admin):
        raise HTTPException(
            status_code=503,
            detail=global_halt_reason(supabase_client=supabase_admin) or "Live trading is currently halted.",
        )
    # 2) Per-user kill switch (.limit(1) not .single() — a missing profile row
    #    must not 500; it simply means "no kill switch set").
    prof = supabase_admin.table("user_profiles").select("kill_switch_active").eq("id", user.id).limit(1).execute()
    prof_row = (prof.data or [{}])[0]
    if prof_row.get("kill_switch_active"):
        raise HTTPException(status_code=400, detail="Kill switch active. Trading paused.")
    # 3) Validate order params (422 on bad input)
    txn, otype, product = _validate_adhoc_order(body)
    # 4) Connected + fresh broker (raises 400/409 if not)
    broker = _load_authenticated_broker(user)
    # 5) Place it
    order = Order(
        symbol=body.symbol.upper(), exchange=body.exchange.upper(),
        transaction_type=txn, quantity=int(body.quantity),
        product=product, order_type=otype, price=float(body.price or 0),
    )
    placed = broker.place_order(order)
    status_val = getattr(placed.status, "value", str(placed.status)) if placed.status else "PENDING"
    return {
        "success": placed.status != OrderStatus.REJECTED,
        "order_id": placed.order_id,
        "status": status_val,
        "message": placed.message or "",
        "symbol": order.symbol, "transaction_type": txn.value, "quantity": order.quantity,
    }
