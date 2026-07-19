"""Refresh-before-use for stored broker credentials.

`needs_refresh` decides from expires_at; `refreshable` decides whether a silent
refresh is even possible for the broker + stored creds (Kite OAuth tokens are
daily and can't be refreshed silently → caller marks the connection expired).
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional


def needs_refresh(expires_at_iso: Optional[str], threshold_s: int = 300) -> bool:
    if not expires_at_iso:
        return False  # unknown expiry → don't churn; rely on 401 handling
    try:
        exp = datetime.fromisoformat(expires_at_iso)
    except ValueError:
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return (exp - datetime.now(timezone.utc)).total_seconds() <= threshold_s


def refreshable(broker: str, creds: dict) -> bool:
    if broker == "upstox":
        return all(creds.get(k) for k in ("refresh_token", "api_key", "api_secret"))
    if broker == "angelone":
        return bool(creds.get("refresh_token") or (creds.get("password") and creds.get("totp_secret")))
    if broker == "zerodha":
        # The enctoken can be re-minted via the stored Kite credentials + TOTP
        # secret (see _zerodha_auto_login). A one-time-OTP connect (no totp_secret)
        # or a Kite OAuth token cannot be refreshed silently → reconnect.
        return bool(
            creds.get("kite_user_id")
            and creds.get("kite_password")
            and creds.get("totp_secret")
        )
    return False
