import asyncio
from datetime import datetime, timedelta, timezone
from backend.data.brokers import freshness

def _iso(dt): return dt.astimezone(timezone.utc).isoformat()

def test_fresh_token_not_refreshed():
    exp = _iso(datetime.now(timezone.utc) + timedelta(hours=2))
    assert freshness.needs_refresh(exp, threshold_s=300) is False

def test_expired_token_needs_refresh():
    exp = _iso(datetime.now(timezone.utc) - timedelta(minutes=1))
    assert freshness.needs_refresh(exp, threshold_s=300) is True

def test_kite_oauth_cannot_refresh_marks_expired():
    assert freshness.refreshable("zerodha", {"access_token": "x", "api_key": "k"}) is False
    # enctoken connect WITH a stored TOTP secret can be re-minted silently
    assert freshness.refreshable(
        "zerodha", {"kite_user_id": "u", "kite_password": "p", "totp_secret": "s"}
    ) is True
    # one-time-OTP connect (no totp_secret) cannot refresh -> must reconnect
    assert freshness.refreshable(
        "zerodha", {"enctoken": "x", "kite_user_id": "u", "kite_password": "p"}
    ) is False
    assert freshness.refreshable("upstox", {"refresh_token": "r", "api_key": "k", "api_secret": "x"}) is True
