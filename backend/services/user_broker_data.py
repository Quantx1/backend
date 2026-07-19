"""
Per-user broker-sourced market data — Path-A "bring your own broker".

When the operator holds no NSE data licence, live NSE display is served from the
USER's OWN connected broker terminal (their licensed feed) rather than any
central scraper/admin account — keeping the platform out of the redistribution
chain. Callers use this whenever ``entitlement.check_entitlement`` returns
``source == "broker"``; if the user's broker session is missing/expired/errors,
these return ``None`` and the caller falls back to the honest-empty broker-lock.

Covered from the connected broker's own API (all three adapters support them):
  * quote(symbol)      — real-time LTP + OHLC
  * indices()          — Nifty 50 / Bank Nifty / India VIX
  * option_chain(sym)  — live option chain

Historical OHLC is broker-specific (instrument-token lookup) and is not sourced
here yet — those endpoints stay broker-locked in production until wired.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Short-lived per-user broker-session cache. Rebuilding an adapter decrypts creds
# + logs in, which is heavy; a connected user hitting several widgets shouldn't
# pay that each call. Keyed by user_id → (built_at, broker_name, adapter).
_session_cache: Dict[str, Tuple[float, str, Any]] = {}
_SESSION_TTL = 60.0

# NSE index display names as the broker quote APIs expect them.
_INDEX_SYMBOLS = [
    ("nifty", "NIFTY 50"),
    ("banknifty", "NIFTY BANK"),
    ("vix", "INDIA VIX"),
]


def _build_adapter(user_id: str):
    """Resolve the user's connected broker → a logged-in adapter, or None.

    Cached 60s. Never raises — returns (None, None) on any problem so the caller
    falls back to the broker-lock.
    """
    hit = _session_cache.get(user_id)
    now = time.monotonic()
    if hit and now - hit[0] < _SESSION_TTL:
        return hit[1], hit[2]

    try:
        from ..api.app import get_supabase_admin  # lazy — avoid import cycle
        from ..data.brokers.credentials import decrypt_credentials
        from ..data.brokers.integration import BrokerFactory

        row = (
            get_supabase_admin()
            .table("broker_connections")
            .select("broker_name, access_token")
            .eq("user_id", user_id)
            .eq("status", "connected")
            .limit(1)
            .execute()
        )
        data = (row.data or [None])[0]
        if not data:
            return None, None
        broker_name = data["broker_name"]
        creds = decrypt_credentials(data["access_token"])
        adapter = BrokerFactory.create(broker_name, creds)
        # BrokerFactory.create does not authenticate — log in so the adapter's
        # data calls (get_quote / historical_data / get_option_chain) work.
        try:
            logged_in = bool(adapter.login())
        except Exception as e:
            logger.warning("user_broker_data: broker login failed: %s", e)
            logged_in = False
        if not logged_in:
            return None, None
        _session_cache[user_id] = (now, broker_name, adapter)
        return broker_name, adapter
    except Exception as e:
        logger.warning("user_broker_data: could not build broker session: %s", e)
        return None, None


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_quote(symbol: str, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a broker's raw quote dict to the API display shape.

    Tolerant across brokers: Kite exposes ``last_price`` + a nested ``ohlc``
    (whose ``close`` is the PREVIOUS close); Upstox/Angel use flatter keys. Any
    shape missing a usable last price → None (caller broker-locks).
    """
    if not isinstance(raw, dict) or not raw:
        return None

    ltp = _f(raw.get("last_price"))
    if ltp is None:
        ltp = _f(raw.get("ltp")) or _f(raw.get("lastPrice")) or _f(raw.get("last_traded_price"))
    if ltp is None:
        return None

    ohlc = raw.get("ohlc") if isinstance(raw.get("ohlc"), dict) else {}
    o = _f(ohlc.get("open")) or _f(raw.get("open"))
    h = _f(ohlc.get("high")) or _f(raw.get("high"))
    lo = _f(ohlc.get("low")) or _f(raw.get("low"))
    prev_close = (
        _f(ohlc.get("close"))
        or _f(raw.get("close"))
        or _f(raw.get("previous_close"))
        or _f(raw.get("prev_close"))
    )
    vol = (
        raw.get("volume")
        or raw.get("volume_traded")
        or raw.get("volume_traded_today")
        or raw.get("vol")
        or 0
    )
    change = _f(raw.get("net_change"))
    if change is None and prev_close:
        change = round(ltp - prev_close, 2)
    change_pct = round(change / prev_close * 100, 2) if (change is not None and prev_close) else 0.0

    try:
        vol_int = int(vol)
    except (TypeError, ValueError):
        vol_int = 0

    return {
        "symbol": symbol.upper(),
        "ltp": round(ltp, 2),
        "open": round(o, 2) if o is not None else 0.0,
        "high": round(h, 2) if h is not None else 0.0,
        "low": round(lo, 2) if lo is not None else 0.0,
        "close": round(prev_close, 2) if prev_close is not None else 0.0,
        "volume": vol_int,
        "change": change if change is not None else 0.0,
        "change_percent": change_pct,
        "timestamp": datetime.now().isoformat(),
        "source": "broker",
    }


def quote(user_id: str, symbol: str, exchange: str = "NSE") -> Optional[Dict[str, Any]]:
    """Real-time quote from the user's own broker feed, or None to broker-lock."""
    _name, adapter = _build_adapter(user_id)
    if adapter is None:
        return None
    sym = symbol.upper().strip().replace(".NS", "")
    try:
        raw = adapter.get_quote(sym, exchange)
    except Exception as e:
        logger.warning("user_broker_data.quote(%s) failed: %s", sym, e)
        return None
    return _normalize_quote(sym, raw or {})


def indices(user_id: str) -> Optional[Dict[str, Any]]:
    """Nifty / Bank Nifty / VIX from the user's broker feed, or None to lock.

    Returns None only when the broker session itself is unavailable; if the
    session is live but an individual index is missing it is reported as zeros
    (honest-empty per index), matching the central endpoint's shape.
    """
    _name, adapter = _build_adapter(user_id)
    if adapter is None:
        return None

    out: Dict[str, Any] = {}
    got_any = False
    for key, disp in _INDEX_SYMBOLS:
        entry = {"ltp": 0, "change": 0, "change_percent": 0}
        try:
            q = _normalize_quote(disp, adapter.get_quote(disp, "NSE") or {})
            if q:
                entry = {"ltp": q["ltp"], "change": q["change"], "change_percent": q["change_percent"]}
                got_any = True
        except Exception as e:
            logger.warning("user_broker_data.indices %s failed: %s", disp, e)
        out[key] = entry
    return out if got_any else None


def option_chain(user_id: str, symbol: str, expiry: str = "") -> Optional[List[Dict[str, Any]]]:
    """Live option chain from the user's own broker feed, or None to broker-lock."""
    _name, adapter = _build_adapter(user_id)
    if adapter is None:
        return None
    try:
        chain = adapter.get_option_chain(symbol.upper().strip(), expiry)
    except Exception as e:
        logger.warning("user_broker_data.option_chain(%s) failed: %s", symbol, e)
        return None
    return chain or None


# ── broker-sourced historical (charts) — Kite OAuth ───────────────────────
# Instrument-token resolution reuses the shared symbol→token cache (reference
# data, identical for everyone); refreshed via whichever user's Kite session is
# to hand, cached 24h. Only the Kite (KiteConnect) path is wired — other brokers
# fall back to the broker-lock until their historical APIs are added.
_token_cache = None
_token_lock = threading.Lock()

_KITE_INTERVAL = {
    "1d": "day", "1day": "day", "day": "day",
    "1wk": "week", "1w": "week", "week": "week",
    "1h": "60minute", "60m": "60minute",
    "15m": "15minute", "5m": "5minute",
}


def _resolve_kite_token(kite: Any, symbol: str) -> Optional[int]:
    global _token_cache
    try:
        from ..data.providers.kite import InstrumentTokenCache  # lazy
    except Exception as e:
        logger.warning("user_broker_data: InstrumentTokenCache unavailable: %s", e)
        return None
    with _token_lock:
        if _token_cache is None:
            _token_cache = InstrumentTokenCache()
        if _token_cache.is_stale():
            try:
                _token_cache.refresh(kite)
            except Exception as e:
                logger.warning("user_broker_data: instrument cache refresh failed: %s", e)
    return _token_cache.get_token(symbol.upper())


def historical(
    user_id: str, symbol: str, interval: str = "1d", days: int = 30
) -> Optional[Dict[str, Any]]:
    """Historical OHLCV from the user's own broker, or None to broker-lock.

    Two paths: Kite via ``kite.historical_data`` (+ instrument-token cache), and
    any adapter exposing a ``get_historical(symbol, period, interval)`` method
    (e.g. Fyers). Brokers with neither → None (broker-lock)."""
    _name, adapter = _build_adapter(user_id)
    if adapter is None:
        return None

    sym = symbol.upper().strip().replace(".NS", "")

    # Generic path: adapter-provided historical (Fyers etc.). Kite has no
    # get_historical method, so it falls through to the Kite path below.
    getter = getattr(adapter, "get_historical", None)
    if callable(getter) and getattr(adapter, "kite", None) is None:
        period = {5: "5d", 30: "1mo", 90: "3mo", 180: "6mo"}.get(int(days), "1y" if days > 180 else "1mo")
        try:
            rows = getter(sym, period, interval)
        except Exception as e:
            logger.warning("user_broker_data.historical adapter get_historical(%s) failed: %s", sym, e)
            return None
        if not rows:
            return None
        return {"symbol": sym, "interval": interval, "data": rows, "source": "broker"}

    kite = getattr(adapter, "kite", None)
    if kite is None:  # enctoken-mode Kite / Upstox / Angel / Dhan — not wired
        return None

    token = _resolve_kite_token(kite, sym)
    if not token:
        return None

    kite_interval = _KITE_INTERVAL.get(interval, "day")
    try:
        to_d = date.today()
        from_d = to_d - timedelta(days=max(int(days), 1))
        rows = kite.historical_data(token, from_d, to_d, kite_interval)
    except Exception as e:
        logger.warning("user_broker_data.historical(%s) failed: %s", sym, e)
        return None

    out: List[Dict[str, Any]] = []
    for r in rows or []:
        ts = r.get("date")
        out.append({
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "open": float(r.get("open", 0) or 0),
            "high": float(r.get("high", 0) or 0),
            "low": float(r.get("low", 0) or 0),
            "close": float(r.get("close", 0) or 0),
            "volume": int(r.get("volume", 0) or 0),
        })
    if not out:
        return None
    return {"symbol": sym, "interval": interval, "data": out, "source": "broker"}
