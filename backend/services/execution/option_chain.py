"""Option chain service (PR-AX).

Wraps each broker's ``get_option_chain`` with a short in-process TTL
cache. Returns broker-fetched chains when a user has a connected
broker; returns ``None`` gracefully when no broker — callers should
fall back to BS-estimated premiums (paper executor already does).

Why 15s TTL:
  Option premiums tick every fraction of a second during market hours.
  15s is the sweet spot between rate-limit pressure on Kite/Upstox/
  Angel chain endpoints (they're slow + expensive) and acceptable
  staleness for mark-to-market display. Same idea as the market
  endpoint SWR cache from PR-AD.

Cache key shape:
  (user_id, symbol, expiry_iso) — user_id keys because each user's
  chain comes through their own broker; symbol + expiry to avoid
  mixing weekly chains.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


CHAIN_TTL_SECONDS = 15.0
_chain_cache: Dict[Tuple[str, str, Optional[str]], Tuple[float, List[Dict[str, Any]]]] = {}


@dataclass
class ChainEntry:
    """One row of the option chain — strike + side + market data."""
    strike: float
    option_type: str       # 'CE' | 'PE'
    expiry: str            # YYYY-MM-DD
    ltp: float
    bid: float = 0.0
    ask: float = 0.0
    oi: int = 0
    volume: int = 0
    iv: float = 0.0
    tradingsymbol: str = ""

    @classmethod
    def from_broker(cls, row: Dict[str, Any]) -> "ChainEntry":
        return cls(
            strike=float(row.get("strike", 0) or 0),
            option_type=str(row.get("option_type", "")).upper(),
            expiry=str(row.get("expiry", "")),
            ltp=float(row.get("ltp", 0) or 0),
            bid=float(row.get("bid", 0) or 0),
            ask=float(row.get("ask", 0) or 0),
            oi=int(row.get("oi", 0) or 0),
            volume=int(row.get("volume", 0) or 0),
            iv=float(row.get("iv", 0) or 0),
            tradingsymbol=str(row.get("tradingsymbol", "")),
        )


def _broker_for_user(supabase: Any, user_id: str):
    """Authenticated broker client; (None, None) on failure.
    Same shape as the helper in live_options_executor — kept local to
    avoid a circular import on cold path.
    """
    try:
        from ...data.brokers.credentials import decrypt_credentials
        from ...data.brokers.integration import BrokerFactory
    except Exception:
        return None, None

    try:
        conn = (
            supabase.table("broker_connections")
            .select("broker_name, access_token")
            .eq("user_id", user_id)
            .eq("status", "connected")
            .single()
            .execute()
        )
    except Exception:
        return None, None
    if not conn.data:
        return None, None

    broker_name = conn.data["broker_name"]
    try:
        creds = decrypt_credentials(conn.data["access_token"])
        broker = BrokerFactory.create(broker_name, creds)
        if broker and broker.login():
            return broker, broker_name
    except Exception as exc:
        logger.debug("option_chain: broker init failed: %s", exc)
    return None, broker_name


def get_option_chain(
    supabase: Any,
    user_id: str,
    symbol: str,
    expiry: Optional[date] = None,
) -> Optional[List[ChainEntry]]:
    """Fetch the option chain for ``symbol`` (and optional ``expiry``)
    via the user's connected broker.

    Returns:
        - A list of ChainEntry rows on success.
        - An empty list when the broker returned no data (rare —
          usually means the symbol is wrong or the expiry has no
          strikes within the broker's scan window).
        - None when no broker is connected — caller should fall
          back to BS-estimated premiums.
    """
    symbol = symbol.upper().strip()
    expiry_iso = expiry.isoformat() if expiry else None
    cache_key = (user_id, symbol, expiry_iso)

    # Cache hit?
    now = time.monotonic()
    hit = _chain_cache.get(cache_key)
    if hit and now - hit[0] < CHAIN_TTL_SECONDS:
        return [ChainEntry.from_broker(r) for r in hit[1]]

    broker, broker_name = _broker_for_user(supabase, user_id)
    if broker is None:
        return None

    try:
        raw = broker.get_option_chain(symbol, expiry=expiry_iso or "") or []
    except Exception as exc:
        logger.warning(
            "option_chain: %s.get_option_chain(%s) failed: %s",
            broker_name, symbol, exc,
        )
        return []

    _chain_cache[cache_key] = (now, raw)
    # Bound the cache to keep memory in check during high-load periods.
    if len(_chain_cache) > 256:
        oldest = min(_chain_cache.items(), key=lambda kv: kv[1][1][0] if False else kv[1][0])[0]
        _chain_cache.pop(oldest, None)
    return [ChainEntry.from_broker(r) for r in raw]


def lookup_leg_ltp(
    chain: List[ChainEntry],
    *,
    strike: float,
    expiry: date,
    option_type: str,
) -> Optional[float]:
    """Find the LTP for a specific leg in a fetched chain.

    Strikes match exact; expiry compared as ISO strings; option_type
    normalised uppercase. Returns None when no match.
    """
    if not chain:
        return None
    expiry_iso = expiry.isoformat()
    opt = option_type.upper()
    for row in chain:
        if (
            row.option_type == opt
            and abs(row.strike - float(strike)) < 0.01
            and row.expiry.startswith(expiry_iso)
            and row.ltp > 0
        ):
            return row.ltp
    return None
