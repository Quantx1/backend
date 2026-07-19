"""
================================================================================
SWING AI — KITE CONNECT DATA PROVIDER
================================================================================
Primary OHLCV data source using admin's Zerodha Kite Connect account.
Secondary: jugaad-data (free NSE bhavcopy) when Kite token is expired.

Architecture:
  - Admin's Kite account = app-wide market data (stored in .env)
  - User's broker account = personal order placement only (stored per-user in DB)
  - No yfinance, no TrueData dependency

Classes:
  RateLimiter          — Token-bucket rate limiter for Kite API (200 req/min)
  InstrumentTokenCache — Caches kite.instruments() mapping symbol → token
  KiteAdminClient      — Manages admin Kite Connect session + daily token refresh
  KiteDataProvider     — MarketDataProvider subclass using Kite + jugaad-data
================================================================================
"""

import logging
import time
import threading
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...core.config import settings

logger = logging.getLogger(__name__)

# Period string → days mapping (matches yfinance-style period strings)
_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1w": 7, "1mo": 30, "3mo": 90,
    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 3650,
}

# Interval mapping: yfinance-style → Kite Connect interval
_INTERVAL_MAP = {
    "1m": "minute", "3m": "3minute", "5m": "5minute",
    "15m": "15minute", "30m": "30minute", "1h": "60minute",
    "1d": "day", "1wk": "day",
}

# Index name mapping: our aliases → Kite tradingsymbol
INDEX_NAME_MAP = {
    "NIFTY": "NIFTY 50",
    "NIFTY50": "NIFTY 50",
    "NIFTY 50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTY BANK": "NIFTY BANK",
    "VIX": "INDIA VIX",
    "INDIA VIX": "INDIA VIX",
    "NIFTYIT": "NIFTY IT",
    "NIFTY IT": "NIFTY IT",
    "NIFTYFIN": "NIFTY FIN SERVICE",
}


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Simple token-bucket rate limiter for Kite API (200 req/min max)."""

    def __init__(self, max_per_minute: int = 180):
        self._max = max_per_minute
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def wait(self):
        """Block until a request slot is available."""
        with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self._max:
                sleep_time = 60 - (now - self._timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._timestamps.append(time.time())


# =============================================================================
# INSTRUMENT TOKEN CACHE
# =============================================================================

class InstrumentTokenCache:
    """
    Cache of NSE + NFO instrument tokens from kite.instruments().
    Refreshed once daily (or on cold start).
    Maps 'RELIANCE' → 738561, 'NIFTY 50' → 256265, etc.
    NFO cache: stores full instrument dicts for options chain lookups.
    """

    def __init__(self):
        self._eq_cache: Dict[str, int] = {}       # symbol → instrument_token (EQ segment)
        self._index_cache: Dict[str, int] = {}     # index name → instrument_token
        self._nfo_instruments: List[Dict] = []      # Full NFO instrument list for options
        self._all_instruments: List[Dict] = []
        self._last_refresh: Optional[datetime] = None

    def refresh(self, kite) -> None:
        """Fetch all NSE + NSE-INDEX + NFO instruments from Kite."""
        try:
            # Equity instruments
            nse_instruments = kite.instruments("NSE")
            self._eq_cache = {}
            for inst in nse_instruments:
                if inst.get("segment") == "NSE" and inst.get("instrument_type") == "EQ":
                    self._eq_cache[inst["tradingsymbol"]] = inst["instrument_token"]

            # Index instruments
            try:
                self._index_cache = {}
                for inst in nse_instruments:
                    if inst.get("instrument_type") == "EQ" and inst.get("segment") == "NSE":
                        continue
                    name = inst.get("tradingsymbol", "")
                    if inst.get("segment") == "INDICES" or name in INDEX_NAME_MAP.values():
                        self._index_cache[name] = inst["instrument_token"]
            except Exception:
                pass

            # Hardcode common index tokens as fallback
            _KNOWN_INDEX_TOKENS = {
                "NIFTY 50": 256265,
                "NIFTY BANK": 260105,
                "INDIA VIX": 264969,
                "NIFTY IT": 259849,
                "NIFTY FIN SERVICE": 257801,
            }
            for name, token in _KNOWN_INDEX_TOKENS.items():
                if name not in self._index_cache:
                    self._index_cache[name] = token

            # NFO instruments (options + futures)
            try:
                self._nfo_instruments = kite.instruments("NFO")
                nfo_ce_pe = sum(
                    1 for i in self._nfo_instruments
                    if i.get("instrument_type") in ("CE", "PE")
                )
                logger.info(f"NFO cache: {len(self._nfo_instruments)} total, {nfo_ce_pe} options")
            except Exception as e:
                logger.warning(f"NFO instruments fetch failed: {e}")
                self._nfo_instruments = []

            self._all_instruments = nse_instruments
            self._last_refresh = datetime.now()
            logger.info(
                f"Instrument cache refreshed: {len(self._eq_cache)} EQ, "
                f"{len(self._index_cache)} indices, {len(self._nfo_instruments)} NFO"
            )
        except Exception as e:
            logger.error(f"Instrument cache refresh failed: {e}")

    def is_stale(self) -> bool:
        """True if cache is empty or older than 24 hours."""
        if self._last_refresh is None:
            return True
        return datetime.now() - self._last_refresh > timedelta(hours=24)

    def get_token(self, symbol: str) -> Optional[int]:
        """Get instrument_token for an equity symbol."""
        return self._eq_cache.get(symbol.upper())

    def get_index_token(self, index_name: str) -> Optional[int]:
        """Get instrument_token for an index."""
        kite_name = INDEX_NAME_MAP.get(index_name.upper(), index_name.upper())
        return self._index_cache.get(kite_name)

    def get_all_eq_symbols(self) -> List[str]:
        """Return all EQ-segment symbols."""
        return list(self._eq_cache.keys())

    def get_nfo_options(self, symbol: str, expiry: Optional[date] = None) -> List[Dict]:
        """
        Get NFO options instruments for a symbol (CE + PE contracts).
        If expiry is None, returns nearest expiry.
        """
        filtered = [
            i for i in self._nfo_instruments
            if i.get("name") == symbol.upper()
            and i.get("instrument_type") in ("CE", "PE")
        ]
        if not filtered:
            return []

        if expiry:
            return [i for i in filtered if i.get("expiry") == expiry]

        # Pick nearest expiry
        expiries = sorted(set(i["expiry"] for i in filtered if i.get("expiry")))
        today = date.today()
        future_expiries = [e for e in expiries if e >= today]
        nearest = future_expiries[0] if future_expiries else (expiries[-1] if expiries else None)
        if not nearest:
            return []
        return [i for i in filtered if i.get("expiry") == nearest]

    def get_nfo_token(self, tradingsymbol: str) -> Optional[int]:
        """Get instrument_token for an NFO tradingsymbol."""
        for i in self._nfo_instruments:
            if i.get("tradingsymbol") == tradingsymbol:
                return i.get("instrument_token")
        return None


# =============================================================================
# KITE ADMIN CLIENT
# =============================================================================

class KiteAdminClient:
    """
    Manages admin-level Kite Connect session for app-wide market data.
    Reads KITE_ADMIN_API_KEY and KITE_ADMIN_ACCESS_TOKEN from settings.
    """

    def __init__(self):
        self._kite = None
        self._instruments = InstrumentTokenCache()
        self._is_connected: bool = False
        self._token_set_at: Optional[datetime] = None
        self._rate_limiter = RateLimiter(max_per_minute=180)

    def initialize(self) -> bool:
        """Create KiteConnect instance, set access token, refresh instruments."""
        api_key = settings.KITE_ADMIN_API_KEY
        access_token = settings.KITE_ADMIN_ACCESS_TOKEN

        if not api_key:
            logger.warning("KITE_ADMIN_API_KEY not set — Kite data provider unavailable")
            return False

        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=api_key)

            if access_token:
                self._kite.set_access_token(access_token)
                self._token_set_at = datetime.now()
                self._is_connected = True
                logger.info("Kite admin client connected")

                # Refresh instrument cache
                self._instruments.refresh(self._kite)
            else:
                logger.warning("KITE_ADMIN_ACCESS_TOKEN not set — set via /admin/kite/refresh-token")
                self._is_connected = False

            return self._is_connected
        except Exception as e:
            logger.error(f"Kite admin client init failed: {e}")
            self._is_connected = False
            return False

    def is_token_valid(self) -> bool:
        """Check if access token is likely still valid.
        Kite tokens expire at 6 AM IST daily."""
        if not self._is_connected or self._kite is None:
            return False
        if self._token_set_at is None:
            return False

        try:
            from zoneinfo import ZoneInfo
            ist = ZoneInfo("Asia/Kolkata")
            now_ist = datetime.now(ist)
            token_set_ist = self._token_set_at.astimezone(
                ist) if self._token_set_at.tzinfo else self._token_set_at.replace(tzinfo=ist)

            # Token expires at 6 AM IST next day
            expiry = token_set_ist.replace(hour=6, minute=0, second=0, microsecond=0)
            if token_set_ist.hour >= 6:
                expiry += timedelta(days=1)

            return now_ist < expiry
        except Exception:
            # Fallback: assume valid for 18 hours
            return datetime.now() - self._token_set_at < timedelta(hours=18)

    def set_access_token(self, token: str) -> None:
        """Update access token (called by admin refresh endpoint)."""
        if self._kite is None:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=settings.KITE_ADMIN_API_KEY)

        self._kite.set_access_token(token)
        self._token_set_at = datetime.now()
        self._is_connected = True
        logger.info("Kite admin access token updated")

        # Refresh instruments with new token
        self._instruments.refresh(self._kite)

    @property
    def kite(self):
        """Return KiteConnect instance."""
        return self._kite

    @property
    def instruments(self) -> InstrumentTokenCache:
        return self._instruments

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._kite is not None


# =============================================================================
# KITE DATA PROVIDER
# =============================================================================

class KiteDataProvider:
    """
    OHLCV data provider using admin's Kite Connect account.
    Primary: kite.historical_data() for per-symbol OHLCV
    Secondary: jugaad-data stock_df() when Kite token is expired
    """

    def __init__(self, admin_client: KiteAdminClient):
        self._admin = admin_client
        self._hist_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._hist_cache_ttl = 300  # 5 minutes
        # Lock around the upstream kiteconnect SDK — its historical_data()
        # call has been observed to interleave columns under thread-pool
        # concurrency (caller gets ['open','open','high','high',…]). Until
        # we know the root cause inside kiteconnect, serialise the calls.
        self._kite_lock = threading.Lock()
        # Per-cache-key locks prevent thundering-herd against the SDK when
        # many threads ask for the same symbol on a cold cache.
        self._cache_locks: Dict[str, threading.Lock] = {}
        self._cache_locks_guard = threading.Lock()

    def _key_lock(self, cache_key: str) -> threading.Lock:
        """Return a stable per-cache-key lock."""
        with self._cache_locks_guard:
            lock = self._cache_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._cache_locks[cache_key] = lock
            return lock

    @staticmethod
    def _sanitize_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Defensive — strip duplicate columns + dedupe index.

        Even with the kite lock, return a fresh, structurally clean
        DataFrame so downstream code can rely on `df['close']` being a
        Series and `df.index.is_unique` being True.
        """
        if df is None or df.empty:
            return df
        # Drop duplicate columns by name (keep first occurrence)
        if not df.columns.is_unique:
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        # Drop duplicate index entries (keep last — newer bar wins)
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]
        return df

    # ── Period / Interval helpers ──

    @staticmethod
    def _period_to_start_date(period: str) -> date:
        """Convert yfinance-style period string to a start date."""
        days = _PERIOD_DAYS.get(period, 180)
        return date.today() - timedelta(days=days)

    @staticmethod
    def _map_interval(interval: str) -> str:
        """Convert yfinance-style interval to Kite interval string."""
        return _INTERVAL_MAP.get(interval, "day")

    # ── Cache helpers ──

    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        if cache_key in self._hist_cache:
            df, cached_at = self._hist_cache[cache_key]
            if (datetime.now() - cached_at).total_seconds() < self._hist_cache_ttl:
                return df
        return None

    def _set_cached(self, cache_key: str, df: pd.DataFrame):
        self._hist_cache[cache_key] = (df, datetime.now())

    # ── Historical OHLCV ──

    def get_historical(
        self, symbol: str, period: str = "6mo", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV for a single symbol.
        Primary: Kite Connect. Secondary: jugaad-data.

        Returns DataFrame with columns: open, high, low, close, volume
        and DatetimeIndex.

        Thread-safety: uses a per-cache-key lock to coalesce concurrent
        requests for the same symbol, and a global lock around the
        upstream kiteconnect SDK call. The returned DataFrame is
        sanitised — duplicate columns / index entries dropped — so
        downstream code can safely do `df["close"]` (always a Series)
        and `df.loc[ts]` (no duplicate-label errors).
        """
        cache_key = f"{symbol}:{period}:{interval}"

        # Fast path — read cache without locking; the read is atomic enough
        # at the dict level and a stale read just falls through to the
        # locked slow path.
        cached = self._get_cached(cache_key)
        if cached is not None:
            return self._sanitize_columns(cached)

        # Slow path — per-key lock prevents N threads from each issuing
        # their own SDK call for the same symbol.
        with self._key_lock(cache_key):
            # Re-check inside the lock — another thread may have filled
            # the cache while we were waiting.
            cached = self._get_cached(cache_key)
            if cached is not None:
                return self._sanitize_columns(cached)

            from_date = self._period_to_start_date(period)
            to_date = date.today()
            kite_interval = self._map_interval(interval)

            df = None
            # Try Kite Connect first
            if self._admin.is_token_valid():
                # Serialise the SDK call — kiteconnect.historical_data has
                # been observed to interleave responses under thread-pool
                # concurrency, returning DataFrames with duplicate columns.
                with self._kite_lock:
                    df = self._fetch_via_kite(symbol, from_date, to_date, kite_interval)

            # Fallback: jugaad-data
            if df is None or df.empty:
                df = self._fetch_via_jugaad(symbol, from_date, to_date)

            df = self._sanitize_columns(df)
            if df is not None and not df.empty:
                self._set_cached(cache_key, df)
                return df

        return None

    def get_historical_index(
        self, index_name: str, period: str = "6mo"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV for an index (NIFTY, BANKNIFTY, VIX, etc.)."""
        cache_key = f"IDX:{index_name}:{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        from_date = self._period_to_start_date(period)
        to_date = date.today()

        # Resolve index name to Kite tradingsymbol
        kite_name = INDEX_NAME_MAP.get(index_name.upper(), index_name.upper())

        if self._admin.is_token_valid():
            token = self._admin.instruments.get_index_token(index_name)
            if token:
                df = self._fetch_kite_historical(token, from_date, to_date, "day")
                if df is not None and not df.empty:
                    self._set_cached(cache_key, df)
                    return df

        # Fallback: jugaad-data for indices
        df = self._fetch_index_via_jugaad(kite_name, from_date, to_date)
        if df is not None and not df.empty:
            self._set_cached(cache_key, df)
            return df

        return None

    def fetch_historical_batch(
        self, symbols: List[str], period: str = "6mo"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for many symbols. Used by screener and signal generator.
        Sequential with rate limiting (200 req/min → ~3 symbols/sec).
        """
        results: Dict[str, pd.DataFrame] = {}
        from_date = self._period_to_start_date(period)
        to_date = date.today()
        use_kite = self._admin.is_token_valid()

        # Refresh instrument cache if stale
        if use_kite and self._admin.instruments.is_stale():
            self._admin.instruments.refresh(self._admin.kite)

        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"{symbol}:{period}:1d"
                cached = self._get_cached(cache_key)
                if cached is not None:
                    results[symbol] = cached
                    continue

                df = None
                if use_kite:
                    df = self._fetch_via_kite(symbol, from_date, to_date, "day")

                if df is None or df.empty:
                    df = self._fetch_via_jugaad(symbol, from_date, to_date)

                if df is not None and not df.empty:
                    self._set_cached(cache_key, df)
                    results[symbol] = df
            except Exception as e:
                logger.debug(f"Batch fetch failed for {symbol}: {e}")
                continue

        logger.info(f"Batch fetch: {len(results)}/{len(symbols)} symbols loaded")
        return results

    # ── Kite Connect fetch ──

    def _fetch_via_kite(
        self, symbol: str, from_date: date, to_date: date, interval: str = "day"
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from Kite Connect historical_data API."""
        try:
            token = self._admin.instruments.get_token(symbol)
            if token is None:
                return None
            return self._fetch_kite_historical(token, from_date, to_date, interval)
        except Exception as e:
            logger.debug(f"Kite fetch failed for {symbol}: {e}")
            return None

    def _fetch_kite_historical(
        self, token: int, from_date: date, to_date: date, interval: str
    ) -> Optional[pd.DataFrame]:
        """Low-level Kite historical_data call with rate limiting."""
        try:
            self._admin.rate_limiter.wait()
            records = self._admin.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
            if not records:
                return None

            df = pd.DataFrame(records)
            # Kite returns: date, open, high, low, close, volume (already lowercase)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            # Drop timezone info if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            return df
        except Exception as e:
            logger.debug(f"Kite historical_data error (token={token}): {e}")
            return None

    # ── jugaad-data fetch ──

    def _fetch_via_jugaad(
        self, symbol: str, from_date: date, to_date: date
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from jugaad-data (NSE bhavcopy, free)."""
        try:
            from jugaad_data.nse import stock_df

            df = stock_df(
                symbol=symbol, from_date=from_date, to_date=to_date, series="EQ"
            )
            if df is None or df.empty:
                return None

            # Normalize columns
            col_map = {}
            for col in df.columns:
                cl = col.strip().upper()
                if cl == "DATE":
                    col_map[col] = "date"
                elif cl == "OPEN":
                    col_map[col] = "open"
                elif cl == "HIGH":
                    col_map[col] = "high"
                elif cl == "LOW":
                    col_map[col] = "low"
                elif cl in ("CLOSE", "LTP"):
                    if "close" not in col_map.values():
                        col_map[col] = "close"
                elif cl == "VOLUME" or "VOLUME" in cl:
                    if "volume" not in col_map.values():
                        col_map[col] = "volume"

            df = df.rename(columns=col_map)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # Ensure required columns exist
            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.debug(f"jugaad-data {symbol} missing columns: {missing}")
                return None

            df = df[required].copy()
            df = df.sort_index()
            return df
        except Exception as e:
            logger.debug(f"jugaad-data fetch failed for {symbol}: {e}")
            return None

    def _fetch_index_via_jugaad(
        self, index_name: str, from_date: date, to_date: date
    ) -> Optional[pd.DataFrame]:
        """Fetch index data from jugaad-data."""
        try:
            from jugaad_data.nse import index_df

            df = index_df(symbol=index_name, from_date=from_date, to_date=to_date)
            if df is None or df.empty:
                return None

            # Normalize columns
            col_map = {}
            for col in df.columns:
                cl = col.strip().upper()
                if cl == "DATE":
                    col_map[col] = "date"
                elif cl == "OPEN":
                    col_map[col] = "open"
                elif cl == "HIGH":
                    col_map[col] = "high"
                elif cl == "LOW":
                    col_map[col] = "low"
                elif cl in ("CLOSE", "CLOSING"):
                    if "close" not in col_map.values():
                        col_map[col] = "close"
                elif "VOLUME" in cl or "SHARES" in cl:
                    if "volume" not in col_map.values():
                        col_map[col] = "volume"

            df = df.rename(columns=col_map)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # Index data may not have volume — fill with 0
            if "volume" not in df.columns:
                df["volume"] = 0

            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                return None

            df = df[required].copy()
            df = df.sort_index()
            return df
        except Exception as e:
            logger.debug(f"jugaad-data index fetch failed for {index_name}: {e}")
            return None

    # ── Quote fetching ──

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote via kite.quote(). Falls back to last close from jugaad."""
        if self._admin.is_token_valid():
            try:
                self._admin.rate_limiter.wait()
                data = self._admin.kite.quote([f"NSE:{symbol}"])
                q = data.get(f"NSE:{symbol}")
                if q:
                    ohlc = q.get("ohlc", {})
                    return {
                        "symbol": symbol,
                        "ltp": q.get("last_price", 0),
                        "open": ohlc.get("open", 0),
                        "high": ohlc.get("high", 0),
                        "low": ohlc.get("low", 0),
                        "close": ohlc.get("close", 0),  # previous close
                        "volume": q.get("volume", 0),
                        "change": q.get("net_change", 0),
                        "change_percent": q.get("last_price", 0) / ohlc.get("close", 1) * 100 - 100
                        if ohlc.get("close", 0) > 0 else 0,
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.debug(f"Kite quote failed for {symbol}: {e}")

        # Fallback: last bar from historical
        df = self.get_historical(symbol, period="5d")
        if df is not None and not df.empty:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            return {
                "symbol": symbol,
                "ltp": float(last["close"]),
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": float(prev["close"]),
                "volume": int(last["volume"]),
                "change": float(last["close"] - prev["close"]),
                "change_percent": float((last["close"] - prev["close"]) / prev["close"] * 100)
                if prev["close"] > 0 else 0,
                "timestamp": datetime.now().isoformat(),
            }
        return None

    def get_depth(self, symbol: str):
        """Live L2 order-book depth (5 levels) via kite.quote — honest-None when
        the admin token is invalid or the quote carries no depth."""
        if not self._admin.is_token_valid():
            return None
        try:
            from ..brokers.depth_models import from_kite_depth
            self._admin.rate_limiter.wait()
            data = self._admin.kite.quote([f"NSE:{symbol}"])
            q = data.get(f"NSE:{symbol}") or {}
            if q.get("depth"):
                return from_kite_depth(symbol, q["depth"])
        except Exception as e:
            logger.debug("Kite depth failed for %s: %s", symbol, e)
        return None

    # ── Options Chain ──

    def get_option_chain(
        self, symbol: str, expiry: Optional[date] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch live options chain from Kite admin using cached NFO instruments.
        Returns list of dicts: strike, option_type, ltp, bid, ask, oi, volume, iv, Greeks.

        Flow:
          1. InstrumentTokenCache.get_nfo_options(symbol) → filtered CE/PE instruments
          2. Batch kite.quote(["NFO:NIFTY2530624100CE", ...]) → live prices + OI
          3. Greeks computed via Black-Scholes (Kite doesn't return Greeks)
        """
        if not self._admin.is_token_valid():
            logger.warning("Kite token invalid — cannot fetch live option chain")
            return None

        # Refresh instruments if stale
        if self._admin.instruments.is_stale():
            self._admin.instruments.refresh(self._admin.kite)

        instruments = self._admin.instruments.get_nfo_options(symbol, expiry)
        if not instruments:
            logger.warning(f"No NFO options instruments found for {symbol}")
            return None

        # Get spot price for Greeks calculation
        spot_price = 0.0
        spot_quote = self.get_quote(symbol)
        if spot_quote:
            spot_price = spot_quote.get("ltp", 0)

        # Fallback spot prices for indices (quote might come from NSE not index)
        if spot_price <= 0:
            _idx_map = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
                        "FINNIFTY": "NIFTY FIN SERVICE", "MIDCPNIFTY": "NIFTY MIDCAP SELECT"}
            idx_name = _idx_map.get(symbol.upper())
            if idx_name:
                idx_token = self._admin.instruments.get_index_token(symbol)
                if idx_token:
                    try:
                        self._admin.rate_limiter.wait()
                        data = self._admin.kite.quote([f"NSE:{idx_name}"])
                        q = data.get(f"NSE:{idx_name}")
                        if q:
                            spot_price = q.get("last_price", 0)
                    except Exception:
                        pass

        if spot_price <= 0:
            _defaults = {"NIFTY": 24000, "BANKNIFTY": 51000, "FINNIFTY": 23000, "MIDCPNIFTY": 12000}
            spot_price = _defaults.get(symbol.upper(), 0)

        # Fetch quotes in batches of 200 (Kite limit per call)
        chain: List[Dict[str, Any]] = []
        expiry_date = instruments[0].get("expiry") if instruments else date.today()
        days_to_expiry = max(1, (expiry_date - date.today()).days) if expiry_date else 1
        T = days_to_expiry / 365.0

        batch_size = 200
        for i in range(0, len(instruments), batch_size):
            batch = instruments[i:i + batch_size]
            nfo_keys = [f"NFO:{inst['tradingsymbol']}" for inst in batch]

            try:
                self._admin.rate_limiter.wait()
                quotes = self._admin.kite.quote(nfo_keys)
            except Exception as e:
                logger.warning(f"Kite NFO quote batch failed: {e}")
                continue

            for inst in batch:
                key = f"NFO:{inst['tradingsymbol']}"
                q = quotes.get(key, {})
                if not q:
                    continue

                ltp = q.get("last_price", 0)
                strike = float(inst.get("strike", 0))
                opt_type = inst.get("instrument_type", "CE")  # 'CE' or 'PE'
                oi = q.get("oi", 0)
                volume = q.get("volume", 0)

                # Compute IV via Newton-Raphson if we have spot + ltp
                iv = 0.0
                delta = 0.0
                gamma = 0.0
                theta = 0.0
                vega = 0.0

                if spot_price > 0 and ltp > 0 and T > 0:
                    try:
                        from scipy.stats import norm
                        import numpy as np

                        # Newton-Raphson IV
                        sigma = 0.3
                        r = 0.07
                        for _ in range(50):
                            d1 = (np.log(spot_price / strike) + (r + 0.5 * sigma**2) * T) / \
                                (sigma * np.sqrt(T)) if sigma > 0 else 0
                            d2 = d1 - sigma * np.sqrt(T) if sigma > 0 else 0

                            if opt_type == "CE":
                                price = spot_price * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
                            else:
                                price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

                            v = spot_price * norm.pdf(d1) * np.sqrt(T) if sigma > 0 else 0
                            if abs(v) < 1e-10:
                                break
                            diff = ltp - price
                            if abs(diff) < 0.01:
                                break
                            sigma = max(0.01, min(sigma + diff / (v * 100), 5.0))

                        iv = round(sigma * 100, 2)  # As percentage

                        # Greeks
                        d1 = (np.log(spot_price / strike) + (r + 0.5 * sigma**2) * T) / \
                            (sigma * np.sqrt(T)) if sigma > 0 else 0
                        d2 = d1 - sigma * np.sqrt(T) if sigma > 0 else 0
                        delta = round(norm.cdf(d1) if opt_type == "CE" else norm.cdf(d1) - 1, 4)
                        gamma = round(norm.pdf(d1) / (spot_price * sigma * np.sqrt(T)), 6) if sigma > 0 and T > 0 else 0
                        theta = round(-(spot_price * norm.pdf(d1) * sigma) /
                                      (2 * np.sqrt(T)) / 365, 2) if sigma > 0 and T > 0 else 0
                        vega = round(spot_price * norm.pdf(d1) * np.sqrt(T) / 100, 2) if sigma > 0 else 0
                    except Exception:
                        iv = 0.0

                depth = q.get("depth", {})
                bid_depth = depth.get("buy", [{}])
                ask_depth = depth.get("sell", [{}])

                chain.append({
                    "strike": strike,
                    "option_type": opt_type,
                    "expiry": str(expiry_date),
                    "tradingsymbol": inst["tradingsymbol"],
                    "instrument_token": inst.get("instrument_token", 0),
                    "lot_size": inst.get("lot_size", 1),
                    "ltp": ltp,
                    "bid": bid_depth[0].get("price", 0) if bid_depth else 0,
                    "ask": ask_depth[0].get("price", 0) if ask_depth else 0,
                    "oi": oi,
                    "oi_change": q.get("oi_day_high", 0) - q.get("oi_day_low", 0),
                    "volume": volume,
                    "iv": iv,
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "vega": vega,
                })

        logger.info(
            f"Option chain for {symbol}: {len(chain)} contracts, "
            f"spot={spot_price}, expiry={expiry_date}, DTE={days_to_expiry}"
        )
        return chain if chain else None

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batch quote fetch. Kite accepts up to 500 instruments per call."""
        results = {}
        if not symbols:
            return results

        if self._admin.is_token_valid():
            try:
                # Kite accepts up to 500 instruments per quote() call
                for i in range(0, len(symbols), 500):
                    batch = symbols[i:i + 500]
                    kite_symbols = [f"NSE:{s}" for s in batch]
                    self._admin.rate_limiter.wait()
                    data = self._admin.kite.quote(kite_symbols)
                    for s in batch:
                        q = data.get(f"NSE:{s}")
                        if q:
                            ohlc = q.get("ohlc", {})
                            results[s] = {
                                "symbol": s,
                                "ltp": q.get("last_price", 0),
                                "open": ohlc.get("open", 0),
                                "high": ohlc.get("high", 0),
                                "low": ohlc.get("low", 0),
                                "close": ohlc.get("close", 0),
                                "volume": q.get("volume", 0),
                                "change": q.get("net_change", 0),
                                "change_percent": q.get("last_price", 0) / ohlc.get("close", 1) * 100 - 100
                                if ohlc.get("close", 0) > 0 else 0,
                            }
                return results
            except Exception as e:
                logger.warning(f"Kite batch quote failed: {e}")

        return results


# =============================================================================
# SINGLETON
# =============================================================================

_kite_admin_client: Optional[KiteAdminClient] = None


def get_kite_admin_client() -> KiteAdminClient:
    """Get or create singleton Kite admin client."""
    global _kite_admin_client
    if _kite_admin_client is None:
        _kite_admin_client = KiteAdminClient()
    return _kite_admin_client


_kite_data_provider: Optional[KiteDataProvider] = None


def get_kite_data_provider() -> KiteDataProvider:
    """Get or create singleton KiteDataProvider."""
    global _kite_data_provider
    if _kite_data_provider is None:
        _kite_data_provider = KiteDataProvider(get_kite_admin_client())
    return _kite_data_provider


# =============================================================================
# AUTOMATED DAILY TOKEN REFRESH
# =============================================================================

def auto_refresh_kite_token() -> bool:
    """
    Automate daily Kite Connect token refresh using headless login.

    Flow:
      1. POST user_id + password to Kite login page
      2. POST TOTP for 2FA
      3. Extract request_token from redirect
      4. generate_session() → new access_token

    Requires env vars: KITE_ADMIN_USER_ID, KITE_ADMIN_PASSWORD, KITE_ADMIN_TOTP_SECRET
    Returns True on success, False on failure (falls back to manual refresh).
    """
    import requests as http_requests

    api_key = settings.KITE_ADMIN_API_KEY
    api_secret = settings.KITE_ADMIN_API_SECRET
    user_id = settings.KITE_ADMIN_USER_ID
    password = settings.KITE_ADMIN_PASSWORD
    totp_secret = settings.KITE_ADMIN_TOTP_SECRET

    if not all([api_key, api_secret, user_id, password, totp_secret]):
        logger.warning(
            "Kite auto-refresh skipped — missing credentials. "
            "Set KITE_ADMIN_USER_ID, KITE_ADMIN_PASSWORD, KITE_ADMIN_TOTP_SECRET in .env"
        )
        return False

    try:
        import pyotp
    except ImportError:
        logger.error("pyotp not installed. Run: pip install pyotp")
        return False

    login_url = f"https://kite.trade/connect/login?v=3&api_key={api_key}"

    try:
        session = http_requests.Session()

        # Step 1: GET login page to establish session cookies
        resp = session.get(login_url, allow_redirects=True, timeout=15)
        resp.raise_for_status()

        # Step 2: POST credentials
        login_payload = {
            "user_id": user_id,
            "password": password,
        }
        resp = session.post(
            "https://kite.zerodha.com/api/login",
            data=login_payload,
            timeout=15,
        )
        resp.raise_for_status()
        login_data = resp.json()

        if login_data.get("status") != "success":
            logger.error(f"Kite login failed: {login_data}")
            return False

        request_id = login_data["data"]["request_id"]

        # Step 3: POST TOTP for 2FA
        totp = pyotp.TOTP(totp_secret)
        twofa_payload = {
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp.now(),
            "twofa_type": "totp",
        }
        resp = session.post(
            "https://kite.zerodha.com/api/twofa",
            data=twofa_payload,
            timeout=15,
            allow_redirects=False,
        )

        # Step 4: Extract request_token from redirect URL
        # After 2FA, Kite redirects to: redirect_url?request_token=xxx&action=login&status=success
        if resp.status_code in (301, 302):
            redirect_url = resp.headers.get("Location", "")
        elif resp.status_code == 200:
            # Sometimes the redirect URL is in the JSON response
            twofa_data = resp.json()
            if twofa_data.get("status") == "success":
                # Follow the redirect manually
                redirect_url = twofa_data.get("data", {}).get("redirect_url", "")
                if not redirect_url:
                    # Try to get it via a follow-up GET
                    resp2 = session.get(
                        f"https://kite.trade/connect/login?v=3&api_key={api_key}",
                        allow_redirects=False,
                        timeout=15,
                    )
                    redirect_url = resp2.headers.get("Location", "")
            else:
                logger.error(f"Kite 2FA failed: {twofa_data}")
                return False
        else:
            logger.error(f"Kite 2FA unexpected status: {resp.status_code}")
            return False

        # Parse request_token from URL
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        request_token = params.get("request_token", [None])[0]

        if not request_token:
            logger.error(f"No request_token in redirect URL: {redirect_url}")
            return False

        # Step 5: Generate session (exchange request_token for access_token)
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        session_data = kite.generate_session(request_token, api_secret)
        new_access_token = session_data["access_token"]

        # Step 6: Update the admin client singleton
        client = get_kite_admin_client()
        client.set_access_token(new_access_token)

        logger.info(
            "Kite admin token auto-refreshed successfully. "
            "Valid until 6:00 AM IST tomorrow."
        )
        return True

    except Exception as e:
        logger.error(f"Kite auto-refresh failed: {e}", exc_info=True)
        return False
