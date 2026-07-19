"""
================================================================================
SWING AI — YFINANCE DATA PROVIDER (Free, no API key needed)
================================================================================
Drop-in replacement for KiteDataProvider using yfinance for all market data.
Provides: historical OHLCV, real-time quotes, batch downloads, option chains.

Usage: Set DATA_PROVIDER=free (default) in .env to use this provider.
================================================================================
"""

import logging
import threading
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Period string → days mapping (matches yfinance-style period strings)
_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1w": 7, "1mo": 30, "3mo": 90,
    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 3650,
}

# Index name → yfinance ticker
INDEX_YF_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTY BANK": "^NSEBANK",
    "VIX": "^INDIAVIX",
    "INDIAVIX": "^INDIAVIX",
    "INDIA VIX": "^INDIAVIX",
    "NIFTYIT": "^CNXIT",
    "NIFTY IT": "^CNXIT",
    "NIFTYFIN": "^CNXFIN",
    "NIFTY FIN SERVICE": "^CNXFIN",
}

# yfinance interval mapping
_YF_INTERVAL_MAP = {
    "1m": "1m", "3m": "5m", "5m": "5m",
    "15m": "15m", "30m": "30m", "1h": "1h",
    "1d": "1d", "1wk": "1wk", "day": "1d",
}


def _yf_symbol(symbol: str) -> str:
    """Convert internal symbol to yfinance ticker."""
    stripped = symbol.strip()
    upper = stripped.upper()
    if upper in INDEX_YF_MAP:
        return INDEX_YF_MAP[upper]
    if stripped.startswith("^") or stripped.endswith(".NS") or stripped.endswith(".BO"):
        return stripped
    return f"{stripped}.NS"


def _clean_symbol(yf_ticker: str) -> str:
    """Convert yfinance ticker back to internal symbol."""
    return yf_ticker.replace(".NS", "").replace(".BO", "")


def _normalize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize yfinance DataFrame to standard OHLCV format."""
    if df is None or df.empty:
        return None

    # Handle multi-level columns from yfinance v1.2+ (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # For single-ticker download: columns are (Price, Ticker) — take first level
        df = df.droplevel(level=1, axis=1) if df.columns.nlevels == 2 else df
        # If still MultiIndex, flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Normalize column names to lowercase
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Map common yfinance column names
    col_map = {"adj close": "adj_close", "adjclose": "adj_close"}
    df = df.rename(columns=col_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.debug(f"Missing columns: {missing}, available: {list(df.columns)}")
        return None

    result = df[required].copy()
    result = result.dropna(subset=["close"])
    result["volume"] = result["volume"].fillna(0).astype(int)

    if result.empty:
        return None

    return result


class YFinanceProvider:
    """
    Market data provider using yfinance (free, no API key).
    Implements the same interface as KiteDataProvider.
    """

    def __init__(self):
        self._hist_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._quote_cache: Dict[str, Tuple[dict, float]] = {}
        self._hist_cache_ttl = 300  # 5 minutes
        self._quote_cache_ttl = 30  # 30 seconds
        self._yf = None
        # Thread-safety locks. yf.download() has been observed to interleave
        # responses under thread-pool concurrency, returning DataFrames with
        # duplicate columns (`['open', 'open', 'high', 'high', …]`). Serialise
        # the SDK call + use a per-cache-key lock to coalesce duplicate
        # requests for the same symbol.
        self._yf_lock = threading.Lock()
        self._cache_locks: Dict[str, threading.Lock] = {}
        self._cache_locks_guard = threading.Lock()
        # PR-AN — rate-limit circuit breaker. yfinance is free but
        # rate-limited; bursting through that limit poisons subsequent
        # calls for ~minutes. Track consecutive failures and apply
        # exponential backoff so the SL/target sweep doesn't pound the
        # endpoint when it's already throttling us.
        self._rl_fail_count = 0
        self._rl_blocked_until = 0.0
        logger.info("YFinanceProvider initialized (free data, no API key needed)")

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

        Even with the yf lock, return a structurally clean DataFrame so
        downstream code can rely on `df['close']` being a Series and
        `df.index.is_unique` being True.
        """
        if df is None or df.empty:
            return df
        if not df.columns.is_unique:
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]
        return df

    def _rl_should_skip(self) -> bool:
        """True when we're within the exponential-backoff window."""
        return time.time() < self._rl_blocked_until

    def _rl_record_failure(self):
        """Bump failure count + extend the backoff window. Caps at 60s
        so the breaker self-heals."""
        self._rl_fail_count = min(self._rl_fail_count + 1, 6)
        backoff = min(60.0, 1.5 * (2 ** self._rl_fail_count))
        self._rl_blocked_until = time.time() + backoff
        logger.warning(
            "yfinance rate-limit backoff: %d consecutive failures → "
            "skipping calls for %.1fs",
            self._rl_fail_count, backoff,
        )

    def _rl_record_success(self):
        """Reset the breaker on the first successful call."""
        if self._rl_fail_count > 0:
            logger.info("yfinance rate-limit recovered after %d failures",
                        self._rl_fail_count)
        self._rl_fail_count = 0
        self._rl_blocked_until = 0.0

    def _get_yf(self):
        """Lazy import yfinance."""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    # ── Cache helpers ──

    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        if cache_key in self._hist_cache:
            df, cached_at = self._hist_cache[cache_key]
            if (datetime.now() - cached_at).total_seconds() < self._hist_cache_ttl:
                return df
        return None

    def _set_cached(self, cache_key: str, df: pd.DataFrame):
        self._hist_cache[cache_key] = (df, datetime.now())

    def _get_quote_cached(self, symbol: str) -> Optional[dict]:
        if symbol in self._quote_cache:
            data, ts = self._quote_cache[symbol]
            if time.time() - ts < self._quote_cache_ttl:
                return data
        return None

    # ── Historical OHLCV ──

    def get_historical(
        self, symbol: str, period: str = "6mo", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV via yfinance.

        Thread-safe: per-cache-key lock coalesces concurrent calls for the
        same symbol; a global `_yf_lock` serialises the upstream `yf.download`
        SDK call which is not safe to invoke concurrently. The returned
        DataFrame is sanitised (no duplicate columns / index) so downstream
        code can rely on `df['close']` being a Series.
        """
        cache_key = f"{symbol}:{period}:{interval}"

        # Fast path — cache hit, sanitize on the way out
        cached = self._get_cached(cache_key)
        if cached is not None:
            return self._sanitize_columns(cached)

        # Slow path — per-key lock coalesces parallel requests for the
        # same symbol, then double-check the cache inside the lock.
        with self._key_lock(cache_key):
            cached = self._get_cached(cache_key)
            if cached is not None:
                return self._sanitize_columns(cached)

            yf = self._get_yf()
            yf_sym = _yf_symbol(symbol)
            yf_interval = _YF_INTERVAL_MAP.get(interval, "1d")

            try:
                # Serialise the SDK call — yf.download is not thread-safe
                # and interleaves responses under concurrent invocation,
                # producing DataFrames with duplicate columns.
                with self._yf_lock:
                    df = yf.download(
                        yf_sym, period=period, interval=yf_interval,
                        progress=False, auto_adjust=True, threads=False,
                    )
                result = _normalize_df(df)
                result = self._sanitize_columns(result)
                if result is not None and not result.empty:
                    self._set_cached(cache_key, result)
                    return result
            except Exception as e:
                logger.warning(f"yfinance historical failed for {symbol}: {e}")

        return None

    def get_historical_index(
        self, index_name: str, period: str = "6mo"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV for an index."""
        return self.get_historical(index_name, period, "1d")

    def fetch_historical_batch(
        self, symbols: List[str], period: str = "6mo"
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch OHLCV for many symbols using yfinance's multi-ticker download.
        Much faster than sequential — single HTTP call for all tickers.
        """
        results: Dict[str, pd.DataFrame] = {}
        if not symbols:
            return results

        # Check cache first, collect uncached
        uncached = []
        for sym in symbols:
            cache_key = f"{sym}:{period}:1d"
            cached = self._get_cached(cache_key)
            if cached is not None:
                results[sym] = cached
            else:
                uncached.append(sym)

        if not uncached:
            return results

        yf = self._get_yf()

        # Batch download in chunks of 100 to avoid URL length limits
        for i in range(0, len(uncached), 100):
            batch = uncached[i:i + 100]
            yf_tickers = [_yf_symbol(s) for s in batch]

            try:
                data = yf.download(
                    yf_tickers, period=period, group_by="ticker",
                    progress=False, auto_adjust=True, threads=True
                )

                if data is None or data.empty:
                    continue

                if len(batch) == 1:
                    # Single ticker: data has MultiIndex (field, ticker) in v1.2+
                    df = _normalize_df(data)
                    if df is not None and not df.empty:
                        sym = batch[0]
                        self._set_cached(f"{sym}:{period}:1d", df)
                        results[sym] = df
                else:
                    # Multiple tickers with group_by='ticker':
                    # level 0 = ticker, level 1 = field (Open, High, Low, Close, Volume)
                    for sym, yf_sym in zip(batch, yf_tickers):
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                top_level = data.columns.get_level_values(0).unique()
                                if yf_sym in top_level:
                                    ticker_df = data[yf_sym].copy()
                                else:
                                    continue
                            else:
                                continue
                            df = _normalize_df(ticker_df)
                            if df is not None and not df.empty:
                                self._set_cached(f"{sym}:{period}:1d", df)
                                results[sym] = df
                        except Exception:
                            continue

            except Exception as e:
                logger.warning(f"yfinance batch download failed for chunk {i}: {e}")
                # Fall back to individual downloads for this batch
                for sym in batch:
                    if sym not in results:
                        df = self.get_historical(sym, period)
                        if df is not None:
                            results[sym] = df

            # Small delay between chunks
            if i + 100 < len(uncached):
                time.sleep(1)

        logger.info(f"Batch fetch: {len(results)}/{len(symbols)} symbols loaded via yfinance")
        return results

    # ── Quote fetching ──

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get near-real-time quote via yfinance fast_info.
        Returns dict matching KiteDataProvider.get_quote() format.
        """
        cached = self._get_quote_cached(symbol)
        if cached is not None:
            return cached

        # Rate-limit circuit breaker: don't hammer yfinance when we're
        # already being throttled. Return cached-but-stale if available,
        # else None — callers should never crash on a missing quote.
        if self._rl_should_skip():
            if symbol in self._quote_cache:
                return self._quote_cache[symbol][0]
            return None

        yf = self._get_yf()
        yf_sym = _yf_symbol(symbol)

        try:
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info

            ltp = float(getattr(info, "last_price", 0) or 0)
            prev_close = float(getattr(info, "previous_close", 0) or 0)
            open_price = float(getattr(info, "open", 0) or 0)
            day_high = float(getattr(info, "day_high", 0) or 0)
            day_low = float(getattr(info, "day_low", 0) or 0)
            volume = int(getattr(info, "last_volume", 0) or 0)

            if ltp <= 0:
                # Try regular info as fallback
                ltp = float(getattr(info, "last_price", 0) or getattr(info, "regularMarketPrice", 0) or 0)

            if ltp <= 0:
                return None

            if prev_close <= 0:
                prev_close = ltp

            change = round(ltp - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close > 0 else 0.0

            quote = {
                "symbol": symbol,
                "ltp": round(ltp, 2),
                "open": round(open_price, 2),
                "high": round(day_high, 2),
                "low": round(day_low, 2),
                "close": round(prev_close, 2),  # previous close
                "volume": volume,
                "change": change,
                "change_percent": change_pct,
                "timestamp": datetime.now().isoformat(),
            }
            self._quote_cache[symbol] = (quote, time.time())
            self._rl_record_success()
            return quote

        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "too many" in err or "429" in err:
                self._rl_record_failure()
            logger.debug(f"yfinance quote failed for {symbol}: {e}")

        # Fallback: last bar from historical
        df = self.get_historical(symbol, period="5d")
        if df is not None and not df.empty:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            quote = {
                "symbol": symbol,
                "ltp": float(last["close"]),
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": float(prev["close"]),
                "volume": int(last["volume"]),
                "change": round(float(last["close"] - prev["close"]), 2),
                "change_percent": round(
                    float((last["close"] - prev["close"]) / prev["close"] * 100), 2
                ) if prev["close"] > 0 else 0,
                "timestamp": datetime.now().isoformat(),
            }
            self._quote_cache[symbol] = (quote, time.time())
            return quote

        return None

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batch quote fetch using yfinance download for today's data."""
        results = {}
        if not symbols:
            return results

        yf = self._get_yf()

        # Check cache first
        uncached = []
        for sym in symbols:
            cached = self._get_quote_cached(sym)
            if cached is not None:
                results[sym] = cached
            else:
                uncached.append(sym)

        if not uncached:
            return results

        # Batch download today's data
        try:
            yf_tickers = [_yf_symbol(s) for s in uncached]
            data = yf.download(
                yf_tickers, period="2d", progress=False,
                auto_adjust=True, threads=True
            )

            if data is not None and not data.empty:
                # Normalize the batch data
                for sym, yf_sym in zip(uncached, yf_tickers):
                    try:
                        if len(uncached) == 1:
                            ticker_df = _normalize_df(data)
                            if ticker_df is None:
                                continue
                        else:
                            if not isinstance(data.columns, pd.MultiIndex):
                                continue
                            # Single-ticker download: (Price, Ticker); multi: depends on group_by
                            top_level = data.columns.get_level_values(0).unique()
                            second_level = data.columns.get_level_values(1).unique()
                            if yf_sym in top_level:
                                ticker_df = data[yf_sym].copy()
                            elif yf_sym in second_level:
                                ticker_df = data.xs(yf_sym, level=1, axis=1).copy()
                            else:
                                continue
                            # Normalize columns to lowercase
                            ticker_df.columns = [str(c).lower().strip() for c in ticker_df.columns]

                        if ticker_df.empty or len(ticker_df) < 1:
                            continue

                        last = ticker_df.iloc[-1]
                        prev = ticker_df.iloc[-2] if len(ticker_df) > 1 else last

                        close_val = float(last.get("close", 0))
                        prev_close = float(prev.get("close", close_val))
                        change = round(close_val - prev_close, 2)
                        change_pct = round((change / prev_close) * 100, 2) if prev_close > 0 else 0

                        quote = {
                            "symbol": sym,
                            "ltp": round(close_val, 2),
                            "open": round(float(last.get("open", 0)), 2),
                            "high": round(float(last.get("high", 0)), 2),
                            "low": round(float(last.get("low", 0)), 2),
                            "close": round(prev_close, 2),
                            "volume": int(last.get("volume", 0)),
                            "change": change,
                            "change_percent": change_pct,
                        }
                        results[sym] = quote
                        self._quote_cache[sym] = (quote, time.time())
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"yfinance batch quote failed: {e}")
            # Fall back to individual quotes
            for sym in uncached:
                if sym not in results:
                    q = self.get_quote(sym)
                    if q:
                        results[sym] = q

        return results

    # ── Options Chain ──

    def get_option_chain(
        self, symbol: str, expiry: Optional[date] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch option chain via yfinance."""
        yf = self._get_yf()
        yf_sym = _yf_symbol(symbol)

        try:
            ticker = yf.Ticker(yf_sym)
            expirations = ticker.options
            if not expirations:
                return None

            # Pick requested expiry or nearest
            if expiry:
                expiry_str = expiry.strftime("%Y-%m-%d")
                if expiry_str in expirations:
                    target_exp = expiry_str
                else:
                    target_exp = expirations[0]
            else:
                target_exp = expirations[0]

            opt = ticker.option_chain(target_exp)
            chain = []

            for _, row in opt.calls.iterrows():
                chain.append({
                    "strike": float(row.get("strike", 0)),
                    "option_type": "CE",
                    "expiry": target_exp,
                    "ltp": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "oi": int(row.get("openInterest", 0)),
                    "volume": int(row.get("volume", 0)),
                    "iv": float(row.get("impliedVolatility", 0)) * 100,
                })

            for _, row in opt.puts.iterrows():
                chain.append({
                    "strike": float(row.get("strike", 0)),
                    "option_type": "PE",
                    "expiry": target_exp,
                    "ltp": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "oi": int(row.get("openInterest", 0)),
                    "volume": int(row.get("volume", 0)),
                    "iv": float(row.get("impliedVolatility", 0)) * 100,
                })

            return chain if chain else None

        except Exception as e:
            logger.warning(f"yfinance option chain failed for {symbol}: {e}")
            return None


# =============================================================================
# SINGLETON
# =============================================================================

_yfinance_provider: Optional[YFinanceProvider] = None


def get_yfinance_provider() -> YFinanceProvider:
    """Get or create singleton YFinanceProvider."""
    global _yfinance_provider
    if _yfinance_provider is None:
        _yfinance_provider = YFinanceProvider()
    return _yfinance_provider
