"""
================================================================================
QUANT X - MARKET DATA SERVICE (Kite Connect + jugaad-data)
================================================================================
Pluggable market data provider using admin Kite Connect for real-time OHLCV
and jugaad-data as secondary source when Kite token is expired.
Supports:
- Real-time quotes via Kite Connect
- Historical OHLCV data (Kite primary, jugaad-data secondary)
- Index data (Nifty, Bank Nifty, VIX)
- Batch price fetching
================================================================================
"""

import os
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio

import pandas as pd
from ..core.config import settings

logger = logging.getLogger(__name__)

# Approx NSE trading days per `period` string — used by the daily read-through
# so a long lookback isn't silently under-served from a short candles cache.
_PERIOD_TRADING_DAYS = {
    "1mo": 22, "3mo": 66, "6mo": 132, "1y": 252, "2y": 504,
    "5y": 1260, "10y": 2520, "max": 5000,
}

# Read-through freshness is decided by the trading calendar, not row count:
# see MarketDataProvider._cache_is_stale. A count-sufficient cache can still be
# weeks out of date if daily ingestion lags — observed 2026-06-22 with the NIFTY
# cache frozen at 06-09 (23242) while the live quote was 24102.9.


def _to_utc_index(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return ``df`` with a tz-aware UTC DatetimeIndex so cache-served and
    live-fetched frames are interchangeable. The candles cache yields UTC-aware
    timestamps while some live providers yield tz-naive ones; mixing them breaks
    downstream reindex/compare (e.g. regime features align VIX to Nifty by date).
    Best-effort — leaves ``df`` untouched if its index isn't datetime-like."""
    if df is None or len(df) == 0:
        return df
    try:
        idx = pd.to_datetime(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        out = df.copy()
        out.index = idx
        return out
    except Exception:
        return df

# ============================================================================
# CONSTANTS
# ============================================================================

# NSE Holiday list fallback (2025)
# NSE 2026 equity-segment trading holidays (weekday full-day closures).
# Primary source is NSE_HOLIDAYS_FILE (data/nse_holidays_2026.json); this is
# the in-code fallback used only when that file is missing/unreadable. Keep in
# sync with the JSON each year (source: NSE exchange-communication holidays).
NSE_HOLIDAYS_FALLBACK = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Ram Navami
    date(2026, 3, 31),   # Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
]


def _load_holidays_from_file(path: str) -> List[date]:
    """Load NSE holidays from JSON file."""
    try:
        if not path or not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = data.get("holidays", [])
        else:
            items = data
        return [date.fromisoformat(d) for d in items if isinstance(d, str)]
    except Exception as e:
        logger.warning(f"Failed to load holidays from {path}: {e}")
        return []


@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    ltp: float  # Last traded price
    open: float
    high: float
    low: float
    close: float  # Previous close
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class MarketStatus:
    """Market status information"""
    is_trading_day: bool
    is_market_open: bool
    market_phase: str  # PRE_OPEN, OPEN, CLOSED, HOLIDAY
    next_open: Optional[datetime] = None
    reason: str = ""


class MarketDataProvider:
    """
    Market data provider backed by admin Kite Connect + jugaad-data.

    Delegates all OHLCV fetching to KiteDataProvider which handles:
    - Real-time quotes via kite.quote()
    - Historical data via kite.historical_data() (primary) / jugaad-data (secondary)
    - Index data via Kite instrument tokens
    - Rate limiting and caching
    """

    def __init__(self):
        self._holidays: List[date] = _load_holidays_from_file(settings.NSE_HOLIDAYS_FILE) or NSE_HOLIDAYS_FALLBACK
        # Lazy import to avoid circular dependency
        self._kite_provider = None
        logger.info("MarketDataProvider initialized (Kite + jugaad-data backend)")

    def _get_kite_provider(self):
        """Lazy-load data provider based on DATA_PROVIDER setting."""
        if self._kite_provider is None:
            if settings.DATA_PROVIDER == "kite":
                from .providers.kite import get_kite_data_provider
                self._kite_provider = get_kite_data_provider()
            else:
                from .providers.yfinance import get_yfinance_provider
                self._kite_provider = get_yfinance_provider()
        return self._kite_provider

    # ========================================================================
    # TRADING DAY / HOLIDAY CHECKS
    # ========================================================================

    def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """Check if given date is a trading day."""
        check_date = check_date or date.today()

        # Weekend check
        if check_date.weekday() >= 5:
            return False

        # Holiday check
        if check_date in self._holidays:
            return False

        return True

    def _cache_is_stale(self, df: "pd.DataFrame") -> bool:
        """True when the newest cached daily candle is more than one trading
        session behind the latest NSE session. Row count can't catch a cache
        that stopped updating (NIFTY frozen at 06-09 while live was 24102.9);
        the trading calendar can. One session of slack absorbs today's
        as-yet-unclosed session so we don't re-fetch live on every call.
        Parsing hiccups fall back to "fresh" — never block a valid cache."""
        try:
            if df is None or len(df) == 0:
                return True
            newest = pd.Timestamp(pd.to_datetime(df["date"] if "date" in df.columns else df.index).max())
            newest_d = (newest.tz_convert("UTC") if newest.tzinfo is not None else newest).date()
            # Two most recent sessions on/before today; threshold = the older one.
            sessions: List[date] = []
            d = date.today()
            for _ in range(20):
                if self.is_trading_day(d):
                    sessions.append(d)
                    if len(sessions) == 2:
                        break
                d -= timedelta(days=1)
            threshold = sessions[-1] if sessions else newest_d
            return newest_d < threshold
        except Exception:
            return False

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        today = now.date()

        if not self.is_trading_day(today):
            return False

        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_market_status(self) -> MarketStatus:
        """Get detailed market status."""
        now = datetime.now()
        today = now.date()

        if today in self._holidays:
            return MarketStatus(
                is_trading_day=False,
                is_market_open=False,
                market_phase="HOLIDAY",
                next_open=self._get_next_trading_day_open(),
                reason="NSE Holiday"
            )

        if today.weekday() >= 5:
            return MarketStatus(
                is_trading_day=False,
                is_market_open=False,
                market_phase="CLOSED",
                next_open=self._get_next_trading_day_open(),
                reason="Weekend"
            )

        current_time = now.time()
        pre_open_start = datetime.strptime("09:00", "%H:%M").time()
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()

        if current_time < pre_open_start:
            return MarketStatus(
                is_trading_day=True,
                is_market_open=False,
                market_phase="PRE_MARKET",
                next_open=now.replace(hour=9, minute=15, second=0),
                reason="Market opens at 9:15 AM"
            )
        elif current_time < market_open:
            return MarketStatus(
                is_trading_day=True,
                is_market_open=False,
                market_phase="PRE_OPEN",
                next_open=now.replace(hour=9, minute=15, second=0),
                reason="Pre-open session"
            )
        elif current_time <= market_close:
            return MarketStatus(
                is_trading_day=True,
                is_market_open=True,
                market_phase="OPEN",
                reason="Market is open"
            )
        else:
            return MarketStatus(
                is_trading_day=True,
                is_market_open=False,
                market_phase="CLOSED",
                next_open=self._get_next_trading_day_open(),
                reason="Market closed for the day"
            )

    def _get_next_trading_day_open(self) -> datetime:
        """Get datetime of next market open."""
        check_date = date.today() + timedelta(days=1)
        while not self.is_trading_day(check_date):
            check_date += timedelta(days=1)
        return datetime.combine(check_date, datetime.strptime("09:15", "%H:%M").time())

    # ========================================================================
    # QUOTE FETCHING — delegates to KiteDataProvider
    # ========================================================================

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote for a symbol."""
        result = self._get_kite_provider().get_quote(symbol)
        if result is None:
            return None
        if isinstance(result, Quote):
            return result
        # Wrap dict → Quote dataclass
        try:
            return Quote(
                symbol=result.get("symbol", symbol),
                ltp=float(result.get("ltp", 0)),
                open=float(result.get("open", 0)),
                high=float(result.get("high", 0)),
                low=float(result.get("low", 0)),
                close=float(result.get("close", 0)),
                volume=int(result.get("volume", 0)),
                change=float(result.get("change", 0)),
                change_percent=float(result.get("change_percent", 0)),
                timestamp=datetime.now(),
            )
        except Exception:
            return None

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols (single batch call)."""
        raw = self._get_kite_provider().get_quotes_batch(symbols)
        result = {}
        for sym, data in raw.items():
            if data is None:
                continue
            if isinstance(data, Quote):
                result[sym] = data
            elif isinstance(data, dict):
                try:
                    result[sym] = Quote(
                        symbol=data.get("symbol", sym),
                        ltp=float(data.get("ltp", 0)),
                        open=float(data.get("open", 0)),
                        high=float(data.get("high", 0)),
                        low=float(data.get("low", 0)),
                        close=float(data.get("close", 0)),
                        volume=int(data.get("volume", 0)),
                        change=float(data.get("change", 0)),
                        change_percent=float(data.get("change_percent", 0)),
                        timestamp=datetime.now(),
                    )
                except Exception:
                    continue
        return result

    # ========================================================================
    # HISTORICAL DATA — delegates to KiteDataProvider
    # ========================================================================

    def get_historical(self, symbol: str, period: str = '6mo', interval: str = '1d',
                       bypass_store: bool = False) -> Optional[pd.DataFrame]:
        """Get historical OHLCV. For the daily interval, read-through the durable
        `candles` store (F1): store -> miss -> live provider -> backfill store.
        Honest: on any store error, fall through to the live provider (never fabricates)."""
        if interval == '1d' and not bypass_store:
            try:
                from .ohlc_store import read_candles, rows_to_df, df_to_candle_rows, upsert_candles
                from ..core.database import get_supabase_admin
                # Serve from cache ONLY if it actually spans the requested period;
                # otherwise treat as a miss and fetch live (don't under-serve a
                # long lookback from a short cache).
                need = _PERIOD_TRADING_DAYS.get(period, 132)
                sb = get_supabase_admin()
                cached = rows_to_df(read_candles(sb, symbol, '1d', limit=max(need + 20, 520)))
                if len(cached) >= need * 0.9 and not self._cache_is_stale(cached):
                    return _to_utc_index(cached)
                fresh = self._get_kite_provider().get_historical(symbol, period, interval)
                if fresh is not None and not fresh.empty:
                    upsert_candles(sb, df_to_candle_rows(symbol, fresh, '1d', 'live'))
                return _to_utc_index(fresh)
            except Exception as e:
                logger.debug("candles read-through failed for %s (%s); using live provider", symbol, e)
        return _to_utc_index(self._get_kite_provider().get_historical(symbol, period, interval))

    # ========================================================================
    # INDEX DATA
    # ========================================================================

    def get_index_data(self, index_name: str = 'NIFTY') -> Optional[Quote]:
        """Get index data (Nifty, Bank Nifty, VIX)."""
        return self.get_quote(index_name)

    def get_market_overview(self) -> Dict:
        """Get overall market data including indices and sentiment."""
        nifty = self.get_quote('NIFTY')
        banknifty = self.get_quote('BANKNIFTY')
        vix = self.get_quote('VIX')

        return {
            'nifty': {
                'ltp': nifty.ltp if nifty else 0,
                'change': nifty.change if nifty else 0,
                'change_percent': nifty.change_percent if nifty else 0,
            },
            'banknifty': {
                'ltp': banknifty.ltp if banknifty else 0,
                'change': banknifty.change if banknifty else 0,
                'change_percent': banknifty.change_percent if banknifty else 0,
            },
            'vix': {
                'ltp': vix.ltp if vix else 15,
                'change': vix.change if vix else 0,
                'change_percent': vix.change_percent if vix else 0,
            },
            'market_status': self.get_market_status().__dict__,
            'timestamp': datetime.now().isoformat(),
        }

    # ========================================================================
    # ASYNC WRAPPERS (for async code compatibility)
    # ========================================================================

    async def get_quote_async(self, symbol: str) -> Optional[Quote]:
        """Async wrapper for get_quote"""
        return await asyncio.to_thread(self.get_quote, symbol)

    async def get_quotes_batch_async(self, symbols: List[str]) -> Dict[str, Quote]:
        """Async wrapper for get_quotes_batch"""
        return await asyncio.to_thread(self.get_quotes_batch, symbols)

    async def get_historical_async(self, symbol: str, period: str = '6mo',
                                   interval: str = '1d') -> Optional[pd.DataFrame]:
        """Async wrapper for get_historical"""
        return await asyncio.to_thread(self.get_historical, symbol, period, interval)

    async def get_market_overview_async(self) -> Dict:
        """Async wrapper for get_market_overview"""
        return await asyncio.to_thread(self.get_market_overview)

    # ========================================================================
    # OPTIONS CHAIN DATA — delegates to KiteDataProvider
    # ========================================================================

    def get_option_chain(self, symbol: str, expiry: str = "") -> List[Dict]:
        """
        Get live options chain via admin Kite Connect.

        Returns ``[]`` when live data is unavailable — no synthetic fallback
        (no-fallbacks lock: never present an invented Black-Scholes chain as
        market truth). Callers surface an "OI feed unavailable" state.

        Data flow: KiteDataProvider.get_option_chain()
          → InstrumentTokenCache.get_nfo_options() for NFO instruments
          → kite.quote() in batches of 200 for live LTP/OI/depth
          → Newton-Raphson IV + Black-Scholes Greeks computed per contract
        """
        try:
            expiry_date = None
            if expiry:
                try:
                    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                except ValueError:
                    expiry_date = None

            provider = self._get_kite_provider()
            chain = provider.get_option_chain(symbol, expiry_date)
            if chain:
                return chain
        except Exception as e:
            logger.warning(f"Kite option chain failed for {symbol}: {e}")

        # No live chain → honest-empty. We do NOT fabricate a Black-Scholes
        # synthetic chain (no-fallbacks lock: never show invented market data).
        # All callers handle [] as "OI feed unavailable" (503 / skip).
        return []

    async def get_option_chain_async(self, symbol: str, expiry: str = "") -> List[Dict]:
        """Async wrapper for get_option_chain."""
        return await asyncio.to_thread(self.get_option_chain, symbol, expiry)

    # ========================================================================
    # L2 MARKET DEPTH (order book)
    # ========================================================================

    def get_depth(self, symbol: str):
        """Live L2 order-book depth (MarketDepth) via the admin Kite provider.
        Honest-None when no live feed is available (no fabrication). The Kite
        accessor can fall back to YFinance when Kite isn't configured — that
        provider has no depth feed, so guard instead of AttributeError-500ing
        the route (the stock page's order-book card polls this)."""
        provider = self._get_kite_provider()
        if not hasattr(provider, "get_depth"):
            return None
        try:
            return provider.get_depth(symbol)
        except Exception as e:  # noqa: BLE001
            logger.debug("get_depth failed for %s: %s", symbol, e)
            return None

    async def get_depth_async(self, symbol: str):
        """Async wrapper for get_depth."""
        return await asyncio.to_thread(self.get_depth, symbol)


# Singleton instance
_market_data_provider: Optional[MarketDataProvider] = None


def get_market_data_provider() -> MarketDataProvider:
    """Get or create singleton market data provider (Kite + jugaad-data)."""
    global _market_data_provider
    if _market_data_provider is None:
        _market_data_provider = MarketDataProvider()
    return _market_data_provider


# Alias for callsites that import ``MarketData`` (watchlist_live_routes,
# fo_strategies_routes, dossier_routes). Without this, those imports
# silently raise ImportError inside try/except blocks and the live-quote
# paths are permanently dead in production. Pinned here to keep the
# fan-out one symbol.
MarketData = MarketDataProvider
