"""
================================================================================
                    BROKER TICKER STREAMING
                    ========================

    Real-time market data via broker WebSocket APIs.
    Supports: Zerodha KiteTicker, Angel One SmartWebSocket, Upstox WS v2

    Data flow:
        BrokerTickerAdapter → BrokerTickerManager._on_tick()
            → PriceService.update_price_if_fresher()
            → ConnectionManager.broadcast_symbol_update()
            → Frontend WebSocket clients
================================================================================
"""

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def build_zerodha_tick_data(symbol: str, tick: dict) -> dict:
    """Pure builder for a canonical tick dict from a KiteTicker tick.
    Includes L2 ``depth`` only when the MODE_FULL tick carries usable levels."""
    ohlc = tick.get("ohlc", {}) or {}
    close = ohlc.get("close", 0) or 0
    change = tick.get("change", 0) or 0
    td = {
        "symbol": symbol,
        "ltp": tick.get("last_price", 0),
        "open": ohlc.get("open", 0),
        "high": ohlc.get("high", 0),
        "low": ohlc.get("low", 0),
        "change": change,
        "change_percent": (change / close * 100) if close > 0 else 0,
        "volume": tick.get("volume_traded", 0),
        "source": "broker",
    }
    raw_depth = tick.get("depth")
    if raw_depth:
        from .depth_models import from_kite_depth
        md = from_kite_depth(symbol, raw_depth)
        if md.levels() > 0:
            td["depth"] = md.to_dict()
    return td


def build_upstox_v3_tick_data(symbol: str, feed: dict) -> dict:
    """Pure builder for a canonical tick dict from one Upstox V3 ``feeds[key]``
    entry (mode full_d30). Attaches L2 ``depth`` (up to 30 levels) only when the
    marketLevel carries usable levels."""
    mff = (feed.get("ff", {}) or {}).get("marketFF", {}) or {}
    ltpc = mff.get("ltpc", {}) or {}
    ohlc_list = (mff.get("marketOHLC", {}) or {}).get("ohlc") or [{}]
    daily = ohlc_list[0] if ohlc_list else {}
    ltp = ltpc.get("ltp", 0) or 0
    close = ltpc.get("cp", 0) or 0
    change = round(ltp - close, 2) if close else 0
    td = {
        "symbol": symbol,
        "ltp": ltp,
        "open": daily.get("open", 0),
        "high": daily.get("high", 0),
        "low": daily.get("low", 0),
        "change": change,
        "change_percent": round((change / close) * 100, 2) if close else 0,
        "volume": daily.get("vol", 0),
        "source": "broker",
    }
    market_level = mff.get("marketLevel")
    if market_level:
        from .depth_models import from_upstox_marketlevel
        md = from_upstox_marketlevel(symbol, market_level)
        if md.levels() > 0:
            td["depth"] = md.to_dict()
    return td


class BrokerTickerSource(Enum):
    ZERODHA = "zerodha"
    ANGELONE = "angelone"
    UPSTOX = "upstox"


# ---------------------------------------------------------------------------
# Zerodha KiteTicker Adapter
# ---------------------------------------------------------------------------

class ZerodhaTickerAdapter:
    """
    Wraps kiteconnect.KiteTicker for real-time streaming.
    Runs in a background thread; bridges ticks to async via run_coroutine_threadsafe.
    """

    def __init__(self, api_key: str, access_token: str, on_tick: Callable, loop: asyncio.AbstractEventLoop):
        self.api_key = api_key
        self.access_token = access_token
        self._on_tick = on_tick
        self._loop = loop
        self._ticker = None
        self._thread: Optional[threading.Thread] = None
        self._instrument_cache: Dict[str, int] = {}  # symbol -> instrument_token
        self._subscribed_tokens: Set[int] = set()
        self.is_connected = False

    def _build_instrument_cache(self):
        """Fetch NSE instruments and build symbol->token lookup (cached for the day)."""
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(self.access_token)
            instruments = kite.instruments("NSE")
            self._instrument_cache = {
                inst["tradingsymbol"]: inst["instrument_token"]
                for inst in instruments
            }
            logger.info(f"Zerodha instrument cache built: {len(self._instrument_cache)} symbols")
        except Exception as e:
            logger.error(f"Failed to build Zerodha instrument cache: {e}")

    def connect(self):
        """Start KiteTicker in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        if not self._instrument_cache:
            self._build_instrument_cache()

        def _run():
            try:
                from kiteconnect import KiteTicker
                self._ticker = KiteTicker(self.api_key, self.access_token)

                def _on_ticks(ws, ticks):
                    for tick in ticks:
                        token = tick.get("instrument_token")
                        # Reverse lookup: token -> symbol
                        symbol = None
                        for sym, tok in self._instrument_cache.items():
                            if tok == token:
                                symbol = sym
                                break
                        if not symbol:
                            continue

                        tick_data = build_zerodha_tick_data(symbol, tick)
                        asyncio.run_coroutine_threadsafe(
                            self._on_tick(symbol, tick_data), self._loop
                        )

                def _on_connect(ws, response):
                    self.is_connected = True
                    logger.info("Zerodha KiteTicker connected")
                    # Re-subscribe any pending tokens
                    if self._subscribed_tokens:
                        ws.subscribe(list(self._subscribed_tokens))
                        ws.set_mode(ws.MODE_FULL, list(self._subscribed_tokens))

                def _on_close(ws, code, reason):
                    self.is_connected = False
                    logger.warning(f"Zerodha KiteTicker closed: {code} {reason}")

                def _on_error(ws, code, reason):
                    logger.error(f"Zerodha KiteTicker error: {code} {reason}")

                self._ticker.on_ticks = _on_ticks
                self._ticker.on_connect = _on_connect
                self._ticker.on_close = _on_close
                self._ticker.on_error = _on_error
                self._ticker.connect(threaded=False)
            except Exception as e:
                self.is_connected = False
                logger.error(f"Zerodha KiteTicker thread error: {e}")

        self._thread = threading.Thread(target=_run, daemon=True, name="zerodha-ticker")
        self._thread.start()

    def subscribe(self, symbols: List[str]):
        """Subscribe to symbols by converting to instrument tokens."""
        tokens = []
        for sym in symbols:
            token = self._instrument_cache.get(sym)
            if token:
                tokens.append(token)
                self._subscribed_tokens.add(token)
            else:
                logger.warning(f"Zerodha: no instrument token for {sym}")
        if tokens and self._ticker and self.is_connected:
            self._ticker.subscribe(tokens)
            self._ticker.set_mode(self._ticker.MODE_FULL, tokens)

    def unsubscribe(self, symbols: List[str]):
        tokens = []
        for sym in symbols:
            token = self._instrument_cache.get(sym)
            if token:
                tokens.append(token)
                self._subscribed_tokens.discard(token)
        if tokens and self._ticker and self.is_connected:
            self._ticker.unsubscribe(tokens)

    def disconnect(self):
        self.is_connected = False
        self._subscribed_tokens.clear()
        if self._ticker:
            try:
                self._ticker.close()
            except Exception:
                pass
            self._ticker = None


# ---------------------------------------------------------------------------
# Angel One SmartWebSocket Adapter
# ---------------------------------------------------------------------------

class AngelOneTickerAdapter:
    """
    Wraps SmartApi SmartWebSocket for Angel One streaming.
    Thread-based with async bridge.
    """

    def __init__(
        self,
        auth_token: str,
        api_key: str,
        client_code: str,
        feed_token: str,
        on_tick: Callable,
        loop: asyncio.AbstractEventLoop,
    ):
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self._on_tick = on_tick
        self._loop = loop
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._subscribed: Set[str] = set()
        self.is_connected = False

    def connect(self):
        if self._thread and self._thread.is_alive():
            return

        def _run():
            try:
                from SmartApi.smartWebSocketV2 import SmartWebSocketV2

                self._ws = SmartWebSocketV2(
                    self.auth_token,
                    self.api_key,
                    self.client_code,
                    self.feed_token,
                )

                def _on_data(ws, message):
                    try:
                        if not isinstance(message, dict):
                            return
                        symbol = message.get("token", "")
                        # Angel sends token as exchange_token; need reverse map
                        # For simplicity, use tradingsymbol if available
                        trading_symbol = message.get("name", symbol)

                        tick_data = {
                            "symbol": trading_symbol,
                            "ltp": float(message.get("last_traded_price", 0)) / 100,
                            "open": float(message.get("open_price_of_the_day", 0)) / 100,
                            "high": float(message.get("high_price_of_the_day", 0)) / 100,
                            "low": float(message.get("low_price_of_the_day", 0)) / 100,
                            "change": 0,
                            "change_percent": 0,
                            "volume": int(message.get("volume_trade_for_the_day", 0)),
                            "source": "broker",
                        }
                        # Calculate change
                        close = float(message.get("closed_price", 0)) / 100
                        if close > 0 and tick_data["ltp"] > 0:
                            tick_data["change"] = round(tick_data["ltp"] - close, 2)
                            tick_data["change_percent"] = round(
                                (tick_data["change"] / close) * 100, 2
                            )

                        asyncio.run_coroutine_threadsafe(
                            self._on_tick(trading_symbol, tick_data), self._loop
                        )
                    except Exception as e:
                        logger.debug(f"Angel tick parse error: {e}")

                def _on_open(ws):
                    self.is_connected = True
                    logger.info("Angel One WebSocket connected")

                def _on_error(ws, error):
                    logger.error(f"Angel One WebSocket error: {error}")

                def _on_close(ws):
                    self.is_connected = False
                    logger.warning("Angel One WebSocket closed")

                self._ws.on_data = _on_data
                self._ws.on_open = _on_open
                self._ws.on_error = _on_error
                self._ws.on_close = _on_close
                self._ws.connect()
            except Exception as e:
                self.is_connected = False
                logger.error(f"Angel One WebSocket thread error: {e}")

        self._thread = threading.Thread(target=_run, daemon=True, name="angelone-ticker")
        self._thread.start()

    def subscribe(self, symbols: List[str]):
        """
        Subscribe to symbols. Angel One expects token list in format:
        [{"exchangeType": 1, "tokens": ["1234", "5678"]}]
        exchangeType 1 = NSE_CM
        """
        if not self._ws or not self.is_connected:
            self._subscribed.update(symbols)
            return
        try:
            # For NSE equity, exchangeType=1
            token_list = [{"exchangeType": 1, "tokens": symbols}]
            self._ws.subscribe("abc123", 1, token_list)  # correlation_id, mode=LTP
            self._subscribed.update(symbols)
        except Exception as e:
            logger.error(f"Angel One subscribe error: {e}")

    def unsubscribe(self, symbols: List[str]):
        if not self._ws or not self.is_connected:
            self._subscribed -= set(symbols)
            return
        try:
            token_list = [{"exchangeType": 1, "tokens": symbols}]
            self._ws.unsubscribe("abc123", 1, token_list)
            self._subscribed -= set(symbols)
        except Exception as e:
            logger.error(f"Angel One unsubscribe error: {e}")

    def disconnect(self):
        self.is_connected = False
        self._subscribed.clear()
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass
            self._ws = None


# ---------------------------------------------------------------------------
# Upstox WebSocket Adapter
# ---------------------------------------------------------------------------

class UpstoxTickerAdapter:
    """Upstox V3 market-data streamer (mode full_d30 -> LTP/OHLC + 30-level depth).

    Drives upstox_client.MarketDataStreamerV3 (the SDK decodes protobuf). Runs the
    streamer in a background thread; bridges each feed to the async on_tick via
    run_coroutine_threadsafe. `streamer_factory` is injectable for tests."""

    def __init__(self, access_token: str, on_tick: Callable,
                 loop: asyncio.AbstractEventLoop, streamer_factory=None):
        self.access_token = access_token
        self._on_tick = on_tick
        self._loop = loop
        self._streamer = None
        self._streamer_factory = streamer_factory  # (token, keys, mode) -> streamer | None
        self._thread: Optional[threading.Thread] = None
        self._subscribed: Set[str] = set()
        self.is_connected = False

    def _make_streamer(self, keys):
        if self._streamer_factory is not None:
            return self._streamer_factory(self.access_token, keys, "full_d30")
        import upstox_client
        cfg = upstox_client.Configuration()
        cfg.access_token = self.access_token
        return upstox_client.MarketDataStreamerV3(
            upstox_client.ApiClient(cfg), list(keys), "full_d30")

    def _handle_message(self, message):
        try:
            feeds = (message or {}).get("feeds", {}) or {}
            for instrument_key, feed in feeds.items():
                parts = instrument_key.split("|")
                symbol = parts[1] if len(parts) > 1 else instrument_key
                tick_data = build_upstox_v3_tick_data(symbol, feed)
                asyncio.run_coroutine_threadsafe(self._on_tick(symbol, tick_data), self._loop)
        except Exception as e:
            logger.debug("Upstox V3 message parse error: %s", e)

    async def connect(self):
        if self._streamer is not None:
            return
        keys = list(self._subscribed) or []
        streamer = self._make_streamer(keys)
        self._streamer = streamer
        streamer.on("message", self._handle_message)

        def _run():
            try:
                streamer.connect()
                self.is_connected = True
            except Exception as e:
                self.is_connected = False
                logger.error("Upstox V3 streamer error: %s", e)

        self._thread = threading.Thread(target=_run, daemon=True, name="upstox-v3-ticker")
        self._thread.start()
        self.is_connected = True

    def _key(self, sym: str) -> str:
        return sym if "|" in sym else f"NSE_EQ|{sym}"

    async def subscribe(self, symbols: List[str]):
        keys = [self._key(s) for s in symbols]
        self._subscribed.update(keys)
        if self._streamer is not None:
            try:
                self._streamer.subscribe(keys, "full_d30")
            except Exception as e:
                logger.warning(
                    "Upstox V3 subscribe failed (dynamic subscribe may differ in this "
                    "SDK build — verify at smoke test): %s", e)

    async def unsubscribe(self, symbols: List[str]):
        keys = [self._key(s) for s in symbols]
        for k in keys:
            self._subscribed.discard(k)
        if self._streamer is not None:
            try:
                self._streamer.unsubscribe(keys)
            except Exception as e:
                logger.debug("Upstox V3 unsubscribe error: %s", e)

    async def disconnect(self):
        self.is_connected = False
        self._subscribed.clear()
        if self._streamer is not None:
            try:
                self._streamer.disconnect()
            except Exception:
                pass
            self._streamer = None


# ---------------------------------------------------------------------------
# Broker Ticker Manager
# ---------------------------------------------------------------------------

class BrokerTickerManager:
    """
    Manages per-user broker ticker connections and routes ticks into PriceService.

    - One broker connection per broker type (first connected user's credentials).
    - Symbol-level dedup: multiple users streaming same symbol don't cause duplicate broadcasts.
    - Falls through to Kite admin polling when no broker ticker is available.
    """

    def __init__(self, price_service, depth_bus=None, bar_sink=None):
        self._price_service = price_service
        self._depth_bus = depth_bus
        self._bar_sink = bar_sink  # callable(symbol, price, cum_volume) | None
        self._user_tickers: Dict[str, object] = {}           # user_id -> adapter instance
        self._user_broker_type: Dict[str, BrokerTickerSource] = {}
        self._active_by_type: Dict[BrokerTickerSource, str] = {}  # broker_type -> user_id (primary)
        self._symbol_sources: Dict[str, Set[str]] = {}       # symbol -> set of user_ids providing ticks
        self._last_tick_time: Dict[str, float] = {}           # symbol -> timestamp for dedup
        self._lock = asyncio.Lock()

    async def connect_user_ticker(
        self, user_id: str, broker_name: str, credentials: Dict
    ) -> bool:
        """
        Connect a user's broker ticker. Called when user connects to WebSocket
        and has an active broker connection.
        """
        async with self._lock:
            if user_id in self._user_tickers:
                return True  # Already connected

            source = BrokerTickerSource(broker_name.lower())
            loop = asyncio.get_event_loop()

            try:
                if source == BrokerTickerSource.ZERODHA:
                    adapter = ZerodhaTickerAdapter(
                        api_key=credentials.get("api_key", ""),
                        access_token=credentials.get("access_token", ""),
                        on_tick=self._make_tick_handler(user_id),
                        loop=loop,
                    )
                    adapter.connect()

                elif source == BrokerTickerSource.ANGELONE:
                    adapter = AngelOneTickerAdapter(
                        auth_token=credentials.get("jwt_token", credentials.get("jwtToken", "")),
                        api_key=credentials.get("api_key", ""),
                        client_code=credentials.get("client_id", ""),
                        feed_token=credentials.get("feed_token", credentials.get("feedToken", "")),
                        on_tick=self._make_tick_handler(user_id),
                        loop=loop,
                    )
                    adapter.connect()

                elif source == BrokerTickerSource.UPSTOX:
                    adapter = UpstoxTickerAdapter(
                        access_token=credentials.get("access_token", ""),
                        on_tick=self._make_tick_handler(user_id),
                        loop=loop,
                    )
                    await adapter.connect()

                else:
                    logger.warning(f"Unknown broker type: {broker_name}")
                    return False

                self._user_tickers[user_id] = adapter
                self._user_broker_type[user_id] = source

                # Track primary connection per broker type
                if source not in self._active_by_type:
                    self._active_by_type[source] = user_id

                logger.info(f"Broker ticker connected: user={user_id} broker={broker_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to connect broker ticker for {user_id}: {e}")
                return False

    async def disconnect_user_ticker(self, user_id: str):
        """Disconnect a user's broker ticker."""
        async with self._lock:
            adapter = self._user_tickers.pop(user_id, None)
            source = self._user_broker_type.pop(user_id, None)

            if adapter:
                try:
                    if isinstance(adapter, UpstoxTickerAdapter):
                        await adapter.disconnect()
                    else:
                        adapter.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting ticker for {user_id}: {e}")

            # Clean up primary tracking
            if source and self._active_by_type.get(source) == user_id:
                del self._active_by_type[source]
                # Promote another user of same broker type if available
                for uid, src in self._user_broker_type.items():
                    if src == source:
                        self._active_by_type[source] = uid
                        break

            # Remove user from symbol sources
            for sym in list(self._symbol_sources.keys()):
                self._symbol_sources[sym].discard(user_id)
                if not self._symbol_sources[sym]:
                    del self._symbol_sources[sym]

            logger.info(f"Broker ticker disconnected: user={user_id}")

    async def subscribe_symbols(self, user_id: str, symbols: List[str]):
        """Subscribe symbols on a user's broker ticker."""
        adapter = self._user_tickers.get(user_id)
        if not adapter:
            return

        for sym in symbols:
            self._symbol_sources.setdefault(sym, set()).add(user_id)

        try:
            if isinstance(adapter, UpstoxTickerAdapter):
                await adapter.subscribe(symbols)
            else:
                adapter.subscribe(symbols)
        except Exception as e:
            logger.error(f"Broker ticker subscribe error for {user_id}: {e}")

    async def unsubscribe_symbols(self, user_id: str, symbols: List[str]):
        """Unsubscribe symbols from a user's broker ticker."""
        adapter = self._user_tickers.get(user_id)

        for sym in symbols:
            if sym in self._symbol_sources:
                self._symbol_sources[sym].discard(user_id)
                if not self._symbol_sources[sym]:
                    del self._symbol_sources[sym]

        if not adapter:
            return

        try:
            if isinstance(adapter, UpstoxTickerAdapter):
                await adapter.unsubscribe(symbols)
            else:
                adapter.unsubscribe(symbols)
        except Exception as e:
            logger.error(f"Broker ticker unsubscribe error for {user_id}: {e}")

    def _make_tick_handler(self, user_id: str) -> Callable:
        """Create a tick handler closure for a specific user."""
        async def handler(symbol: str, tick_data: Dict):
            await self._on_tick(user_id, symbol, tick_data)
        return handler

    async def _on_tick(self, user_id: str, symbol: str, tick_data: Dict):
        """
        Internal tick handler. Deduplicates across sources and forwards to PriceService.
        """
        now = time.monotonic()
        last = self._last_tick_time.get(symbol, 0)

        # Dedup: skip if we got a tick for this symbol within 100ms
        if now - last < 0.1:
            return

        self._last_tick_time[symbol] = now

        if self._price_service:
            await self._price_service.update_price_if_fresher(symbol, tick_data)

        depth = tick_data.get("depth")
        if depth and self._depth_bus:
            await self._depth_bus.publish(symbol, depth)

        if self._bar_sink is not None:
            ltp = tick_data.get("ltp")
            if ltp:
                try:
                    self._bar_sink(symbol, float(ltp), float(tick_data.get("volume") or 0))
                except Exception as e:
                    logger.debug("bar_sink failed for %s: %s", symbol, e)

    def get_status(self) -> Dict:
        """Return status for health/admin endpoints."""
        return {
            "connected_users": len(self._user_tickers),
            "broker_types": {
                src.value: uid
                for src, uid in self._active_by_type.items()
            },
            "symbols_streaming": len(self._symbol_sources),
            "adapters": {
                uid: {
                    "broker": self._user_broker_type.get(uid, "unknown").value
                    if isinstance(self._user_broker_type.get(uid), BrokerTickerSource)
                    else "unknown",
                    "connected": getattr(adapter, "is_connected", False),
                }
                for uid, adapter in self._user_tickers.items()
            },
        }
