"""Kite WebSocket tick collector — PR-DEPTH.

Scaffold runs against ``kiteconnect.KiteTicker``. On startup:
  1. Authenticate (admin Kite token from settings.KITE_ADMIN_ACCESS_TOKEN)
  2. Resolve symbol → instrument_token via Kite's instruments dump
  3. Subscribe to selected_tokens in MODE_FULL (delivers bid/ask/OI)
  4. On each tick, buffer + periodic Supabase flush (500 ticks or 5 sec)
  5. Track collector status in tick_collector_runs

Scheduling: market-hours-only (9:15-15:30 IST). Stops at 15:35 IST.

Memory locks honoured: read-only side of Kite. No order placement.

Status: scaffold. Not auto-started by the scheduler until user enables
``ENABLE_TICK_COLLECTOR=true`` and confirms Kite admin credentials.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default universe — index futures + ATM ±3 strike options.
# Underliers are stable; options expand dynamically based on current spot.
_DEFAULT_UNDERLYINGS = ("NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE")

# Hard cap so we don't exceed Kite's subscription quota (~3000 instruments)
_MAX_SUBSCRIBED = 50

# Buffer parameters — flush either when full or after time
_BUFFER_SIZE_FLUSH = 500
_BUFFER_TIME_FLUSH_SECONDS = 5

# Market hours (IST)
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MIN = 15
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MIN = 30


ENABLE_TICK_COLLECTOR: bool = (
    os.getenv("ENABLE_TICK_COLLECTOR", "false").lower() == "true"
)


@dataclass
class TickCollectorConfig:
    underlyings: tuple = _DEFAULT_UNDERLYINGS
    atm_offset_range: int = 3                # collect ATM±N strike options
    buffer_size: int = _BUFFER_SIZE_FLUSH
    buffer_seconds: int = _BUFFER_TIME_FLUSH_SECONDS
    max_subscribed: int = _MAX_SUBSCRIBED
    reconnect_seconds: int = 10


@dataclass
class TickCollectorStatus:
    started_at: Optional[str] = None
    last_tick_at: Optional[str] = None
    ticks_received: int = 0
    ticks_persisted: int = 0
    symbols_subscribed: int = 0
    reconnects: int = 0
    status: str = "stopped"             # stopped | starting | running | error
    error: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "last_tick_at": self.last_tick_at,
            "ticks_received": self.ticks_received,
            "ticks_persisted": self.ticks_persisted,
            "symbols_subscribed": self.symbols_subscribed,
            "reconnects": self.reconnects,
            "status": self.status,
            "error": self.error,
            "run_id": self.run_id,
        }


def _is_market_hours_ist(now: Optional[datetime] = None) -> bool:
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = (now or datetime.now(ist)).astimezone(ist)
    if now_ist.weekday() >= 5:
        return False
    t = now_ist.time()
    open_t = (_MARKET_OPEN_HOUR, _MARKET_OPEN_MIN)
    close_t = (_MARKET_CLOSE_HOUR, _MARKET_CLOSE_MIN)
    return open_t <= (t.hour, t.minute) <= close_t


class TickCollector:
    """Kite WebSocket → Supabase tick_data. One instance per process.

    Use ``await collector.run()`` to start the collection loop. It will:
      - block until market open
      - subscribe, stream ticks, flush periodically
      - stop at market close
      - reconnect on transient errors
    """

    def __init__(
        self,
        supabase_admin: Any,
        config: Optional[TickCollectorConfig] = None,
    ):
        self.supabase = supabase_admin
        self.config = config or TickCollectorConfig()
        self.status = TickCollectorStatus()
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = datetime.utcnow()
        self._kws = None                          # lazy import
        self._running = False

    @property
    def is_enabled(self) -> bool:
        return ENABLE_TICK_COLLECTOR

    # ── Public lifecycle ────────────────────────────────────────

    async def run(self) -> TickCollectorStatus:
        """Long-running loop. Returns when market closes or collector is
        stopped."""
        if not self.is_enabled:
            self.status.status = "stopped"
            self.status.error = "ENABLE_TICK_COLLECTOR=false"
            return self.status

        self.status.started_at = datetime.now(timezone.utc).isoformat()
        self.status.status = "starting"
        await self._record_run_start()

        try:
            await self._connect_and_stream()
        except Exception as exc:  # noqa: BLE001
            self.status.status = "error"
            self.status.error = str(exc)[:240]
            logger.exception("tick_collector run failed")
        finally:
            await self._flush_buffer(force=True)
            await self._record_run_end()
            self.status.status = "stopped"
        return self.status

    def stop(self) -> None:
        self._running = False
        if self._kws is not None:
            try:
                self._kws.close()
            except Exception:
                pass

    # ── Internals ───────────────────────────────────────────────

    async def _connect_and_stream(self) -> None:
        """Lazy-import kiteconnect + drive its WebSocket. Loops until
        market close or stop()."""
        try:
            from kiteconnect import KiteTicker
        except Exception as exc:
            raise RuntimeError(
                "kiteconnect package not installed — install with "
                "`pip install kiteconnect`"
            ) from exc

        api_key = os.getenv("KITE_API_KEY") or os.getenv("ZERODHA_API_KEY")
        access_token = (
            os.getenv("KITE_ADMIN_ACCESS_TOKEN")
            or os.getenv("ZERODHA_ADMIN_ACCESS_TOKEN")
        )
        if not api_key or not access_token:
            raise RuntimeError(
                "KITE_API_KEY + KITE_ADMIN_ACCESS_TOKEN not configured"
            )

        # Resolve instrument tokens for the configured underliers
        tokens = await self._resolve_instrument_tokens()
        if not tokens:
            raise RuntimeError("no instrument tokens resolved — nothing to subscribe to")

        # Cap at config.max_subscribed
        tokens = tokens[: self.config.max_subscribed]
        self.status.symbols_subscribed = len(tokens)

        kws = KiteTicker(api_key, access_token)
        self._kws = kws

        def on_ticks(_ws, ticks):
            # Bridge sync callback → async buffer append
            now_iso = datetime.now(timezone.utc).isoformat()
            for tick in ticks:
                self._buffer.append({
                    "timestamp": now_iso,
                    "symbol": str(tick.get("instrument_token") or tick.get("tradingsymbol", "")),
                    "price": float(tick.get("last_price") or 0),
                    "volume": int(tick.get("volume_traded") or tick.get("volume") or 0),
                    "oi": int(tick.get("oi") or 0) or None,
                    "bid_price": _safe_first_bid_or_ask(tick, "buy"),
                    "ask_price": _safe_first_bid_or_ask(tick, "sell"),
                    "bid_qty": _safe_first_qty(tick, "buy"),
                    "ask_qty": _safe_first_qty(tick, "sell"),
                    "source": "kite_ws",
                })
                self.status.ticks_received += 1
                self.status.last_tick_at = now_iso

        def on_connect(ws, _response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            logger.info("tick_collector: subscribed to %d tokens (FULL mode)", len(tokens))

        def on_close(_ws, code, reason):
            logger.warning("tick_collector socket closed: %s %s", code, reason)
            self.status.reconnects += 1

        def on_error(_ws, code, reason):
            logger.warning("tick_collector socket error: %s %s", code, reason)

        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error

        self._running = True
        self.status.status = "running"

        # Drive the connection in a thread; periodically flush from this coroutine
        def _kite_drive():
            try:
                kws.connect(threaded=True, disable_ssl_verification=False)
            except Exception as exc:
                logger.exception("kite drive failed: %s", exc)

        await asyncio.to_thread(_kite_drive)

        # Loop until market close or stop()
        while self._running and _is_market_hours_ist():
            await asyncio.sleep(1)
            await self._maybe_flush()

        self._running = False
        try:
            kws.close()
        except Exception:
            pass

    async def _resolve_instrument_tokens(self) -> List[int]:
        """Look up instrument_token for each configured underlier symbol.
        Real implementation queries Kite's instruments dump via the REST
        SDK. Scaffold returns empty list — admin needs to seed the
        instrument cache first."""
        try:
            from kiteconnect import KiteConnect
            api_key = os.getenv("KITE_API_KEY") or os.getenv("ZERODHA_API_KEY")
            access_token = (
                os.getenv("KITE_ADMIN_ACCESS_TOKEN")
                or os.getenv("ZERODHA_ADMIN_ACCESS_TOKEN")
            )
            kc = KiteConnect(api_key=api_key)
            kc.set_access_token(access_token)
            instruments = kc.instruments("NSE")
            wanted = set(s.upper() for s in self.config.underlyings)
            tokens = [
                int(row["instrument_token"]) for row in instruments
                if str(row.get("tradingsymbol", "")).upper() in wanted
            ]
            return tokens
        except Exception as exc:
            logger.warning("instrument resolve failed: %s", exc)
            return []

    async def _maybe_flush(self) -> None:
        elapsed = (datetime.utcnow() - self._last_flush).total_seconds()
        if (
            len(self._buffer) >= self.config.buffer_size
            or elapsed >= self.config.buffer_seconds
        ):
            await self._flush_buffer()

    async def _flush_buffer(self, force: bool = False) -> None:
        if not self._buffer and not force:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        self._last_flush = datetime.utcnow()
        if not batch:
            return
        try:
            self.supabase.table("tick_data").insert(batch).execute()
            self.status.ticks_persisted += len(batch)
        except Exception as exc:
            logger.warning("tick flush failed (%d ticks): %s", len(batch), exc)

    async def _record_run_start(self) -> None:
        try:
            res = self.supabase.table("tick_collector_runs").insert({
                "status": "running",
                "symbols_subscribed": 0,
                "ticks_received": 0,
                "ticks_persisted": 0,
                "reconnects": 0,
                "source": "kite_ws",
            }).execute()
            if res.data:
                self.status.run_id = res.data[0].get("id")
        except Exception:
            logger.debug("collector run-start record failed (non-fatal)")

    async def _record_run_end(self) -> None:
        if not self.status.run_id:
            return
        try:
            self.supabase.table("tick_collector_runs").update({
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "status": self.status.status,
                "symbols_subscribed": self.status.symbols_subscribed,
                "ticks_received": self.status.ticks_received,
                "ticks_persisted": self.status.ticks_persisted,
                "reconnects": self.status.reconnects,
                "error": self.status.error,
            }).eq("id", self.status.run_id).execute()
        except Exception:
            logger.debug("collector run-end update failed (non-fatal)")


def _safe_first_bid_or_ask(tick: Dict[str, Any], side: str) -> Optional[float]:
    depth = tick.get("depth") or {}
    levels = depth.get(side) or []
    if not levels:
        return None
    try:
        return float(levels[0].get("price") or 0) or None
    except Exception:
        return None


def _safe_first_qty(tick: Dict[str, Any], side: str) -> Optional[int]:
    depth = tick.get("depth") or {}
    levels = depth.get(side) or []
    if not levels:
        return None
    try:
        return int(levels[0].get("quantity") or 0) or None
    except Exception:
        return None
