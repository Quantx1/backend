"""
================================================================================
QUANT X - REAL-TIME WEBSOCKET SYSTEM
================================================================================
Production-grade real-time updates for:
- Live price updates
- Signal notifications
- Trade status updates
- Portfolio P&L updates
- Market data streaming
================================================================================
"""

import asyncio
import json
from collections import deque
from datetime import datetime
from typing import Dict, Deque, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from fastapi import WebSocket, WebSocketDisconnect
from redis import asyncio as aioredis
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# MESSAGE TYPES
# ============================================================================


class MessageType(Enum):
    # Connection
    CONNECTED = "connected"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"

    # Signals
    NEW_SIGNAL = "new_signal"
    SIGNAL_TRIGGERED = "signal_triggered"
    SIGNAL_EXPIRED = "signal_expired"

    # Trades
    TRADE_PENDING = "trade_pending"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"
    TRADE_REJECTED = "trade_rejected"

    # Positions
    POSITION_UPDATE = "position_update"
    SL_HIT = "sl_hit"
    TARGET_HIT = "target_hit"
    TRAILING_SL_UPDATE = "trailing_sl_update"

    # Portfolio
    PORTFOLIO_UPDATE = "portfolio_update"
    PNL_UPDATE = "pnl_update"
    MARGIN_ALERT = "margin_alert"

    # Market
    MARKET_DATA = "market_data"
    PRICE_UPDATE = "price_update"
    DEPTH_UPDATE = "depth_update"
    INTRADAY_SIGNAL = "intraday_signal"
    VIX_ALERT = "vix_alert"
    CIRCUIT_BREAKER = "circuit_breaker"

    # Notifications
    NOTIFICATION = "notification"
    ALERT = "alert"

    # ── PR 13 additions — regime / AI pipeline / system events ──
    REGIME_CHANGE = "regime_change"              # HMM detects bull↔sideways↔bear transition
    AUTO_TRADE_EXECUTED = "auto_trade_executed"  # FinRL-X auto-trader fired (Elite)
    AUTO_TRADE_BLOCKED = "auto_trade_blocked"    # kill-switch / risk gate blocked an auto-trade
    DEBATE_COMPLETED = "debate_completed"        # TradingAgents Bull/Bear finished on a signal
    KILL_SWITCH_FIRED = "kill_switch_fired"      # per-user or admin-global kill-switch activated
    KILL_SWITCH_CLEARED = "kill_switch_cleared"  # kill-switch lifted
    PAPER_SNAPSHOT_UPDATED = "paper_snapshot_updated"  # nightly paper snapshot written
    SENTIMENT_UPDATED = "sentiment_updated"      # sentiment refresh landed
    MODEL_PROMOTED = "model_promoted"            # admin promoted a shadow model to prod


@dataclass
class WSMessage:
    type: MessageType
    data: Dict
    timestamp: str = None
    user_id: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp
        })

# ============================================================================
# CONNECTION MANAGER
# ============================================================================


class ConnectionManager:
    """
    Manages WebSocket connections with Redis pub/sub for scaling
    """

    # Important messages (signals, trades, alerts) are queued for offline users.
    # Price updates and pings are ephemeral and NOT queued.
    _QUEUEABLE_TYPES: Set[str] = {
        MessageType.NEW_SIGNAL.value,
        MessageType.SIGNAL_TRIGGERED.value,
        MessageType.TRADE_EXECUTED.value,
        MessageType.TRADE_CLOSED.value,
        MessageType.TRADE_REJECTED.value,
        MessageType.SL_HIT.value,
        MessageType.TARGET_HIT.value,
        MessageType.NOTIFICATION.value,
        MessageType.ALERT.value,
        MessageType.MARGIN_ALERT.value,
        # PR 13 additions — all are high-signal events users should see on reconnect.
        MessageType.REGIME_CHANGE.value,
        MessageType.AUTO_TRADE_EXECUTED.value,
        MessageType.AUTO_TRADE_BLOCKED.value,
        MessageType.KILL_SWITCH_FIRED.value,
    }
    _MAX_QUEUED_PER_USER = 50  # cap to prevent unbounded memory growth

    def __init__(self, redis_url: str = None):
        # Local connections (per server instance)
        self.active_connections: Dict[str, WebSocket] = {}

        # User subscriptions (what each user is subscribed to)
        self.user_subscriptions: Dict[str, Set[str]] = {}

        # Symbol subscriptions (which users are watching which symbols)
        self.symbol_watchers: Dict[str, Set[str]] = {}

        # Offline message queue: messages buffered while user is disconnected
        self._offline_queue: Dict[str, Deque[str]] = {}

        # Redis for cross-server messaging
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub = None

    async def init_redis(self):
        """Initialize Redis connection for pub/sub"""
        if self.redis_url:
            try:
                self.redis = await aioredis.from_url(self.redis_url)
                self.pubsub = self.redis.pubsub()
                await self.pubsub.subscribe("quantx:broadcast")
                logger.info("Redis pub/sub initialized")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")

    async def connect(self, websocket: WebSocket, user_id: str) -> bool:
        """Accept new WebSocket connection and deliver queued messages."""
        try:
            await websocket.accept()
            self.active_connections[user_id] = websocket
            self.user_subscriptions[user_id] = set(["global"])

            # Send connection confirmation
            await self.send_to_user(user_id, WSMessage(
                type=MessageType.CONNECTED,
                data={"user_id": user_id, "status": "connected"}
            ))

            # Flush any messages queued while the user was offline
            await self._flush_offline_queue(user_id)

            logger.info(f"WebSocket connected: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Connection failed for {user_id}: {e}")
            return False

    def disconnect(self, user_id: str):
        """Handle disconnection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]

        if user_id in self.user_subscriptions:
            del self.user_subscriptions[user_id]

        # Remove from symbol watchers
        for symbol, watchers in self.symbol_watchers.items():
            watchers.discard(user_id)

        logger.info(f"WebSocket disconnected: {user_id}")

    async def send_to_user(self, user_id: str, message: WSMessage):
        """Send message to a specific user, queuing important messages if offline."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message.to_json())
                return
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")
                self.disconnect(user_id)

        # User is offline — queue important messages for later delivery
        if message.type.value in self._QUEUEABLE_TYPES:
            queue = self._offline_queue.setdefault(user_id, deque(maxlen=self._MAX_QUEUED_PER_USER))
            queue.append(message.to_json())

    async def _flush_offline_queue(self, user_id: str):
        """Deliver all queued messages to a newly connected user."""
        queue = self._offline_queue.pop(user_id, None)
        if not queue:
            return
        ws = self.active_connections.get(user_id)
        if not ws:
            return
        count = len(queue)
        for msg_json in queue:
            try:
                await ws.send_text(msg_json)
            except Exception:
                break
        if count:
            logger.info(f"Flushed {count} queued messages to {user_id}")

    async def broadcast(self, message: WSMessage):
        """Broadcast to all connected users"""
        disconnected = []

        for user_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message.to_json())
            except Exception as e:
                logger.error(f"Broadcast failed for {user_id}: {e}")
                disconnected.append(user_id)

        # Cleanup disconnected
        for user_id in disconnected:
            self.disconnect(user_id)

        # Also publish to Redis for other server instances
        if self.redis:
            await self.redis.publish("quantx:broadcast", message.to_json())

    async def broadcast_to_subscribers(self, channel: str, message: WSMessage):
        """Broadcast to users subscribed to a channel"""
        for user_id, subs in self.user_subscriptions.items():
            if channel in subs:
                await self.send_to_user(user_id, message)

    async def broadcast_symbol_update(self, symbol: str, message: WSMessage):
        """Broadcast to users watching a specific symbol"""
        if symbol in self.symbol_watchers:
            for user_id in self.symbol_watchers[symbol]:
                await self.send_to_user(user_id, message)

    def subscribe_to_symbol(self, user_id: str, symbol: str):
        """Subscribe user to symbol updates"""
        if symbol not in self.symbol_watchers:
            self.symbol_watchers[symbol] = set()
        self.symbol_watchers[symbol].add(user_id)

    def unsubscribe_from_symbol(self, user_id: str, symbol: str):
        """Unsubscribe user from symbol updates"""
        if symbol in self.symbol_watchers:
            self.symbol_watchers[symbol].discard(user_id)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def is_connected(self, user_id: str) -> bool:
        """Check if user is connected"""
        return user_id in self.active_connections

# ============================================================================
# REAL-TIME PRICE SERVICE
# ============================================================================


class PriceService:
    """
    Real-time price updates service
    Integrates with broker APIs for live prices
    """

    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.price_cache: Dict[str, Dict] = {}
        self.last_update: Dict[str, datetime] = {}

    async def update_price(self, symbol: str, price_data: Dict):
        """
        Update price and broadcast to watchers

        price_data: {
            "symbol": "TRENT",
            "ltp": 2765.50,
            "open": 2745.00,
            "high": 2780.00,
            "low": 2740.00,
            "change": 20.50,
            "change_percent": 0.75,
            "volume": 125000
        }
        """
        self.price_cache[symbol] = price_data
        self.last_update[symbol] = datetime.utcnow()

        message = WSMessage(
            type=MessageType.PRICE_UPDATE,
            data=price_data
        )

        await self.manager.broadcast_symbol_update(symbol, message)

    async def update_position_pnl(self, user_id: str, positions: List[Dict]):
        """
        Update position P&L with live prices
        """
        updated_positions = []

        from ..trading.pnl import compute_unrealized_pnl
        for pos in positions:
            symbol = pos["symbol"]
            if symbol in self.price_cache:
                current_price = self.price_cache[symbol]["ltp"]
                pnl, pnl_percent = compute_unrealized_pnl(
                    direction=pos["direction"],
                    average_price=pos["average_price"],
                    current_price=current_price,
                    quantity=pos["quantity"],
                )
                updated_positions.append({
                    **pos,
                    "current_price": current_price,
                    "unrealized_pnl": round(pnl, 2),
                    "unrealized_pnl_percent": round(pnl_percent, 2)
                })

        message = WSMessage(
            type=MessageType.POSITION_UPDATE,
            data={"positions": updated_positions}
        )

        await self.manager.send_to_user(user_id, message)

    async def start_polling(self, interval: int = 30):
        """Poll prices via Kite Connect for subscribed symbols."""
        while True:
            try:
                symbols = list(self.manager.symbol_watchers.keys())
                if not symbols:
                    await asyncio.sleep(interval)
                    continue

                # Use market data provider (delegates to Kite)
                from ..data.market import get_market_data_provider
                provider = get_market_data_provider()
                quotes = provider.get_quotes_batch(symbols[:50])

                for sym, quote in quotes.items():
                    try:
                        if quote:
                            price_data = {
                                "symbol": sym,
                                "ltp": round(float(quote.ltp), 2),
                                "open": round(float(quote.open), 2),
                                "high": round(float(quote.high), 2),
                                "low": round(float(quote.low), 2),
                                "change": round(float(quote.change), 2),
                                "change_percent": round(float(quote.change_percent), 2),
                                "volume": int(quote.volume),
                            }
                            await self.update_price_if_fresher(sym, price_data)
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"Price polling error: {e}")

            await asyncio.sleep(interval)

    async def update_price_if_fresher(self, symbol: str, price_data: Dict):
        """
        Only update if this tick is newer than the last update for this symbol.
        Broker ticks have priority over polling within 100ms window.
        """
        last = self.last_update.get(symbol)
        if last and (datetime.utcnow() - last).total_seconds() < 0.1:
            # Within 100ms window — only allow broker source through
            if price_data.get("source") != "broker":
                return
        await self.update_price(symbol, price_data)

    def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price for symbol"""
        if symbol in self.price_cache:
            return self.price_cache[symbol].get("ltp")
        return None

# ============================================================================
# NOTIFICATION SERVICE
# ============================================================================


class NotificationService:
    """
    Real-time notification service
    """

    def __init__(self, connection_manager: ConnectionManager, supabase_client):
        self.manager = connection_manager
        self.supabase = supabase_client
        self.telegram_token = settings.TELEGRAM_BOT_TOKEN

        # Web Push + Email services (initialized lazily, fail-safe)
        from .push import WebPushService, EmailService
        self.push_service = WebPushService(
            vapid_private_key=settings.VAPID_PRIVATE_KEY,
            vapid_claims_email=settings.VAPID_CLAIMS_EMAIL,
        )
        self.email_service = EmailService(
            api_key=settings.RESEND_API_KEY,
            from_email=settings.EMAIL_FROM,
        )

    async def send_signal_notification(self, signal: Dict, user_ids: List[str] = None):
        """Send new signal notification"""
        message = WSMessage(
            type=MessageType.NEW_SIGNAL,
            data={
                "signal": signal,
                "title": f"New {signal['direction']} Signal: {signal['symbol']}",
                "message": f"Confidence: {signal['confidence']}% | Entry: ₹{signal['entry_price']}"
            }
        )

        title = message.data.get("title", "")
        body = message.data.get("message", "")

        if user_ids:
            for user_id in user_ids:
                await self.manager.send_to_user(user_id, message)
                await self._send_telegram_if_enabled(
                    user_id, title, body, event_key="new_signal",
                )
                await self._send_push_if_enabled(
                    user_id, title, body,
                    {"type": "signal", "symbol": signal.get("symbol", "")},
                    event_key="new_signal",
                )
        else:
            await self.manager.broadcast(message)
            await self._broadcast_telegram(title, body)
            await self._broadcast_push(title, body, {"type": "signal", "symbol": signal.get("symbol", "")})

    async def send_trade_notification(self, user_id: str, trade: Dict, status: str):
        """Send trade status notification"""
        type_map = {
            "pending": MessageType.TRADE_PENDING,
            "executed": MessageType.TRADE_EXECUTED,
            "closed": MessageType.TRADE_CLOSED,
            "rejected": MessageType.TRADE_REJECTED
        }

        title = f"Trade {status.capitalize()}: {trade['symbol']}"
        body = (
            f"{trade['direction']} {trade['quantity']} @ "
            f"₹{trade.get('average_price', trade['entry_price'])}"
        )
        message = WSMessage(
            type=type_map.get(status, MessageType.NOTIFICATION),
            data={"trade": trade, "title": title, "message": body})

        await self.manager.send_to_user(user_id, message)

        # Also save to database
        await self._save_notification(user_id, message)

    async def broadcast_signals(self, signals: List[Dict]):
        """Broadcast a batch of signals to all connected users"""
        for signal in signals:
            payload = signal.__dict__ if hasattr(signal, "__dict__") else signal
            await self.send_signal_notification(payload)

    async def broadcast_alert(self, title: str, message: str, priority: str = "normal"):
        """Broadcast a general alert to all users"""
        alert = WSMessage(
            type=MessageType.ALERT,
            data={
                "title": title,
                "message": message,
                "priority": priority,
            },
        )
        await self.manager.broadcast(alert)

    async def send_admin_alert(self, title: str, message: str):
        """Send admin alert (broadcast for now)"""
        await self.broadcast_alert(title, message, priority="high")

    async def send_to_user(
        self,
        user_id: str,
        notif_type: str,
        message: str,
        title: Optional[str] = None,
        data: Optional[Dict] = None,
    ):
        """Send a generic notification to a user and persist it"""
        payload = {
            "type": notif_type,
            "title": title or "Notification",
            "message": message,
            "priority": "normal",
        }
        if data:
            payload["data"] = data

        ws_message = WSMessage(type=MessageType.NOTIFICATION, data=payload)
        await self.manager.send_to_user(user_id, ws_message)
        await self._save_notification(user_id, ws_message)
        await self._send_telegram_if_enabled(user_id, payload.get("title", ""), payload.get("message", ""))

    async def send_daily_summary(self, user_id: str):
        """Send a lightweight daily summary notification"""
        try:
            profile = self.supabase.table("user_profiles").select(
                "total_trades, winning_trades, total_pnl"
            ).eq("id", user_id).single().execute()
            data = profile.data or {}
            total = data.get("total_trades", 0)
            wins = data.get("winning_trades", 0)
            win_rate = (wins / total * 100) if total else 0
            total_pnl = data.get("total_pnl", 0)

            await self.send_to_user(
                user_id,
                "daily_summary",
                f"Trades: {total} | Win rate: {win_rate:.1f}% | P&L: ₹{total_pnl:,.0f}",
                title="Daily Summary",
            )

            # Send daily summary email
            await self._send_email_daily_summary(user_id, {
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "trades_closed": total,
                "open_positions": 0,
            })
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

    async def send_sl_alert(self, user_id: str, position: Dict):
        """Send stop loss hit alert"""
        message = WSMessage(
            type=MessageType.SL_HIT,
            data={
                "position": position,
                "title": f"⚠️ Stop Loss Hit: {position['symbol']}",
                "message": f"Position closed at ₹{position.get('exit_price', position['stop_loss'])}",
                "priority": "high"
            }
        )

        await self.manager.send_to_user(user_id, message)
        await self._save_notification(user_id, message)
        await self._send_push_if_enabled(
            user_id, message.data["title"], message.data["message"],
            {"type": "sl_hit", "symbol": position.get("symbol", "")},
            event_key="sl_hit",
        )
        await self._send_telegram_if_enabled(
            user_id, message.data["title"], message.data["message"],
            event_key="sl_hit",
        )
        await self._send_email_position_alert(user_id, position, "sl_hit")

    async def send_target_alert(self, user_id: str, position: Dict):
        """Send target hit alert"""
        message = WSMessage(
            type=MessageType.TARGET_HIT,
            data={
                "position": position,
                "title": f"🎯 Target Hit: {position['symbol']}",
                "message": f"Position closed at ₹{position.get('exit_price', position['target'])}",
                "priority": "high"
            }
        )

        await self.manager.send_to_user(user_id, message)
        await self._save_notification(user_id, message)
        await self._send_push_if_enabled(
            user_id, message.data["title"], message.data["message"],
            {"type": "target_hit", "symbol": position.get("symbol", "")},
            event_key="target_hit",
        )
        await self._send_telegram_if_enabled(
            user_id, message.data["title"], message.data["message"],
            event_key="target_hit",
        )
        await self._send_email_position_alert(user_id, position, "target_hit")

    async def send_vix_alert(self, vix_level: float):
        """Send VIX alert to all users.

        Uses the shared ``classify_risk_band`` so the alert severity
        matches what users see on ``/api/market/risk`` and what the
        scheduler stores in ``market_data.risk_level``. Previously
        this method had its own thresholds (>25 to fire, >30 for
        EXTREME) that drifted from the canonical bands and from the
        on-demand risk endpoint.
        """
        from ..data.market_overview import classify_risk_band
        band = classify_risk_band(vix_level)
        if band not in ("HIGH", "EXTREME"):
            return  # LOW / MODERATE — no alert
        message_body = (
            "Trading paused due to extreme volatility."
            if band == "EXTREME"
            else "High volatility detected. Position sizes reduced."
        )
        message = WSMessage(
            type=MessageType.VIX_ALERT,
            data={
                "vix": vix_level,
                "severity": band,
                "title": f"⚠️ VIX Alert: {vix_level}",
                "message": message_body,
                "priority": "urgent",
            },
        )
        await self.manager.broadcast(message)

    async def send_margin_alert(self, user_id: str, margin_data: Dict):
        """Send margin warning"""
        message = WSMessage(
            type=MessageType.MARGIN_ALERT,
            data={
                **margin_data,
                "title": "⚠️ Margin Warning",
                "message": f"Available margin: ₹{margin_data['available']:,.0f} ({margin_data['percent']:.1f}%)",
                "priority": "high"
            }
        )

        await self.manager.send_to_user(user_id, message)

    async def _save_notification(self, user_id: str, message: WSMessage):
        """Save notification to database"""
        try:
            self.supabase.table("notifications").insert({
                "user_id": user_id,
                "type": message.type.value,
                "title": message.data.get("title", ""),
                "message": message.data.get("message", ""),
                "data": message.data,
                "priority": message.data.get("priority", "normal")
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save notification: {e}")

    async def _send_telegram_if_enabled(
        self,
        user_id: str,
        title: str,
        body: str,
        event_key: Optional[str] = None,
    ):
        if not self.telegram_token:
            return
        # PR 41 — per-event Alerts Studio gate. event_key=None preserves
        # the legacy blanket-flag behavior for generic emitters.
        if event_key is not None:
            try:
                from .alerts import channels_for_event
                channels = await channels_for_event(user_id, event_key, supabase_client=self.supabase)
                if "telegram" not in channels:
                    return
            except Exception as exc:
                logger.debug("telegram channel-gate check skipped: %s", exc)
        try:
            profile = self.supabase.table("user_profiles").select(
                "telegram_chat_id, telegram_connected, notifications_enabled"
            ).eq("id", user_id).single().execute()
            data = profile.data or {}
            chat_id = data.get("telegram_chat_id")
            if not chat_id or not data.get("notifications_enabled"):
                return
            await self._send_telegram(chat_id, f"{title}\n{body}")
        except Exception as e:
            logger.debug(f"Telegram send failed: {e}")

    async def _broadcast_telegram(self, title: str, body: str):
        if not self.telegram_token:
            return
        try:
            result = self.supabase.table("user_profiles").select(
                "id, telegram_chat_id"
            ).neq("telegram_chat_id", None).execute()
            for row in result.data or []:
                await self._send_telegram(row["telegram_chat_id"], f"{title}\n{body}")
        except Exception as e:
            logger.debug(f"Telegram broadcast failed: {e}")

    async def _send_telegram(self, chat_id: str, text: str):
        if not self.telegram_token:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        async with httpx.AsyncClient() as client:
            await client.post(url, json={"chat_id": chat_id, "text": text})

    # ---- Web Push delivery ----

    async def _send_push_if_enabled(
        self,
        user_id: str,
        title: str,
        body: str,
        data: Dict = None,
        event_key: Optional[str] = None,
    ):
        """Send Web Push to all user's registered devices."""
        if not self.push_service.is_available:
            return
        # PR 41 — per-event Alerts Studio gate.
        if event_key is not None:
            try:
                from .alerts import channels_for_event
                channels = await channels_for_event(user_id, event_key, supabase_client=self.supabase)
                if "push" not in channels:
                    return
            except Exception as exc:
                logger.debug("push channel-gate check skipped: %s", exc)
        try:
            profile = self.supabase.table("user_profiles").select(
                "push_notifications, notifications_enabled"
            ).eq("id", user_id).single().execute()
            pdata = profile.data or {}
            if not pdata.get("push_notifications", True) or not pdata.get("notifications_enabled", True):
                return

            subs = self.supabase.table("push_subscriptions").select(
                "id, endpoint, p256dh, auth"
            ).eq("user_id", user_id).execute()

            for sub in (subs.data or []):
                sub_info = {
                    "endpoint": sub["endpoint"],
                    "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
                }
                try:
                    await self.push_service.send(sub_info, title, body, data)
                except Exception as push_err:
                    # 410 Gone = subscription expired, clean up
                    if hasattr(push_err, "response") and getattr(push_err.response, "status_code", 0) == 410:
                        self.supabase.table("push_subscriptions").delete().eq("id", sub["id"]).execute()
                        logger.info(f"Removed expired push subscription {sub['id']}")
        except Exception as e:
            logger.debug(f"Push send failed for {user_id}: {e}")

    async def _broadcast_push(self, title: str, body: str, data: Dict = None):
        """Send Web Push to ALL users with push enabled."""
        if not self.push_service.is_available:
            return
        try:
            subs = self.supabase.table("push_subscriptions").select(
                "id, user_id, endpoint, p256dh, auth"
            ).execute()
            for sub in (subs.data or []):
                sub_info = {
                    "endpoint": sub["endpoint"],
                    "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
                }
                try:
                    await self.push_service.send(sub_info, title, body, data)
                except Exception as push_err:
                    if hasattr(push_err, "response") and getattr(push_err.response, "status_code", 0) == 410:
                        self.supabase.table("push_subscriptions").delete().eq("id", sub["id"]).execute()
        except Exception as e:
            logger.debug(f"Push broadcast failed: {e}")

    # ---- Email delivery ----

    async def _send_email_if_enabled(
        self,
        user_id: str,
        subject: str,
        html: str,
        event_key: Optional[str] = None,
    ):
        """Send email if user has email_notifications enabled."""
        if not self.email_service.is_available:
            return
        # PR 41 — per-event Alerts Studio gate.
        if event_key is not None:
            try:
                from .alerts import channels_for_event
                channels = await channels_for_event(user_id, event_key, supabase_client=self.supabase)
                if "email" not in channels:
                    return
            except Exception as exc:
                logger.debug("email channel-gate check skipped: %s", exc)
        try:
            profile = self.supabase.table("user_profiles").select(
                "email, email_notifications, notifications_enabled"
            ).eq("id", user_id).single().execute()
            pdata = profile.data or {}
            if not pdata.get("email_notifications", True) or not pdata.get("notifications_enabled", True):
                return
            email = pdata.get("email")
            if not email:
                return
            await self.email_service.send(email, subject, html)
        except Exception as e:
            logger.debug(f"Email send failed for {user_id}: {e}")

    async def _send_email_position_alert(self, user_id: str, position: Dict, alert_type: str):
        """Send SL/target hit email via EmailService template.
        ``alert_type`` doubles as the event key (``sl_hit`` / ``target_hit``).
        """
        if not self.email_service.is_available:
            return
        # PR 41 — per-event Alerts Studio gate (alert_type = event key).
        try:
            from .alerts import channels_for_event
            channels = await channels_for_event(user_id, alert_type, supabase_client=self.supabase)
            if "email" not in channels:
                return
        except Exception as exc:
            logger.debug("email position channel-gate check skipped: %s", exc)
        try:
            profile = self.supabase.table("user_profiles").select(
                "email, email_notifications, notifications_enabled"
            ).eq("id", user_id).single().execute()
            pdata = profile.data or {}
            if not pdata.get("email_notifications", True) or not pdata.get("notifications_enabled", True):
                return
            email = pdata.get("email")
            if not email:
                return
            await self.email_service.send_position_alert(email, position, alert_type)
        except Exception as e:
            logger.debug(f"Email position alert failed for {user_id}: {e}")

    async def _send_email_daily_summary(self, user_id: str, summary: Dict):
        """Send daily summary email via EmailService template."""
        if not self.email_service.is_available:
            return
        try:
            profile = self.supabase.table("user_profiles").select(
                "email, email_notifications, notifications_enabled"
            ).eq("id", user_id).single().execute()
            pdata = profile.data or {}
            if not pdata.get("email_notifications", True) or not pdata.get("notifications_enabled", True):
                return
            email = pdata.get("email")
            if not email:
                return
            await self.email_service.send_daily_summary(email, summary)
        except Exception as e:
            logger.debug(f"Email daily summary failed for {user_id}: {e}")

# ============================================================================
# MARKET DATA STREAMER
# ============================================================================


class MarketDataStreamer:
    """
    Streams market data updates
    """

    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.market_data: Dict = {}

    async def update_market_data(self, data: Dict):
        """
        Update and broadcast market data

        data: {
            "nifty": {"price": 21850, "change": 0.45},
            "banknifty": {"price": 46230, "change": 0.32},
            "vix": {"level": 14.5, "change": -2.1},
            "fii_net": 2450,
            "advances": 1250,
            "declines": 750
        }
        """
        self.market_data = data

        message = WSMessage(
            type=MessageType.MARKET_DATA,
            data=data
        )

        await self.manager.broadcast_to_subscribers("market", message)

    async def send_circuit_breaker(self, reason: str, action: str):
        """Send circuit breaker alert"""
        message = WSMessage(
            type=MessageType.CIRCUIT_BREAKER,
            data={
                "reason": reason,
                "action": action,
                "title": "🚨 Circuit Breaker Activated",
                "message": f"{reason}. Action: {action}",
                "priority": "urgent"
            }
        )

        await self.manager.broadcast(message)

# ============================================================================
# WEBSOCKET HANDLER
# ============================================================================


async def websocket_handler(
    websocket: WebSocket,
    token: str,
    manager: ConnectionManager,
    price_service: PriceService,
    supabase_client
):
    """
    Main WebSocket handler
    """
    # Verify token and get user
    try:
        user = supabase_client.auth.get_user(token)
        if not user:
            await websocket.close(code=4001, reason="Invalid token")
            return

        user_id = user.user.id
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Connect
    if not await manager.connect(websocket, user_id):
        return

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "ping":
                    await manager.send_to_user(user_id, WSMessage(
                        type=MessageType.PONG,
                        data={"timestamp": datetime.utcnow().isoformat()}
                    ))

                elif msg_type == "subscribe_symbol":
                    symbol = message.get("symbol")
                    if symbol:
                        manager.subscribe_to_symbol(user_id, symbol)
                        await manager.send_to_user(user_id, WSMessage(
                            type=MessageType.NOTIFICATION,
                            data={"message": f"Subscribed to {symbol}"}
                        ))

                elif msg_type == "unsubscribe_symbol":
                    symbol = message.get("symbol")
                    if symbol:
                        manager.unsubscribe_from_symbol(user_id, symbol)

                elif msg_type == "get_positions":
                    # Fetch and send current positions with live P&L
                    positions = supabase_client.table("positions").select("*").eq(
                        "user_id", user_id
                    ).eq("is_active", True).execute()

                    await price_service.update_position_pnl(user_id, positions.data or [])

            except json.JSONDecodeError:
                await manager.send_to_user(user_id, WSMessage(
                    type=MessageType.ERROR,
                    data={"message": "Invalid JSON"}
                ))

    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for {user_id}: {e}")
        manager.disconnect(user_id)


# ============================================================================
# INITIALIZATION
# ============================================================================

def create_realtime_services(supabase_client, redis_url: str = None):
    """
    Create and return all real-time services
    """
    manager = ConnectionManager(redis_url)
    price_service = PriceService(manager)
    notification_service = NotificationService(manager, supabase_client)
    market_streamer = MarketDataStreamer(manager)

    return {
        "manager": manager,
        "price_service": price_service,
        "notification_service": notification_service,
        "market_streamer": market_streamer
    }
