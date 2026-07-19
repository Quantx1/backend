"""
EventBus — unified fire-and-forget event emitter for PR 13.

Before PR 13 every feature that wanted to reach the frontend either
poked ``ConnectionManager.send_to_user`` directly or went through
``NotificationService.send_xxx`` (which handles DB persistence + push
+ email + Telegram). Both paths worked; neither was a single source
of truth for "a thing happened — tell the user + audit it".

This module is that one entry point. Every caller does::

    from backend.platform.events import emit_event, MessageType

    await emit_event(
        MessageType.REGIME_CHANGE,
        {"old": "bull", "new": "sideways", "confidence": 0.82},
    )

The bus never blocks the caller — delivery is ``asyncio.create_task``'d.
If the realtime stack isn't wired yet (unit tests, background worker
without app context), emit_event degrades to a log line.

Delivery fans out to:
  1. WebSocket broadcast via ``ConnectionManager``
  2. Optional per-user targeting when ``user_id`` is passed
  3. Supabase Realtime broadcast channel (``swingai:events``) —
     lets the Next.js client subscribe via
     ``@supabase/realtime-js`` without holding an open WS to us.
  4. (future) append to an ``events`` audit table for admin replay.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional

from .realtime import ConnectionManager, MessageType, WSMessage

logger = logging.getLogger(__name__)


class EventBus:
    """Singleton wrapper over the existing ConnectionManager.

    Constructor is intentionally tolerant: both ``manager`` and
    ``supabase_client`` can be None, in which case emit is a no-op.
    That lets the bus be imported + tested without the full app
    lifecycle.
    """

    def __init__(
        self,
        manager: Optional[ConnectionManager] = None,
        supabase_client: Any = None,
    ):
        self.manager = manager
        self.supabase = supabase_client

    @property
    def ready(self) -> bool:
        return self.manager is not None

    async def emit(
        self,
        event_type: MessageType,
        data: Dict[str, Any],
        *,
        user_id: Optional[str] = None,
        broadcast: bool = True,
    ) -> None:
        """Send one event. ``user_id`` targets a single user; when
        ``user_id`` is omitted and ``broadcast`` is True, the event
        fans out to every connected WS client.
        """
        if not self.ready:
            logger.debug(
                "EventBus not ready — dropping %s event", event_type.value,
            )
            return

        ws_message = WSMessage(type=event_type, data=data, user_id=user_id)

        try:
            if user_id:
                await self.manager.send_to_user(user_id, ws_message)
            elif broadcast:
                await self.manager.broadcast(ws_message)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("EventBus WS dispatch failed (%s): %s", event_type.value, exc)

        # Supabase Realtime broadcast — optional sideband so frontends
        # subscribed via @supabase/realtime-js see the event even when
        # our own WS isn't connected. Fire-and-forget; never raises.
        asyncio.create_task(self._broadcast_supabase(ws_message))

    async def _broadcast_supabase(self, ws_message: WSMessage) -> None:
        if self.supabase is None:
            return
        try:
            # Supabase Python SDK exposes realtime via ``.realtime``;
            # broadcast channels accept JSON-serializable payloads.
            channel = self.supabase.realtime.channel("swingai:events")
            if hasattr(channel, "send_broadcast"):
                await _maybe_await(channel.send_broadcast(
                    event=ws_message.type.value,
                    payload=json.loads(ws_message.to_json()),
                ))
        except Exception as exc:  # pragma: no cover — Supabase realtime is best-effort
            logger.debug("Supabase realtime broadcast skipped: %s", exc)


async def _maybe_await(value):
    """``asyncio.iscoroutine`` + ``await`` in one."""
    if asyncio.iscoroutine(value):
        await value


# ---------------------------------------------------------------- singleton

_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def set_event_bus(
    manager: Optional[ConnectionManager],
    supabase_client: Any,
) -> None:
    """Wire the bus. Call once from ``app.py`` startup after
    ``create_realtime_services`` returns."""
    global _bus
    with _bus_lock:
        _bus = EventBus(manager=manager, supabase_client=supabase_client)
    logger.info(
        "EventBus wired (manager=%s supabase=%s)",
        type(manager).__name__ if manager else None,
        "yes" if supabase_client else "no",
    )


def get_event_bus() -> EventBus:
    """Return the current bus; returns a no-op bus if not yet wired."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus


async def emit_event(
    event_type: MessageType,
    data: Dict[str, Any],
    *,
    user_id: Optional[str] = None,
    broadcast: bool = True,
) -> None:
    """Module-level shortcut — the everyday call-site used by features.

    Never raises. If the bus isn't wired yet, logs at DEBUG and returns.
    """
    await get_event_bus().emit(
        event_type, data, user_id=user_id, broadcast=broadcast,
    )


__all__ = [
    "EventBus",
    "MessageType",
    "emit_event",
    "get_event_bus",
    "set_event_bus",
]
