"""Bridges DepthBus depth dicts to /ws symbol-watchers as DEPTH_UPDATE."""
from __future__ import annotations

from typing import Awaitable, Callable, Dict

from .realtime import ConnectionManager, MessageType, WSMessage


def make_depth_handler(manager: ConnectionManager) -> Callable[[str, Dict], Awaitable[None]]:
    async def handler(symbol: str, depth: Dict) -> None:
        await manager.broadcast_symbol_update(
            symbol, WSMessage(type=MessageType.DEPTH_UPDATE, data=depth)
        )
    return handler
