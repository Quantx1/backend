"""On a completed 5-min bar, run the intraday scanner over the symbol's rolling
frame and broadcast any setup as an INTRADAY_SIGNAL. Honest-empty (no setup ->
no message) and deduped (same setup_id+direction suppressed until it changes)."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from ...platform.realtime import MessageType, WSMessage
from .scanner import IntradayMatch
from .signal_mapper import match_to_ws_payload

logger = logging.getLogger(__name__)


class IntradayLiveConsumer:
    def __init__(self, manager, scan_fn: Callable[[str, object], List[IntradayMatch]],
                 frame_fn: Callable[[str], Optional[object]]):
        self._manager = manager
        self._scan_fn = scan_fn      # (symbol, frame) -> list[IntradayMatch]
        self._frame_fn = frame_fn    # (symbol) -> frame | None
        self._last_key: dict = {}    # symbol -> "setup_id:direction" last emitted

    async def on_bar_close(self, symbol: str) -> None:
        frame = self._frame_fn(symbol)
        if frame is None:
            return
        try:
            matches = self._scan_fn(symbol, frame)
        except Exception as e:
            logger.debug("intraday scan failed for %s: %s", symbol, e)
            return
        if not matches:
            return
        match = matches[0]  # scanner returns best-first (confidence, then R:R)
        key = f"{match.setup_id}:{match.direction}"
        if self._last_key.get(symbol) == key:
            return
        self._last_key[symbol] = key
        await self._manager.broadcast_symbol_update(
            symbol, WSMessage(type=MessageType.INTRADAY_SIGNAL, data=match_to_ws_payload(match))
        )
