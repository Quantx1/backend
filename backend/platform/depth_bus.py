"""Redis-Streams transport for per-symbol L2 depth.

Producers (broker tickers) XADD canonical depth dicts; the WS consumer XREADs
and fans them to symbol-watchers. Decouples ingestion from fan-out and is the
seam the later intraday-signal-compute consumer attaches to. Best-effort:
publish/consume failures are logged, never raised (honest-empty upstream)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Dict

logger = logging.getLogger(__name__)

DEPTH_STREAM = "quantx:depth"
_MAXLEN = 10000  # cap stream growth


class DepthBus:
    def __init__(self, redis):
        self._redis = redis  # redis client created with decode_responses=True

    async def publish(self, symbol: str, depth: Dict) -> None:
        try:
            await self._redis.xadd(
                DEPTH_STREAM,
                {"symbol": symbol, "depth": json.dumps(depth)},
                maxlen=_MAXLEN, approximate=True,
            )
        except Exception as e:  # transport hiccup must not kill the tick path
            logger.debug("depth publish failed for %s: %s", symbol, e)

    async def consume_once(
        self, handler: Callable[[str, Dict], Awaitable[None]], last_id: str = "$", block: int = 5000
    ) -> str:
        """Read one batch and dispatch. Returns the new last_id."""
        resp = await self._redis.xread({DEPTH_STREAM: last_id}, block=block, count=100)
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                last_id = entry_id
                try:
                    symbol = fields["symbol"]
                    depth = json.loads(fields["depth"])
                    await handler(symbol, depth)
                except Exception as e:
                    logger.debug("depth handler error: %s", e)
        return last_id

    async def consume_forever(
        self, handler: Callable[[str, Dict], Awaitable[None]], start_id: str = "$"
    ) -> None:
        # NOTE: on an exception mid-batch the partially-advanced cursor is dropped and
        # the loop restarts from the last *successful* id, so in-flight entries may be
        # redelivered (at-least-once). Downstream handlers must stay idempotent.
        last_id = start_id
        while True:
            try:
                last_id = await self.consume_once(handler, last_id=last_id)
            except Exception as e:
                logger.warning("depth consume loop error: %s", e)
                await asyncio.sleep(1)

    async def aclose(self) -> None:
        """Close the underlying Redis client (best-effort)."""
        try:
            await self._redis.aclose()
        except Exception as e:
            logger.debug("depth bus close failed: %s", e)
