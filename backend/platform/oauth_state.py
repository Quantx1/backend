"""Redis-backed OAuth state for broker connect flows.

Replaces the in-process dict (lost on restart / wrong across instances).
State is single-use with a 10-minute TTL. Payload carries the user, broker,
and where to return the user after a successful callback.
"""
from __future__ import annotations
import json
import secrets
from typing import Optional

import redis.asyncio as aioredis

from backend.core.config import settings

_TTL_SECONDS = 600
_PREFIX = "oauth_state:"
_client: Optional["aioredis.Redis"] = None


def _get_redis() -> "aioredis.Redis":
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _client


async def store_state(user_id: str, broker: str, return_to: str = "settings") -> str:
    state = secrets.token_urlsafe(32)
    payload = json.dumps({"user_id": user_id, "broker": broker, "return_to": return_to})
    await _get_redis().setex(_PREFIX + state, _TTL_SECONDS, payload)
    return state


async def consume_state(state: str) -> Optional[dict]:
    if not state:
        return None
    r = _get_redis()
    key = _PREFIX + state
    raw = await r.get(key)
    if raw is None:
        return None
    await r.delete(key)  # single-use
    return json.loads(raw)
