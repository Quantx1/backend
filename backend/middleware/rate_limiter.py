"""
Rate-limit middleware.

Per-IP + per-path sliding-window counter, 1-minute window. Sensitive
endpoints (auth, broker-connect, auto-trader kill, payments, Telegram
link-start) get their own stricter per-path limits on a composite
``ip:path`` key so bots can't exhaust them through a normal user's
global budget.

PR 56 additions:
    * Expanded sensitive-path list (auto-trader, broker, payments,
      Telegram onboarding).
    * Background cleanup of idle IP buckets — prevents the in-memory
      store from growing unbounded across a long-lived process. Runs
      lazily on dispatch, at most once per minute, so there's no
      separate task to manage.
    * ``fetch``-style identifier support: if the request carries a
      ``Cf-Connecting-Ip`` header (Cloudflare) or
      ``X-Forwarded-For`` (Vercel / other reverse proxies), we use
      the first hop instead of the direct socket peer — otherwise
      every request looks like it came from the reverse proxy.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings

logger = logging.getLogger(__name__)

# In-memory storage — good enough for single-worker FastAPI. Swap for
# Redis when the process count > 1 (rate limits diverge across workers).
rate_limit_storage: Dict[str, List[datetime]] = defaultdict(list)

# Stricter per-path limits. Composite key (ip:path) so these don't
# share the global bucket.
AUTH_PATH_LIMITS: Dict[str, int] = {
    # Auth endpoints (brute-force protection) — already enforced pre-PR 56.
    "/api/auth/login": 5,
    "/api/auth/signup": 3,
    "/api/auth/forgot-password": 3,
    "/api/auth/resend-verification": 3,
    # PR 56 — sensitive mutations. Kept generous enough for legitimate
    # retries but low enough to make brute force painful.
    "/api/broker/connect": 6,
    "/api/broker/disconnect": 6,
    "/api/payments/verify": 10,
    "/api/auto-trader/enable": 4,
    "/api/auto-trader/kill": 10,
    "/api/telegram/link/start": 10,
    # Copilot chat — each call hits the OpenRouter gateway. Keep it from being hot-looped.
    "/api/ai/copilot/chat": 40,
    "/api/ai/copilot/chat/stream": 40,
}

# Paths that bypass the limiter entirely. Kept to things where
# correctness-by-signature is already enforced and retries are part of
# the protocol.
_BYPASS_PATHS = {
    "/health",
    "/api/health",
    "/api/payments/webhook",        # Razorpay signature-verified
    "/api/telegram/webhook",        # path includes secret; middleware sees prefix
}

# How often we scrub the in-memory store. Keys with no hits in the last
# 5 minutes are dropped.
_CLEANUP_INTERVAL_SECONDS = 60
_CLEANUP_IDLE_SECONDS = 300


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Respects Cf-Connecting-IP then X-Forwarded-For's
    first hop; falls back to the socket peer."""
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",", 1)[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def _bypass(path: str) -> bool:
    if path in _BYPASS_PATHS:
        return True
    # Telegram webhook path includes a secret segment — match the prefix.
    if path.startswith("/api/telegram/webhook/"):
        return True
    return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        requests_per_minute: Optional[int] = None,
        path_overrides: Optional[Dict[str, int]] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self.path_overrides = path_overrides or AUTH_PATH_LIMITS
        self._last_cleanup_ts = 0.0

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _bypass(path):
            return await call_next(request)

        self._maybe_cleanup()

        client_ip = _client_ip(request)
        now = datetime.now()

        limit = self.path_overrides.get(path, self.requests_per_minute)
        storage_key = f"{client_ip}:{path}" if path in self.path_overrides else client_ip

        # Prune old entries for this key, then evaluate.
        window_start = now - timedelta(minutes=1)
        bucket = rate_limit_storage[storage_key]
        # Find cutoff index — entries are appended in order so the list
        # stays monotonic. Linear scan is fine for limits in the
        # hundreds.
        cutoff = 0
        for i, ts in enumerate(bucket):
            if ts >= window_start:
                cutoff = i
                break
        else:
            cutoff = len(bucket)
        if cutoff > 0:
            del bucket[:cutoff]

        if len(bucket) >= limit:
            logger.warning(
                "Rate limit exceeded ip=%s path=%s limit=%d/min",
                client_ip, path, limit,
            )
            # PR 103 — fan out to PostHog so the ops dashboard can
            # surface throttling cohorts (which paths are hammered,
            # by how many distinct clients). We hash the IP rather
            # than emit it raw so PostHog never holds the original
            # — first 12 chars of SHA-256 are stable for grouping
            # but non-reversible. Best-effort; never blocks the 429.
            try:
                from ..observability import EventName, track
                ip_hash = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()[:12]
                track(EventName.RATE_LIMITED, None, {
                    "path": path,
                    "limit": limit,
                    "window_seconds": 60,
                    "ip_hash": ip_hash,
                    "path_specific": path in self.path_overrides,
                })
            except Exception:
                pass
            # NOTE: raising HTTPException here previously surfaced as a
            # 500 to the client. BaseHTTPMiddleware.dispatch runs inside
            # an anyio TaskGroup that wraps the HTTPException into a
            # nested ExceptionGroup, which Starlette's error middleware
            # turns into an "Internal Server Error". Discovered by the
            # API contract audit 2026-05-18. Returning a JSONResponse
            # directly bypasses the wrap and delivers a proper 429 with
            # the same body shape clients already expected.
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "limit": limit,
                    "window": "1 minute",
                    "retry_after": 60,
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int((now + timedelta(minutes=1)).timestamp())),
                },
            )

        bucket.append(now)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - len(bucket)))
        response.headers["X-RateLimit-Reset"] = str(int((now + timedelta(minutes=1)).timestamp()))
        return response

    def _maybe_cleanup(self) -> None:
        """Drop buckets whose most-recent request is older than the idle
        threshold. Keeps memory bounded in long-lived workers."""
        now_ts = time.monotonic()
        if now_ts - self._last_cleanup_ts < _CLEANUP_INTERVAL_SECONDS:
            return
        self._last_cleanup_ts = now_ts
        cutoff = datetime.now() - timedelta(seconds=_CLEANUP_IDLE_SECONDS)
        # Snapshot keys to avoid mutating during iteration.
        for key in list(rate_limit_storage.keys()):
            bucket = rate_limit_storage.get(key)
            if not bucket or bucket[-1] < cutoff:
                rate_limit_storage.pop(key, None)
