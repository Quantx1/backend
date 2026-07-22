"""
================================================================================
QUANT X PRODUCTION BACKEND
================================================================================
FastAPI + Supabase + Razorpay + Real-time WebSocket
Complete API for AI-powered swing trading platform
================================================================================
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import logging

# Load .env into the process env so raw os.getenv() consumers (e.g. DATABASE_URL
# for the direct-Postgres path used by breadth / sector-rotation / live-alerts /
# the EOD upsert) work locally. No-op in prod where env vars are set directly.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from supabase import create_client, Client
import razorpay

from ..core.config import settings, validate_startup
from ..middleware import RateLimitMiddleware, LoggingMiddleware, SecurityHeadersMiddleware
from ..platform.realtime import create_realtime_services
from ..data.brokers.ticker_mapping import BrokerTickerManager
from ..platform.scheduler import SchedulerService
from ..ai.signals import SignalGenerator
from ..trading.execution import TradeExecutionService

# ============================================================================
# LOGGING
# ============================================================================

# Optional: Sentry error tracking
if settings.SENTRY_DSN:
    try:
        import os as _os
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        # PR 101 — release tagging. Sentry uses `release` to group
        # errors by deploy so a regression in build N+1 doesn't look
        # the same as a fixed issue in build N. Prefer the platform's
        # git SHA when available (Railway / Vercel / Render set these);
        # fall back to APP_VERSION so we always tag something.
        _git_sha = (
            _os.getenv("RAILWAY_GIT_COMMIT_SHA")
            or _os.getenv("VERCEL_GIT_COMMIT_SHA")
            or _os.getenv("RENDER_GIT_COMMIT")
            or _os.getenv("GIT_SHA")
            or ""
        )
        _release = f"swingai-backend@{_git_sha[:7]}" if _git_sha else f"swingai-backend@{settings.APP_VERSION}"

        # PR 101 — privacy filter. Strip Authorization / Cookie headers
        # and any payload key whose name suggests a credential before
        # the event reaches Sentry. Defense in depth — Sentry treats
        # default PII as off below, but request bodies can still carry
        # broker_token / api_key / password / etc. without those flags.
        _CREDENTIAL_KEYS = (
            "authorization", "cookie", "set-cookie",
            "password", "secret", "api_key", "apikey",
            "token", "access_token", "refresh_token",
            "totp_secret", "broker_token",
        )

        def _scrub(obj):
            if isinstance(obj, dict):
                return {
                    k: ("[redacted]" if any(c in k.lower() for c in _CREDENTIAL_KEYS) else _scrub(v))
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [_scrub(x) for x in obj]
            return obj

        def _before_send(event, _hint):
            try:
                req = event.get("request") if isinstance(event, dict) else None
                if isinstance(req, dict):
                    if isinstance(req.get("headers"), dict):
                        req["headers"] = _scrub(req["headers"])
                    if isinstance(req.get("cookies"), dict):
                        req["cookies"] = {}
                    if isinstance(req.get("data"), (dict, list)):
                        req["data"] = _scrub(req["data"])
                if isinstance(event, dict) and isinstance(event.get("extra"), dict):
                    event["extra"] = _scrub(event["extra"])
            except Exception:
                pass
            return event

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.APP_ENV,
            release=_release,
            traces_sample_rate=0.1,
            send_default_pii=False,
            before_send=_before_send,
            integrations=[FastApiIntegration(), StarletteIntegration()],
        )
    except ImportError:
        pass  # sentry-sdk not installed

from ..middleware import configure_structured_logging
configure_structured_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ============================================================================
# SUPABASE QUERY HELPER WITH RETRY
# ============================================================================


async def supabase_query_with_retry(fn, retries=2, timeout_fallback=None):
    """Execute a Supabase query with retry on timeout. Returns fallback on failure.

    ``fn`` is a synchronous closure (the supabase-py client is sync/blocking), so
    we run it via ``asyncio.to_thread`` — otherwise a slow query would block the
    single event-loop thread and wedge every other in-flight request.
    """
    for attempt in range(retries + 1):
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            err_str = str(e)
            if "timed out" in err_str or "ConnectTimeout" in err_str or "Connection reset" in err_str:
                if attempt < retries:
                    await asyncio.sleep(0.5)
                    continue
            if timeout_fallback is not None:
                logging.getLogger(__name__).warning(
                    f"Supabase query failed after {attempt + 1} attempts: {err_str[:80]}")
                return timeout_fallback
            raise

# ============================================================================
# CLIENTS
# ============================================================================

_supabase_anon: Optional[Client] = None
_supabase_admin: Optional[Client] = None
_razorpay_client: Optional[razorpay.Client] = None


def get_supabase() -> Client:
    """Get Supabase client (anon key) — singleton"""
    global _supabase_anon
    if _supabase_anon is None:
        _supabase_anon = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    return _supabase_anon


def get_supabase_admin() -> Client:
    """Get Supabase admin client (service role key) — singleton"""
    global _supabase_admin
    if _supabase_admin is None:
        _supabase_admin = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    return _supabase_admin


def get_razorpay() -> razorpay.Client:
    """Get Razorpay client — singleton"""
    global _razorpay_client
    if _razorpay_client is None:
        _razorpay_client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))
    return _razorpay_client

# ============================================================================
# AUTH DEPENDENCY
# ============================================================================


security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Get current authenticated user from JWT token.

    Decodes the Supabase JWT locally with signature verification (HS256 via
    SUPABASE_JWT_SECRET). Falls back to Supabase auth.get_user() network call
    only if local decode fails with a non-signature error (e.g. unusual alg).

    Security: if SUPABASE_JWT_SECRET is set (required in production — see
    core.config.validate_startup), ALL tokens must pass signature verification.
    Forged tokens → 401. If the secret is unset (dev only), signature check is
    skipped with a WARNING log.
    """
    import jwt as pyjwt
    from types import SimpleNamespace

    # DEV-ONLY auth bypass — shares the single hard-gated check in core.security
    # (never honored when APP_ENV=production). Lets local testing exercise gated
    # endpoints (assistant, trades/*) without a Supabase login.
    from ..core.security import _dev_auth_enabled, _DEV_USER
    if _dev_auth_enabled():
        return _DEV_USER
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials
    jwt_secret = settings.SUPABASE_JWT_SECRET

    # --- Fast path: decode JWT locally (no network call) ---
    try:
        if jwt_secret:
            # Production path: verify signature + expiry via pyjwt.
            # Supabase uses HS256 for JWTs signed with the project JWT secret.
            # audience claim is "authenticated" by default on Supabase JWTs.
            payload = pyjwt.decode(
                token,
                key=jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_signature": True, "verify_exp": True, "verify_aud": True},
            )
        else:
            # Dev-only fallback: no secret configured. Decode without verification
            # but still enforce role + expiry at the claim level.
            logger.warning(
                "JWT signature verification is DISABLED — SUPABASE_JWT_SECRET is unset. "
                "DO NOT run this in production."
            )
            payload = pyjwt.decode(
                token,
                options={"verify_signature": False},
                algorithms=["HS256", "ES256"],
            )

        user_id = payload.get("sub")
        email = payload.get("email")
        role = payload.get("role")

        if not user_id or role != "authenticated":
            raise HTTPException(status_code=401, detail="Invalid token")

        # Double-check expiry when we skipped verify_exp (unset-secret branch only).
        if not jwt_secret:
            import time as _time
            exp = payload.get("exp", 0)
            if exp and exp < _time.time():
                raise HTTPException(status_code=401, detail="Token expired")

        # Return a user-like object compatible with the rest of the app.
        return SimpleNamespace(
            id=user_id, email=email, role=role,
            user_metadata=payload.get("user_metadata", {}),
            app_metadata=payload.get("app_metadata", {}),
        )
    except HTTPException:
        raise
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidSignatureError:
        # Signature mismatch = forged or wrong-project token. NEVER fall back to
        # network verify for this — fail loudly.
        logger.warning("JWT signature verification FAILED — rejecting token")
        raise HTTPException(status_code=401, detail="Invalid token signature")
    except pyjwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except pyjwt.InvalidTokenError as decode_err:
        # Other structural decode failures — try network fallback once.
        logger.warning(f"Local JWT decode failed ({decode_err}), falling back to Supabase API")

    # --- Slow path: verify via Supabase API (only for non-signature decode failures) ---
    try:
        supabase = get_supabase()
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user.user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


async def get_user_profile(user=Depends(get_current_user)):
    """Get user profile with subscription details.

    Phase 1.7 audit fix #1.8 — the legacy fallback synthesized a
    ``subscription_status="active"`` Pro profile on any DB exception.
    That turned every transient Supabase hiccup (network blip, RLS
    misconfig, pool exhaustion) into a free Pro-tier upgrade for any
    authenticated user, which is also exploitable: an attacker who can
    force a DB error on the profile lookup gains Pro entitlements
    until the issue resolves. Fail closed instead.
    """
    # DEV-ONLY: synthesize an Elite profile for the auth-bypass mock user (no DB
    # row exists for it). Hard gated (never production), scoped to the mock id —
    # mirrors the tier grant in core.tiers so profile-dependent agents work.
    from ..core.security import _dev_auth_enabled, _DEV_USER
    if _dev_auth_enabled() and getattr(user, "id", None) == _DEV_USER.id:
        return {
            "id": _DEV_USER.id, "email": _DEV_USER.email, "full_name": "Dev User",
            "tier": "elite", "subscription_status": "active", "is_admin": True,
        }
    try:
        supabase = get_supabase_admin()
        result = supabase.table("user_profiles").select("*, subscription_plans(*)").eq("id", user.id).single().execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        return result.data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Profile fetch failed for user_id=%s: %s — refusing to synthesize a Pro fallback",
            getattr(user, "id", "<unknown>"), e,
        )
        raise HTTPException(
            status_code=503,
            detail="Profile temporarily unavailable. Please retry.",
        )

# ============================================================================
# RUNTIME SERVICES (realtime + scheduler)
# ============================================================================

realtime_services: Dict[str, Any] = {}
manager: Optional[Any] = None
scheduler_service: Optional[SchedulerService] = None

# ============================================================================
# APP INITIALIZATION
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global realtime_services, manager, scheduler_service
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.APP_ENV}")

    # Validate required environment variables
    validate_startup()

    # Verify Supabase database connectivity and schema
    if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
        try:
            supabase = get_supabase_admin()
            critical_tables = ["subscription_plans", "signals", "trades", "positions", "user_profiles"]
            for table in critical_tables:
                supabase.table(table).select("id").limit(1).execute()
            logger.info(f"✅ Supabase database connected — {len(critical_tables)} critical tables verified")
        except Exception as e:
            logger.error(f"⚠️ Database check failed: {e}")
            logger.error("   Run the SQL migrations in infrastructure/database/complete_schema.sql")
            if settings.APP_ENV == "production":
                raise RuntimeError("Database schema not initialized — run migrations first")
    else:
        logger.warning("⚠️ Supabase not configured — database features unavailable")

    # Initialize market data provider
    if settings.DATA_PROVIDER == "kite":
        try:
            from ..data.providers.kite import get_kite_admin_client
            kite_admin = get_kite_admin_client()
            kite_admin.initialize()
            logger.info("✅ Kite admin client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Kite admin init failed: {e} — will use jugaad-data fallback")
    else:
        logger.info("✅ Free data mode (yfinance) — no broker credentials needed")

    # Initialize realtime services (WebSocket manager + notifications)
    try:
        realtime_services = create_realtime_services(get_supabase_admin(), settings.REDIS_URL)
        manager = realtime_services.get("manager")
        app.state.realtime = realtime_services

        if manager and settings.ENABLE_REDIS:
            await manager.init_redis()
            logger.info("✅ Realtime services initialized with Redis")
        else:
            logger.info("✅ Realtime services initialized (in-memory)")

        # PR 13: wire the unified EventBus so feature code can
        # ``await emit_event(...)`` without reaching into app.state.
        try:
            from ..platform.events import set_event_bus
            set_event_bus(manager, get_supabase_admin())
            logger.info("✅ EventBus wired")
        except Exception as bus_err:
            logger.warning(f"EventBus wiring failed: {bus_err}")
    except Exception as e:
        logger.error(f"Realtime initialization failed: {e}")

    # Start price polling for WebSocket (uses Kite quotes)
    price_service = realtime_services.get("price_service")
    if price_service:
        asyncio.create_task(price_service.start_polling(interval=30))
        logger.info("✅ Price polling started (30s interval)")

    # Initialize broker ticker manager (real-time streaming via broker WebSockets)
    try:
        price_service = realtime_services.get("price_service")
        if price_service and settings.ENABLE_BROKER_TICKER:
            # F6: depth bus (Redis Streams) — only when Redis is enabled. Without it
            # the ticker still runs (depth_bus=None) and depth is simply not streamed,
            # rather than spawning a consume loop that error-spins on an absent Redis.
            depth_bus = None
            if settings.ENABLE_REDIS:
                import redis.asyncio as _aioredis
                from ..platform.depth_bus import DepthBus
                from ..platform.depth_to_ws import make_depth_handler

                _depth_redis = _aioredis.from_url(settings.REDIS_URL, decode_responses=True)
                depth_bus = DepthBus(_depth_redis)

            # Intraday: 5-min bar aggregator + live scanner consumer (per-user feed)
            from ..data.brokers.intraday_bars import IntradayBarAggregator
            from ..services.intraday_scanner.live_consumer import IntradayLiveConsumer
            from ..services.intraday_scanner.scanner import scan_intraday_setups
            import datetime as _dt

            _bar_agg = IntradayBarAggregator(interval_min=5)
            _pending_bars: list = []

            def _bar_sink(symbol, price, volume):
                ist = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
                closed = _bar_agg.feed(symbol, price, volume, _dt.datetime.now(ist))
                if closed is not None:
                    _pending_bars.append(symbol)

            def _scan(sym, frame):
                return scan_intraday_setups([sym], bars_fetcher=lambda s: frame)

            intraday_consumer = IntradayLiveConsumer(manager, scan_fn=_scan, frame_fn=_bar_agg.frame)
            app.state.intraday_consumer = intraday_consumer
            app.state.intraday_bar_agg = _bar_agg

            async def _drain_bars():
                import asyncio as _a
                while True:
                    while _pending_bars:
                        sym = _pending_bars.pop(0)
                        try:
                            await intraday_consumer.on_bar_close(sym)
                        except Exception as _e:
                            logger.debug("intraday consumer error: %s", _e)
                    await _a.sleep(1)

            app.state.intraday_drain_task = asyncio.create_task(_drain_bars())
            broker_ticker_mgr = BrokerTickerManager(price_service, depth_bus=depth_bus, bar_sink=_bar_sink)
            app.state.broker_ticker_manager = broker_ticker_mgr
            app.state.depth_bus = depth_bus
            app.state.depth_consumer_task = None
            if depth_bus is not None:
                # consumer: depth stream -> /ws symbol watchers
                app.state.depth_consumer_task = asyncio.create_task(
                    depth_bus.consume_forever(make_depth_handler(manager))
                )
            logger.info("✅ Broker ticker manager initialized")
        else:
            app.state.broker_ticker_manager = None
            app.state.depth_bus = None
            app.state.depth_consumer_task = None
            app.state.intraday_drain_task = None
    except Exception as e:
        logger.warning(f"Broker ticker manager init failed: {e}")
        app.state.broker_ticker_manager = None
        app.state.depth_bus = None
        app.state.depth_consumer_task = None
        app.state.intraday_drain_task = None

    # Initialize signal generator + model download (startup health).
    # Per the "no fallbacks" rule (memory: feedback_no_fallbacks_no_refunds_2026_04_19),
    # SignalGenerator REQUIRES all model voters (TFT, Qlib, FinBERT,
    # LGBM, regime_hmm). If any is missing the constructor raises and we
    # surface the failure. The HTTP server keeps serving — auth, paper
    # trading, agents, public pages all still work; only the live
    # signal pipeline + scheduler are inactive until the operator
    # ingests the missing artifacts.
    app.state.signal_generator = None
    try:
        signal_generator = SignalGenerator(get_supabase_admin())
        app.state.signal_generator = signal_generator
        app.state.model_status = {
            "lgbm_gate": signal_generator._lgbm_gate is not None,
            "regime_detector": signal_generator._regime_detector is not None,
            "tft_predictor": getattr(signal_generator, '_tft_predictor', None) is not None,
            "ensemble": getattr(signal_generator, '_ensemble', None) is not None,
        }
        loaded = [k for k, v in app.state.model_status.items() if v]
        logger.info(f"Signal generator initialized — models loaded: {loaded}")
    except Exception as e:
        app.state.model_status = {
            "lgbm_gate": False, "regime_detector": False,
            "tft_predictor": False, "ensemble": False,
        }
        logger.error(f"Signal generator initialization failed: {e}")

    # Initialize scheduler (optional). If signal_generator failed to
    # boot (e.g. Qlib data not yet ingested) we skip the scheduler too
    # — it depends on signal_generator. App-level routes that don't
    # need the scheduler (public trust pages, agent endpoints, paper
    # trading display, frontend rendering) continue serving normally.
    if settings.ENABLE_SCHEDULER:
        signal_generator = getattr(app.state, 'signal_generator', None)
        if signal_generator is None:
            logger.warning(
                "Scheduler skipped — signal_generator did not initialize. "
                "Bootstrap Qlib provider + retrain models, then restart."
            )
        else:
            try:
                trade_executor = TradeExecutionService(get_supabase_admin())
                notification_service = realtime_services.get("notification_service")
                scheduler_service = SchedulerService(
                    get_supabase_admin(),
                    signal_generator,
                    trade_executor,
                    notification_service,
                )
                scheduler_service.start()
                try:
                    scheduler_service.set_ws_manager(manager)
                except Exception:
                    pass
                app.state.scheduler = scheduler_service
                logger.info("✅ Scheduler started")
            except Exception as e:
                logger.error(f"Scheduler initialization failed: {e}")

    # Pre-warm slow public market caches so the first dashboard render
    # never pays the cold cost. Regime alone takes 40+s cold; running it
    # in the background here means the first user gets a 1.6ms response.
    async def _prewarm_market_caches():
        try:
            from . import market_routes as _mkt
            # Warm every public market cache the /markets page reads on first
            # paint — regime, indices, global cues, FII/DII and the AI Daily
            # Briefing (headline + narrative) — so the first visitor after a
            # (re)start gets an instant, fully-populated render instead of a
            # cold yfinance/LLM block or an honest-empty flash.
            # Heal the regime timeline first (idempotent ensemble refresh over
            # the candle store) so the gauge never serves a stale row after a
            # reaped dev server / missed 8:15 job. A few seconds, off-thread.
            async def _heal_regime():
                from ..services.regime.refresh import refresh_regime_history
                return await asyncio.to_thread(refresh_regime_history, 30)

            # get_market_indices is NOT warmed: it takes a Request (entitlement
            # gate) and calling it bare raised a synchronous TypeError that
            # aborted this whole gather — leaving every cache cold on boot.
            results = await asyncio.gather(
                _heal_regime(),
                _mkt.get_market_regime_public(),
                _mkt.get_global_cues(),
                _mkt.get_fii_dii_eod(),
                _mkt.get_market_briefing("auto"),
                _mkt.get_market_news(),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Market pre-warm failed: {r}")
            logger.info("✅ Market caches pre-warmed (regime + indices + global + fii/dii + briefing)")
        except Exception as e:
            logger.warning(f"Market pre-warm setup failed: {e}")

    asyncio.create_task(_prewarm_market_caches())

    # Warm the LiveScreener computed table (500-symbol indicator grid) in the
    # background so the sector-heatmap / power-screeners serve instantly instead
    # of 503-ing on the first cold request. Fire-and-forget off-thread — the
    # ~25-90s compute must never block boot or other requests.
    async def _prewarm_screener_cache():
        try:
            from ..data.screener.engine import get_live_screener
            await asyncio.to_thread(get_live_screener()._get_computed_data)
            logger.info("✅ Screener/sector cache pre-warmed")
        except Exception as e:
            logger.warning(f"Screener pre-warm failed: {e}")

    asyncio.create_task(_prewarm_screener_cache())

    yield

    if scheduler_service:
        scheduler_service.stop()

    _intraday_task = getattr(app.state, "intraday_drain_task", None)
    if _intraday_task is not None:
        _intraday_task.cancel()

    # F6: stop the depth consumer + close its Redis client
    _depth_task = getattr(app.state, "depth_consumer_task", None)
    if _depth_task is not None:
        _depth_task.cancel()
    _depth_bus = getattr(app.state, "depth_bus", None)
    if _depth_bus is not None:
        await _depth_bus.aclose()

    # Kite admin client is stateless — no shutdown needed

    logger.info("🛑 Shutting down Quant X")

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Powered Swing Trading Platform for Indian Markets",
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS - Allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=600,  # Cache preflight responses for 10 minutes
)

# Security Headers
app.add_middleware(SecurityHeadersMiddleware)

# Logging
app.add_middleware(LoggingMiddleware)

# Rate Limiting
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)

# GZip Compression (compress responses > 500 bytes)
app.add_middleware(GZipMiddleware, minimum_size=500)


# ============================================================================
# GLOBAL EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return consistent 422 error format for validation failures"""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        errors.append({"field": field, "message": error.get("msg", "Validation error")})
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "details": errors},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardize HTTP error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail or "Request failed"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — never leak internal details"""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs",
        "health": "/api/health"
    }


@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health():
    """Liveness probe — returns 200 if the process is alive."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
    }


@app.get("/ready", tags=["Health"])
@app.get("/api/ready", tags=["Health"])
async def readiness():
    """Readiness probe — returns 200 only when all critical dependencies are up.

    Use this for Kubernetes / Railway / load-balancer health checks so traffic
    is only routed to instances that can actually serve requests.
    """
    checks: Dict[str, str] = {}

    # 1. Database
    try:
        supabase = get_supabase_admin()
        supabase.table("subscription_plans").select("id").limit(1).execute()
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    # 2. ML models (optional — degraded but not down)
    model_status = getattr(app.state, "model_status", {})
    loaded_count = sum(1 for v in model_status.values() if v)
    checks["ml_models"] = f"{loaded_count}/{len(model_status)} loaded"
    checks["ml_labeler"] = "ok" if model_status.get("ml_labeler") else "unavailable"

    # 3. Redis (only if enabled)
    if settings.ENABLE_REDIS:
        try:
            rt = getattr(app.state, "realtime", {})
            mgr = rt.get("manager")
            if mgr and mgr.redis:
                await mgr.redis.ping()
                checks["redis"] = "ok"
            else:
                checks["redis"] = "not_connected"
        except Exception:
            checks["redis"] = "error"

    # 4. Scheduler
    sched = getattr(app.state, "scheduler", None)
    checks["scheduler"] = "running" if sched and getattr(sched, "running", False) else "stopped"

    # Overall: ready if database is OK (everything else is degraded-OK)
    is_ready = checks["database"] == "ok"
    status_code = 200 if is_ready else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "ready": is_ready,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION,
        },
    )


# ============================================================================
# BROKER ROUTES — handled by broker_routes.py (included via include_router)
# ============================================================================

# ============================================================================
# WEBSOCKET ENDPOINTS (token via header — preferred — and legacy URL path)
# ============================================================================

def _verify_ws_token(token: str) -> Optional[str]:
    """Verify a JWT and return the user_id (sub claim) or None on failure.

    Mirrors get_current_user's local verification path so WebSocket handshakes
    don't need to round-trip Supabase. Respects SUPABASE_JWT_SECRET — when set,
    signatures are verified. When unset (dev only), signature verification is
    skipped with the same warning as REST.
    """
    import jwt as pyjwt
    jwt_secret = settings.SUPABASE_JWT_SECRET
    try:
        if jwt_secret:
            payload = pyjwt.decode(
                token,
                key=jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_signature": True, "verify_exp": True, "verify_aud": True},
            )
        else:
            payload = pyjwt.decode(
                token,
                options={"verify_signature": False},
                algorithms=["HS256", "ES256"],
            )
            # Enforce role + expiry when signature skipped
            import time as _time
            if payload.get("exp", 0) and payload["exp"] < _time.time():
                return None
        if payload.get("role") != "authenticated":
            return None
        return payload.get("sub")
    except pyjwt.InvalidTokenError as e:
        logger.warning(f"WebSocket JWT verification failed: {e}")
        return None


async def _handle_ws_session(websocket: WebSocket, user_id: str):
    """Shared WebSocket session loop used by both /ws (header auth) and
    /ws/{token} (legacy URL auth). Assumes websocket is already authenticated
    but NOT yet accepted."""
    if not manager:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Realtime services not available"})
        await websocket.close(code=4002)
        return

    await manager.connect(websocket, user_id)

    broker_ticker_mgr = getattr(app.state, 'broker_ticker_manager', None)
    if broker_ticker_mgr:
        try:
            supabase_admin = get_supabase_admin()
            conn_resp = supabase_admin.table("broker_connections").select("broker_name,access_token").eq(
                "user_id", user_id).eq("status", "connected").maybe_single().execute()
            if conn_resp and conn_resp.data:
                from ..data.brokers.credentials import decrypt_credentials
                broker_name = conn_resp.data["broker_name"]
                creds = decrypt_credentials(conn_resp.data["access_token"])
                await broker_ticker_mgr.connect_user_ticker(user_id, broker_name, creds)
        except Exception as e:
            logger.debug(f"Broker ticker auto-connect skipped for {user_id}: {e}")

    try:
        while True:
            data = await websocket.receive_text()

            if data == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                continue

            try:
                message = json.loads(data)
                action = message.get("action", "")
                channel = message.get("channel", "")

                if action == "subscribe":
                    await handle_subscribe(user_id, message)
                    await websocket.send_json({
                        "type": "subscribed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif action == "unsubscribe":
                    await handle_unsubscribe(user_id, message)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif action == "get_prices":
                    symbols = message.get("symbols", [])
                    await send_price_update(user_id, symbols)
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown action: {action}"
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "echo",
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                })
    except WebSocketDisconnect:
        if user_id:
            manager.disconnect(user_id)
            if broker_ticker_mgr:
                await broker_ticker_mgr.disconnect_user_ticker(user_id)


@app.websocket("/ws")
async def websocket_endpoint_header(websocket: WebSocket):
    """WebSocket endpoint with bearer-token auth via the Sec-WebSocket-Protocol
    header — the preferred path (tokens don't leak to server logs / CDN).

    Clients connect with two subprotocols:
        new WebSocket(url, ['access_token', userJwt])
    The server echoes back 'access_token' to complete the handshake and uses
    the second protocol string as the JWT.
    """
    user_id: Optional[str] = None
    try:
        subprotocols = websocket.headers.get("sec-websocket-protocol", "")
        parts = [p.strip() for p in subprotocols.split(",") if p.strip()]
        if len(parts) < 2 or parts[0] != "access_token":
            await websocket.close(code=4003)
            return
        token = parts[1]
        user_id = _verify_ws_token(token)
        if not user_id:
            await websocket.close(code=4001)
            return

        # Accept the handshake echoing the 'access_token' subprotocol.
        await websocket.accept(subprotocol="access_token")

        # Hand off to the shared session loop. Reuse its setup code but skip
        # the accept (we already accepted with the subprotocol echo).
        if not manager:
            await websocket.send_json({"type": "error", "message": "Realtime services not available"})
            await websocket.close(code=4002)
            return
        await manager.connect(websocket, user_id)
        broker_ticker_mgr = getattr(app.state, 'broker_ticker_manager', None)
        if broker_ticker_mgr:
            try:
                supabase_admin = get_supabase_admin()
                conn_resp = supabase_admin.table("broker_connections").select("broker_name,access_token").eq(
                    "user_id", user_id).eq("status", "connected").maybe_single().execute()
                if conn_resp and conn_resp.data:
                    from ..data.brokers.credentials import decrypt_credentials
                    broker_name = conn_resp.data["broker_name"]
                    creds = decrypt_credentials(conn_resp.data["access_token"])
                    await broker_ticker_mgr.connect_user_ticker(user_id, broker_name, creds)
            except Exception as e:
                logger.debug(f"Broker ticker auto-connect skipped for {user_id}: {e}")
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                    continue
                try:
                    message = json.loads(data)
                    action = message.get("action", "")
                    channel = message.get("channel", "")
                    if action == "subscribe":
                        await handle_subscribe(user_id, message)
                        await websocket.send_json({"type": "subscribed", "channel": channel, "timestamp": datetime.utcnow().isoformat()})
                    elif action == "unsubscribe":
                        await handle_unsubscribe(user_id, message)
                        await websocket.send_json({"type": "unsubscribed", "channel": channel, "timestamp": datetime.utcnow().isoformat()})
                    elif action == "get_prices":
                        await send_price_update(user_id, message.get("symbols", []))
                    else:
                        await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "echo", "data": data, "timestamp": datetime.utcnow().isoformat()})
        except WebSocketDisconnect:
            if user_id and manager:
                manager.disconnect(user_id)
                if broker_ticker_mgr:
                    await broker_ticker_mgr.disconnect_user_ticker(user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if user_id and manager:
            manager.disconnect(user_id)
        await websocket.close(code=4000)


# PR 56 — legacy /ws/{token} endpoint removed. Tokens in URLs leaked to
# server logs, CDN caches, browser history, and proxy access logs. The
# frontend has been on the header-auth /ws path since PR 20+; removing
# the legacy path prevents downgrade attacks (an attacker who captures
# the token via logs can't open an auth'd socket on /ws/{token}).


async def handle_subscribe(user_id: str, message: Dict):
    """Handle WebSocket subscription requests"""
    channel = message.get("channel", "")
    symbols = message.get("symbols", [])

    if channel == "price" and symbols:
        for symbol in symbols:
            manager.subscribe_to_symbol(user_id, symbol)
        # Also subscribe on broker ticker for real-time streaming
        broker_ticker_mgr = getattr(app.state, 'broker_ticker_manager', None)
        if broker_ticker_mgr and user_id in broker_ticker_mgr._user_tickers:
            await broker_ticker_mgr.subscribe_symbols(user_id, symbols)
        logger.info(f"User {user_id} subscribed to price updates for: {symbols}")

    elif channel == "signals":
        if user_id in manager.user_subscriptions:
            manager.user_subscriptions[user_id].add("signals")
        logger.info(f"User {user_id} subscribed to signal updates")

    elif channel == "portfolio":
        if user_id in manager.user_subscriptions:
            manager.user_subscriptions[user_id].add("portfolio")
        logger.info(f"User {user_id} subscribed to portfolio updates")

    elif channel == "notifications":
        if user_id in manager.user_subscriptions:
            manager.user_subscriptions[user_id].add("notifications")
        logger.info(f"User {user_id} subscribed to notifications")


async def handle_unsubscribe(user_id: str, message: Dict):
    """Handle WebSocket unsubscription requests"""
    channel = message.get("channel", "")
    symbols = message.get("symbols", [])

    if channel == "price" and symbols:
        for symbol in symbols:
            manager.unsubscribe_from_symbol(user_id, symbol)
        logger.info(f"User {user_id} unsubscribed from price updates for: {symbols}")

    elif channel in ["signals", "portfolio", "notifications"]:
        if user_id in manager.user_subscriptions:
            manager.user_subscriptions[user_id].discard(channel)
        logger.info(f"User {user_id} unsubscribed from {channel}")


async def send_price_update(user_id: str, symbols: List[str]):
    """Send price update to specific user"""
    try:
        from ..data.market import get_market_data_provider
        provider = get_market_data_provider()

        quotes = await asyncio.to_thread(provider.get_quotes_batch, symbols)

        price_data = []
        for symbol, quote in quotes.items():
            if quote:
                price_data.append({
                    "symbol": symbol,
                    "ltp": quote.ltp,
                    "change": quote.change,
                    "change_percent": quote.change_percent,
                    "volume": quote.volume,
                    "timestamp": quote.timestamp.isoformat()
                })

        if manager and user_id in manager.active_connections:
            await manager.active_connections[user_id].send_json({
                "type": "price_update",
                "data": price_data,
                "timestamp": datetime.utcnow().isoformat()
            })
    except Exception as e:
        logger.error(f"Failed to send price update: {e}")

# ============================================================================
# SCREENER ROUTES (Quant X Screener)
# ============================================================================

try:
    from .screener_routes import register_screener_routes
    register_screener_routes(app)
    logger.info("✅ Quant X Screener routes registered")
except Exception as e:
    logger.warning(f"Screener routes not available: {e}")

# ============================================================================
# BROKER ROUTES
# ============================================================================

try:
    from .broker_routes import router as broker_router
    app.include_router(broker_router, prefix="/api")
    logger.info("✅ Broker OAuth routes registered")
except Exception as e:
    logger.warning(f"Broker routes not available: {e}")

# ============================================================================
# PAYMENT ROUTES (Razorpay)
# ============================================================================

try:
    from .payment_routes import router as payment_router
    app.include_router(payment_router, prefix="/api")
    logger.info("✅ Payment routes registered")
except Exception as e:
    logger.warning(f"Payment routes not available: {e}")

# ============================================================================
# MARKETPLACE ROUTES
# ============================================================================

try:
    from .marketplace_routes import router as marketplace_router
    app.include_router(marketplace_router)
    logger.info("✅ Marketplace routes registered")
except Exception as e:
    logger.warning(f"Marketplace routes not available: {e}")

# ============================================================================
# AUTH ROUTES — extracted from app.py 2026-05-07 (P1-5j)
# ============================================================================

try:
    from .auth_routes import router as auth_router
    app.include_router(auth_router)
    logger.info("✅ Auth routes registered")
except Exception as e:
    logger.warning(f"Auth routes not available: {e}")

# ============================================================================
# USER PROFILE ROUTES — extracted from app.py 2026-05-07 (P1-5k)
# ============================================================================

try:
    from .user_routes import router as user_router
    app.include_router(user_router)
    logger.info("✅ User profile routes registered")
except Exception as e:
    logger.warning(f"User profile routes not available: {e}")

# ============================================================================
# SIGNALS ROUTES — extracted from app.py 2026-05-07 (P1-5l)
# ============================================================================

try:
    from .signals_routes import router as signals_router
    app.include_router(signals_router)
    logger.info("✅ Signals routes registered")
except Exception as e:
    logger.warning(f"Signals routes not available: {e}")

# PR-D Strategy DSL validate + discovery routes (no CRUD yet — that's PR-F)
try:
    from .strategies_routes import router as strategies_router
    app.include_router(strategies_router)
    logger.info("✅ Strategies routes registered (PR-D DSL validate)")
except Exception as e:
    logger.warning(f"Strategies routes not available: {e}")

# PR-G2 — Strategy Discovery Engine routes
try:
    from .discovery_routes import router as discovery_router
    app.include_router(discovery_router)
    logger.info("✅ Strategy Discovery routes registered (PR-G2)")
except Exception as e:
    logger.warning(f"Strategy Discovery routes not available: {e}")

# ============================================================================
# TRADES ROUTES — extracted from app.py 2026-05-07 (P1-5m)
# ============================================================================

try:
    from .trades_routes import router as trades_router
    app.include_router(trades_router)
    logger.info("✅ Trades routes registered")
except Exception as e:
    logger.warning(f"Trades routes not available: {e}")

# ============================================================================
# POSITIONS ROUTES — extracted from app.py 2026-05-07 (P1-5n)
# ============================================================================

try:
    from .positions_routes import router as positions_router
    app.include_router(positions_router)
    logger.info("✅ Positions routes registered")
except Exception as e:
    logger.warning(f"Positions routes not available: {e}")

# ============================================================================
# PORTFOLIO ROUTES — extracted from app.py 2026-05-07 (P1-5o)
# ============================================================================

try:
    from .portfolio_routes import router as portfolio_router
    app.include_router(portfolio_router)
    logger.info("✅ Portfolio routes registered")
except Exception as e:
    logger.warning(f"Portfolio routes not available: {e}")

# ============================================================================
# MARKET ROUTES — extracted from app.py 2026-05-07 (P1-5p)
# ============================================================================

try:
    from .market_routes import router as market_router
    app.include_router(market_router)
    logger.info("✅ Market routes registered")
except Exception as e:
    logger.warning(f"Market routes not available: {e}")

# ============================================================================
# NOTIFICATIONS ROUTES — extracted from app.py 2026-05-07 (P1-5q)
# ============================================================================

try:
    from .notifications_routes import router as notifications_router
    app.include_router(notifications_router)
    logger.info("✅ Notifications routes registered")
except Exception as e:
    logger.warning(f"Notifications routes not available: {e}")

# ============================================================================
# PUSH SUBSCRIPTION ROUTES — extracted from app.py 2026-05-07 (P1-5q)
# ============================================================================

try:
    from .push_routes import router as push_router
    app.include_router(push_router)
    logger.info("✅ Push subscription routes registered")
except Exception as e:
    logger.warning(f"Push subscription routes not available: {e}")

# ============================================================================
# WATCHLIST ROUTES — extracted from app.py 2026-05-07 (P1-5q)
# ============================================================================

try:
    from .watchlist_routes import router as watchlist_router
    app.include_router(watchlist_router)
    logger.info("✅ Watchlist routes registered")
except Exception as e:
    logger.warning(f"Watchlist routes not available: {e}")

# ============================================================================
# SUBSCRIPTION ROUTES — extracted from app.py 2026-05-07 (P1-5r)
# ============================================================================

try:
    from .subscription_routes import router as subscription_router
    app.include_router(subscription_router)
    logger.info("✅ Subscription routes registered")
except Exception as e:
    logger.warning(f"Subscription routes not available: {e}")

# ============================================================================
# SYSTEM / DASHBOARD / ASSISTANT ROUTES — extracted from app.py 2026-05-07 (P1-5s)
# ============================================================================

try:
    from .system_routes import router as system_router
    app.include_router(system_router)
    logger.info("✅ System status route registered")
except Exception as e:
    logger.warning(f"System status route not available: {e}")

try:
    from .dashboard_routes import router as dashboard_router
    app.include_router(dashboard_router)
    logger.info("✅ Dashboard route registered")
except Exception as e:
    logger.warning(f"Dashboard route not available: {e}")

try:
    from .assistant_routes import router as assistant_router
    app.include_router(assistant_router)
    logger.info("✅ Assistant routes registered")
except Exception as e:
    logger.warning(f"Assistant routes not available: {e}")

try:
    from .autopilot_streams_routes import router as autopilot_streams_router
    app.include_router(autopilot_streams_router)
    logger.info("✅ AutoPilot Streams routes registered (PR-AS — per-stream toggles)")
except Exception as e:
    logger.warning(f"AutoPilot Streams routes not available: {e}")

try:
    from .strategy_runner_routes import router as strategy_runner_router
    app.include_router(strategy_runner_router)
    logger.info("✅ Strategy Runner routes registered (PR-FAN — fan-out engine)")
except Exception as e:
    logger.warning(f"Strategy Runner routes not available: {e}")

# ============================================================================
# PAPER TRADING ROUTES
# ============================================================================

try:
    from .paper_routes import router as paper_router
    app.include_router(paper_router)
    logger.info("✅ Paper Trading routes registered")
except Exception as e:
    logger.warning(f"Paper Trading routes not available: {e}")

# ============================================================================
# AI AGENT ROUTES (PR 8) — Copilot / FinRobot / TradingAgents
# ============================================================================

try:
    from .ai_routes import router as ai_router
    app.include_router(ai_router, prefix="/api")
    logger.info("✅ AI agent routes registered (copilot / finrobot / debate)")
except Exception as e:
    logger.warning(f"AI agent routes not available: {e}")

# ============================================================================
# PUBLIC TRUST-SURFACE ROUTES (PR 18) — /regime /track-record /models
# ============================================================================

try:
    from .public_routes import router as public_router
    app.include_router(public_router, prefix="/api")
    logger.info("✅ Public trust-surface routes registered")
except Exception as e:
    logger.warning(f"Public routes not available: {e}")

# ============================================================================
# AUTO-TRADER ROUTES (PR 28) — F4 Elite dashboard control plane
# ============================================================================

try:
    from .auto_trader_routes import router as auto_trader_router
    app.include_router(auto_trader_router, prefix="/api")
    logger.info("✅ Auto-trader routes registered (F4)")
except Exception as e:
    logger.warning(f"Auto-trader routes not available: {e}")

# ============================================================================
# AI PORTFOLIO ROUTES (PR 29) — F5 AI SIP Elite rebalance dashboard
# ============================================================================

# F5 AI SIP / AI Portfolio routes removed 2026-06-06 (feature deleted).

# ============================================================================
# F&O STRATEGIES ROUTES (PR 30) — F6 Elite options strategy recommender
# ============================================================================

try:
    from .fo_strategies_routes import router as fo_strategies_router
    app.include_router(fo_strategies_router, prefix="/api")
    logger.info("✅ F&O Strategies routes registered (F6)")
except Exception as e:
    logger.warning(f"F&O Strategies routes not available: {e}")

# ============================================================================
# EARNINGS ROUTES (PR 31) — F9 Earnings predictor + calendar
# ============================================================================

try:
    from .earnings_routes import router as earnings_router
    app.include_router(earnings_router, prefix="/api")
    logger.info("✅ Earnings routes registered (F9)")
except Exception as e:
    logger.warning(f"Earnings routes not available: {e}")

# ============================================================================
# DOSSIER ROUTES (PR 33) — per-stock consolidated AI engine output
# ============================================================================

try:
    from .dossier_routes import router as dossier_router
    app.include_router(dossier_router, prefix="/api")
    logger.info("✅ Dossier routes registered (N2)")
except Exception as e:
    logger.warning(f"Dossier routes not available: {e}")

# ============================================================================
# PORTFOLIO DOCTOR ROUTES (PR 34) — F7 InsightAI whole-portfolio analysis
# ============================================================================

try:
    from .portfolio_doctor_routes import router as portfolio_doctor_router
    app.include_router(portfolio_doctor_router, prefix="/api")
    logger.info("✅ Portfolio Doctor routes registered (F7)")
except Exception as e:
    logger.warning(f"Portfolio Doctor routes not available: {e}")

# ============================================================================
# ONBOARDING ROUTES (PR 37) — N5 risk-profile quiz
# ============================================================================

try:
    from .onboarding_routes import router as onboarding_router
    app.include_router(onboarding_router, prefix="/api")
    logger.info("✅ Onboarding routes registered (N5)")
except Exception as e:
    logger.warning(f"Onboarding routes not available: {e}")

# ============================================================================
# MANAGED-MODE ROUTES (dual-mode 2026-06-12) — beginner Home aggregate
# ============================================================================

try:
    from .managed_routes import router as managed_router
    app.include_router(managed_router, prefix="/api")
    logger.info("✅ Managed-mode routes registered")
except Exception as e:
    logger.warning(f"Managed-mode routes not available: {e}")

# ============================================================================
# WEEKLY REVIEW ROUTES (PR 38) — N10 Sunday personal review
# ============================================================================

try:
    from .weekly_review_routes import router as weekly_review_router
    app.include_router(weekly_review_router, prefix="/api")
    logger.info("✅ Weekly Review routes registered (N10)")
except Exception as e:
    logger.warning(f"Weekly Review routes not available: {e}")

# ============================================================================
# IPO ROUTES (Phase 4) — primary-market calendar (open + upcoming, NSE feed)
# ============================================================================
try:
    from .ipo_routes import router as ipo_router
    app.include_router(ipo_router, prefix="/api")
    logger.info("✅ IPO routes registered")
except Exception as e:
    logger.warning(f"IPO routes not available: {e}")

# ============================================================================
# WATCHLIST LIVE ROUTES (PR 39) — enriched per-symbol engine snapshots
# ============================================================================

try:
    from .watchlist_live_routes import router as watchlist_live_router
    app.include_router(watchlist_live_router, prefix="/api")
    logger.info("✅ Watchlist Live routes registered")
except Exception as e:
    logger.warning(f"Watchlist Live routes not available: {e}")

# ============================================================================
# ALERTS STUDIO ROUTES (PR 40) — N11 per-event channel routing
# ============================================================================

try:
    from .alerts_routes import router as alerts_router
    app.include_router(alerts_router, prefix="/api")
    logger.info("✅ Alerts Studio routes registered (N11)")
except Exception as e:
    logger.warning(f"Alerts Studio routes not available: {e}")

# ============================================================================
# REFERRALS ROUTES (PR 42) — N12 virality loop
# ============================================================================

try:
    from .referrals_routes import router as referrals_router
    app.include_router(referrals_router, prefix="/api")
    logger.info("✅ Referrals routes registered (N12)")
except Exception as e:
    logger.warning(f"Referrals routes not available: {e}")

# ============================================================================
# VISION ROUTES (PR 46) — B2 chart-vision analysis
# ============================================================================

try:
    from .vision_routes import router as vision_router
    app.include_router(vision_router, prefix="/api")
    logger.info("✅ Chart-vision routes registered (B2)")
except Exception as e:
    logger.warning(f"Chart-vision routes not available: {e}")

# ============================================================================
# TELEGRAM ROUTES (PR 55) — onboarding connect + bot webhook
# ============================================================================

try:
    from .telegram_routes import router as telegram_router
    app.include_router(telegram_router, prefix="/api")
    logger.info("✅ Telegram connect routes registered (PR 55)")
except Exception as e:
    logger.warning(f"Telegram connect routes not available: {e}")

# ============================================================================
# TELEMETRY ROUTES (PR 57) — client-side error ingestion
# ============================================================================

try:
    from .telemetry_routes import router as telemetry_router
    app.include_router(telemetry_router, prefix="/api")
    logger.info("✅ Client-error telemetry routes registered (PR 57)")
except Exception as e:
    logger.warning(f"Telemetry routes not available: {e}")

# ============================================================================
# WHATSAPP ROUTES (PR 60) — F12 Pro digest channel (opt-in + OTP)
# ============================================================================

try:
    from .whatsapp_routes import router as whatsapp_router
    app.include_router(whatsapp_router, prefix="/api")
    logger.info("✅ WhatsApp routes registered (PR 60)")
except Exception as e:
    logger.warning(f"WhatsApp routes not available: {e}")

# ============================================================================
# ADMIN ROUTES
# ============================================================================

try:
    from .admin_routes import register_admin_routes
    register_admin_routes(app)
    logger.info("✅ Admin routes registered")
except Exception as e:
    logger.warning(f"Admin routes not available: {e}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.app:app",
        host="0.0.0.0",  # nosec B104 - container server bind
        port=8000,
        reload=settings.DEBUG
    )
