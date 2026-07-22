"""
Market data API routes — quotes, indices, regime, OHLC, status, risk.

Read-only views over Kite live data + ``market_data`` table snapshots.
``/api/market/regime`` and ``/api/ai/performance`` are public (no auth)
since they back unauthenticated landing-page widgets.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, Tuple

from fastapi import APIRouter, HTTPException, Query, Request

from ..services.entitlement import (
    DataClass,
    broker_lock,
    entitlement_and_user,
    entitlement_for,
    entitlement_marker,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Market"])


# ── in-process TTL cache with stale-while-revalidate ──────────────────
# Live market endpoints hit yfinance / Kite on every request. The
# regime endpoint in particular takes 40+ seconds cold because it
# scans the full universe. A naive TTL cache still makes the first user
# after expiry wait the cold time.
#
# Pattern: cache values forever. Each entry tracks when it was last
# refreshed. Within TTL → fresh hit, return immediately. Past TTL →
# return stale value immediately AND fire a background refresh. New
# requesters within the refresh window all share the in-flight task
# (no thundering herd).
_market_cache: Dict[str, Tuple[float, Any]] = {}
_refresh_locks: Dict[str, asyncio.Lock] = {}


def _cache_get_any(key: str) -> Tuple[float, Any] | None:
    return _market_cache.get(key)


def _cache_get(key: str, ttl: float):
    hit = _market_cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]
    return None


def _cache_put(key: str, value: Any) -> Any:
    _market_cache[key] = (time.monotonic(), value)
    if len(_market_cache) > 512:
        oldest = min(_market_cache.items(), key=lambda kv: kv[1][0])[0]
        _market_cache.pop(oldest, None)
    return value


async def _stale_while_revalidate(key: str, ttl: float, producer, *, is_valid=None, retry_ttl: float = 25.0):
    """Return cached value immediately if any exists; refresh in background
    if it's beyond `ttl`. If nothing is cached, block on first compute.

    Reliability hardening (so data loads accurately every time):
      • ``is_valid(value)`` decides whether a produced value is "good" (has real
        data). Default: not None. A transient upstream failure — e.g. yfinance
        throttled → empty ``{"items": []}`` — is NEVER allowed to overwrite a
        previously-good cached value.
      • An invalid cached value (nothing good yet) is re-fetched after the short
        ``retry_ttl`` instead of the full ``ttl``, so a cold-start blip clears in
        seconds rather than serving empty for minutes.
    """
    ok = is_valid or (lambda v: v is not None)
    entry = _cache_get_any(key)
    now = time.monotonic()
    if entry is None:
        # Cold — must block. Compute + cache (even if invalid; the next caller
        # retries after retry_ttl instead of stampeding upstream).
        value = await producer()
        _cache_put(key, value)
        return value

    cached_at, value = entry
    # Good values live for `ttl`; not-yet-good values only for `retry_ttl`.
    age_limit = ttl if ok(value) else retry_ttl
    if now - cached_at < age_limit:
        return value

    # Stale — fire-and-forget refresh, return current value immediately.
    lock = _refresh_locks.setdefault(key, asyncio.Lock())
    if not lock.locked():
        async def _refresh():
            async with lock:
                # Re-check inside lock to avoid double work.
                fresh_entry = _cache_get_any(key)
                if fresh_entry:
                    fresh_limit = ttl if ok(fresh_entry[1]) else retry_ttl
                    if time.monotonic() - fresh_entry[0] < fresh_limit:
                        return
                try:
                    new_value = await producer()
                    if ok(new_value) or not ok(value):
                        # Accept a good result, or any result when we had nothing
                        # good to protect. Never overwrite good data with empty.
                        _cache_put(key, new_value)
                    else:
                        # New fetch came back empty but we hold a good value:
                        # keep it and reset its clock (serve last-good, retry
                        # next cycle) rather than blanking the UI.
                        _cache_put(key, value)
                except Exception as e:
                    logger.warning("Background refresh failed for %s: %s", key, e)
        asyncio.create_task(_refresh())
    return value


def _get_supabase_admin():
    from .app import get_supabase_admin
    return get_supabase_admin()


def _get_current_user_dep():
    from .app import get_current_user
    return get_current_user


@router.get("/api/market/status")
async def get_market_status():
    """Get current market status — open/closed + trading-day check."""
    try:
        from ..data.market import get_market_data_provider
        provider = get_market_data_provider()
        status = provider.get_market_status()
        return {
            "is_trading_day": status.is_trading_day,
            "is_market_open": status.is_market_open,
            "market_phase": status.market_phase,
            "next_open": status.next_open.isoformat() if status.next_open else None,
            "reason": status.reason,
        }
    except Exception as e:
        logger.error(f"Market status error: {e}")
        return {
            "is_trading_day": True,
            "is_market_open": False,
            "market_phase": "UNKNOWN",
            "reason": str(e),
        }


@router.get("/api/market/quote/{symbol}")
async def get_market_quote(symbol: str, request: Request):
    """Get real-time quote for a symbol via the Kite data provider.

    Cached 5s in-process — display precision, not order-routing precision.
    Collapses N concurrent requests for the same symbol into one upstream call.

    Path-A: raw live NSE quotes require a data licence or the user's own
    connected broker. A broker-entitled user is served from THEIR broker feed;
    otherwise honest-empty + broker-lock marker.
    """
    ent, uid = entitlement_and_user(request, DataClass.LIVE_QUOTE)
    if not ent.allowed:
        return entitlement_marker(ent, {"symbol": symbol.upper(), "ltp": None})

    # Bring-your-own-broker: source from the user's own licensed feed.
    if ent.source == "broker" and uid:
        from ..services.user_broker_data import quote as _user_quote

        user_q = await asyncio.to_thread(_user_quote, uid, symbol)
        if user_q is not None:
            return user_q
        return broker_lock(DataClass.LIVE_QUOTE, {"symbol": symbol.upper(), "ltp": None})

    cache_key = f"quote:{symbol}"
    cached = _cache_get(cache_key, ttl=5.0)
    if cached is not None:
        return cached

    try:
        from ..data.market import get_market_data_provider
        provider = get_market_data_provider()
        quote = await asyncio.to_thread(provider.get_quote, symbol)

        if not quote:
            raise HTTPException(status_code=404, detail="Quote not found")

        return _cache_put(cache_key, {
            "symbol": quote.symbol,
            "ltp": quote.ltp,
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "close": quote.close,
            "volume": quote.volume,
            "change": quote.change,
            "change_percent": quote.change_percent,
            "timestamp": quote.timestamp.isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/market/sentiment/{symbol}")
async def get_stock_sentiment(symbol: str):
    """On-demand news 'Mood' for ANY stock.

    Runs the standalone sentiment engine live against the latest headlines —
    not the stored ``news_sentiment`` table — so a user can pull Mood for any
    symbol on the stock page / Markets desk. Honest-empty (``available=False``)
    when there's no recent news or the classifier is unavailable: never a
    fabricated score (no-fallbacks lock). Successful results cached 10 min;
    empty/failed results are NOT cached so they self-heal.
    """
    sym = symbol.upper().strip().replace(".NS", "")
    cache_key = f"sentiment:{sym}"
    cached = _cache_get(cache_key, ttl=600.0)
    if cached is not None:
        return cached

    empty = {
        "symbol": sym, "available": False, "mean_score": None, "label": None,
        "headline_count": 0, "positive_count": 0, "negative_count": 0,
        "neutral_count": 0, "headlines": [], "sources": [],
    }
    try:
        from ..ai.sentiment.engine import get_sentiment_engine
        row = await get_sentiment_engine().score_symbol(sym, lookback_days=3)
    except Exception as e:
        logger.warning("sentiment fetch failed for %s: %s", sym, e)
        return empty
    if not row:
        return empty

    score = float(row.get("mean_score") or 0.0)
    label = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
    return _cache_put(cache_key, {
        "symbol": sym,
        "available": True,
        "mean_score": round(score, 3),
        "label": label,
        "headline_count": row.get("headline_count", 0),
        "positive_count": row.get("positive_count", 0),
        "negative_count": row.get("negative_count", 0),
        "neutral_count": row.get("neutral_count", 0),
        "headlines": row.get("sample_headlines", []),
        "sources": row.get("sources", []),
    })


@router.get("/api/market/indices")
async def get_market_indices(request: Request):
    """Get index data (Nifty, Bank Nifty, VIX).

    Stale-while-revalidate: 10s TTL, stale served instantly while a
    background refresh runs. First-ever request is the only blocking one.

    Path-A: raw NSE index levels require a data licence or the user's own
    connected broker. A broker-entitled user is served from THEIR broker feed;
    otherwise honest-empty + broker-lock marker.
    """
    _empty = {
        "nifty": {"ltp": 0, "change": 0, "change_percent": 0},
        "banknifty": {"ltp": 0, "change": 0, "change_percent": 0},
    }
    ent, uid = entitlement_and_user(request, DataClass.INDEX)
    if not ent.allowed:
        return entitlement_marker(ent, _empty)

    # Bring-your-own-broker: source index levels from the user's own feed.
    if ent.source == "broker" and uid:
        from ..services.user_broker_data import indices as _user_indices

        user_idx = await asyncio.to_thread(_user_indices, uid)
        if user_idx is not None:
            return user_idx
        return broker_lock(DataClass.INDEX, _empty)

    async def _produce():
        try:
            from ..data.market import get_market_data_provider
            provider = get_market_data_provider()
            overview = await provider.get_market_overview_async()
            if overview.get("nifty", {}).get("ltp", 0) > 0:
                return overview
        except Exception as e:
            logger.warning(f"Kite indices error: {e}")
        return {
            "nifty": {"ltp": 0, "change": 0, "change_percent": 0},
            "banknifty": {"ltp": 0, "change": 0, "change_percent": 0},
        }

    return await _stale_while_revalidate("indices", ttl=10.0, producer=_produce)


# ── Global cues (pre-market research hub) ─────────────────────────────────
# GIFT Nifty (gap-direction proxy) + US close + Asia (live during India
# pre-open) + commodities + DXY + US 10Y + BTC. All FOREIGN / non-NSE feeds,
# so SAFE to show to everyone (not licence-gated NSE market data). Public;
# yfinance (Tier-3 source); 5-min stale-while-revalidate. On any failure a
# given item is honest-skipped (null last dropped) — never faked.
#
# NOTE (SEBI): every ticker here is a foreign / global instrument. We never
# use ^NSEI (NIFTY 50 spot) as a GIFT proxy — that is a gated NSE index level.
# GIFT Nifty has no reliable free yfinance symbol; when unavailable it is
# honest-skipped rather than substituted.
_GLOBAL_TICKERS = [
    ("giftnifty", ["GIFTNIFTY", "NIFTY_F1"], "GIFT NIFTY"),
    ("sp500", ["^GSPC"], "S&P 500"),
    ("nasdaq", ["^IXIC"], "Nasdaq"),
    ("dow", ["^DJI"], "Dow Jones"),
    ("nikkei", ["^N225"], "Nikkei 225"),
    ("hangseng", ["^HSI"], "Hang Seng"),
    ("crude", ["BZ=F"], "Brent crude"),
    ("gold", ["GC=F"], "Gold"),
    ("dxy", ["DX-Y.NYB", "DX=F"], "Dollar index"),
    ("us10y", ["^TNX"], "US 10Y"),
    ("btc", ["BTC-USD"], "Bitcoin"),
]


def _closes_from(df, sym):
    """Extract a clean Close series for ``sym`` from a yfinance frame that may
    be single- or multi-indexed. Returns (last, prev) or (None, None)."""
    try:
        if df is None or len(df) == 0:
            return None, None
        if hasattr(df.columns, "levels"):          # multi-ticker frame
            closes = df[sym]["Close"].dropna()
        else:                                        # single-ticker frame
            closes = df["Close"].dropna()
        if len(closes) >= 2:
            return float(closes.iloc[-1]), float(closes.iloc[-2])
        if len(closes) == 1:
            return float(closes.iloc[-1]), None
    except Exception:
        pass
    return None, None


def _fetch_global_cues() -> Dict[str, Any]:
    """Best-effort global cues. One batch yfinance download, then a per-ticker
    retry for anything that came back empty (batch group_by is flaky across
    yfinance versions). Honest-skip null items."""
    items: list[dict[str, Any]] = []
    try:
        import yfinance as yf
    except Exception as e:  # yfinance unavailable — honest-empty
        logger.warning("Global cues: yfinance import failed: %s", e)
        return {"items": items, "source": "yfinance"}

    # First symbol per key is the primary; keep the flat batch list.
    primary = {key: syms[0] for key, syms, _ in _GLOBAL_TICKERS}
    batch = None
    try:
        batch = yf.download(
            " ".join(primary.values()), period="5d", interval="1d",
            progress=False, group_by="ticker", threads=True,
        )
    except Exception as e:
        logger.debug("Global cues batch download failed (will per-ticker): %s", e)

    for key, syms, label in _GLOBAL_TICKERS:
        last = prev = None
        # Try each candidate symbol until one yields a close.
        for sym in syms:
            if batch is not None and sym == primary[key]:
                last, prev = _closes_from(batch, sym)
            if last is None:
                try:
                    hist = yf.Ticker(sym).history(period="5d")
                    last, prev = _closes_from(hist, sym)
                except Exception:
                    last, prev = None, None
            if last is not None:
                break
        if last is None:
            continue  # honest-skip: no real value for this item
        chg = round((last - prev) / prev * 100, 2) if prev else None
        items.append({
            "key": key, "label": label,
            "last": round(last, 2),
            "change_pct": chg,
        })
    return {"items": items, "source": "yfinance"}


@router.get("/api/market/global")
async def get_global_cues():
    """Global pre-market cues (GIFT Nifty / US / Asia / commodities / DXY /
    US 10Y / BTC).

    Public (no auth) — backs the Markets research hub + Daily Briefing. All
    foreign / global instruments (SAFE, non-NSE). yfinance; 5-minute
    stale-while-revalidate so only the first request after boot pays the cost.

    Item shape: ``{key, label, last, change_pct}``; null items are skipped.
    """
    async def _produce():
        return await asyncio.to_thread(_fetch_global_cues)

    return await _stale_while_revalidate(
        "global_cues", ttl=300.0, producer=_produce,
        is_valid=lambda v: bool(v and v.get("items")),
    )


@router.get("/api/market/deals")
async def get_big_deals():
    """Big bulk/block deals (by ₹ value, last few sessions) + upcoming
    corporate actions for F&O names. Public (no auth) — these are NSE's own
    EOD-PUBLISHED disclosure reports (same SEBI lane as /fii-dii), labelled
    as such. 1h service cache + stale-while-revalidate."""
    async def _produce():
        from ..services.market.deals import big_deals
        return await asyncio.to_thread(big_deals)

    return await _stale_while_revalidate(
        "big_deals", ttl=1800.0, producer=_produce,
        is_valid=lambda v: bool(v and (v.get("deals") or v.get("corporate_actions"))),
    )


@router.get("/api/market/fii-dii")
async def get_fii_dii_eod():
    """FII/DII EOD daily net (cash) + a short trailing trend. Public (no auth).

    These are EOD-published market statistics (net figures NSE/SEBI publish
    after the close, shown on every finance site) — NOT a real-time intraday
    feed. Served to EVERYONE, labelled ``provisional``. Values in ₹ Cr.

    SEBI note: a SEBI-registered professional should confirm the
    EOD-published-statistics classification before paid / public launch. This
    endpoint never exposes intraday NSE quotes.

    Shape: ``{date, provisional, fii:{cash_net, fno_net}, dii:{cash_net},
    trend:[{date, fii_cash, dii_cash}], source}``. Cached 5-min.
    """
    async def _produce():
        from ..services.briefing.market_briefing import fii_dii_eod
        return await asyncio.to_thread(fii_dii_eod, 5)

    return await _stale_while_revalidate(
        "fii_dii_eod", ttl=300.0, producer=_produce,
        is_valid=lambda v: bool(v and ((v.get("fii") or {}).get("cash_net") is not None or v.get("trend"))),
    )


@router.get("/api/market/briefing")
async def get_market_briefing(session: str = Query("auto", description="auto | premarket | postmarket")):
    """AI Daily Market Briefing — pre-market read + post-market wrap.

    Public (no auth) — built ENTIRELY from SAFE data (global overnight cues +
    EOD/derived India context + FII/DII EOD statistics + calendar events) plus
    one cached daily LLM narrative. Shows to everyone; never leaks real-time
    intraday NSE quotes.

    ``session=auto`` resolves premarket / intraday / postmarket from the clock
    (IST). Generated once per (trading-date, session) and shared via the daily
    LLM cache — first visitor triggers, everyone else shares it.
    """
    from ..services.briefing.market_briefing import build_briefing, current_session

    sess = session if session in ("premarket", "postmarket") else current_session()
    cache_key = f"briefing:{sess}:{date.today().isoformat()}"

    async def _produce():
        return await asyncio.to_thread(build_briefing, sess)

    # Short TTL: the LLM narrative itself is day-cached inside build_briefing;
    # this in-process cache just spares repeat fact-assembly within a few min.
    # A briefing is only "good" once it carries a headline (the deterministic
    # core) — so a rare failed assembly never gets cached over a real one.
    return await _stale_while_revalidate(
        cache_key, ttl=180.0, producer=_produce,
        is_valid=lambda v: bool(v and v.get("headline")),
    )


@router.get("/api/market/news")
async def get_market_news():
    """Live market-moving headlines from FREE keyless RSS feeds.

    Public (no auth) — news headlines are public information, not licensed NSE
    market data, so this is NOT entitlement-gated. 3-minute stale-while-
    revalidate (kept short so the tape reads near real-time); honest-empty on
    total feed failure.
    """
    async def _produce():
        try:
            from ..services.news.market_news import fetch_market_news
            items = await fetch_market_news(15)
            return {"items": items, "source": "rss"}
        except Exception as e:
            logger.warning("market news endpoint failed: %s", e)
            return {"items": [], "source": "rss"}

    return await _stale_while_revalidate("market_news", ttl=180.0, producer=_produce)


@router.get("/api/market/regime")
async def get_market_regime_public():
    """Get current market regime (Bull / Bear / Sideways).

    Public endpoint (no auth) — backs the dashboard RegimeBanner and
    AIPerformanceWidget. Proxies to the live screener's regime detector.

    Cold compute = 40+ seconds (full universe scan). Stale-while-revalidate
    with 5-min TTL: once warm, every user gets a sub-millisecond response
    while a background task refreshes when stale. Only the very first
    request after server boot pays the cold cost.
    """
    async def _produce():
        try:
            from ..data.screener.engine import get_live_screener
            screener = get_live_screener()
            regime_data = await screener.get_market_regime()

            regime_raw = regime_data.get("regime", "SIDEWAYS").upper()
            regime_map = {"BULL": "bull", "BEAR": "bear", "SIDEWAYS": "sideways"}
            regime = regime_map.get(regime_raw, "sideways")
            confidence = regime_data.get("confidence", 50)
            confidence_norm = round(confidence / 100, 2) if confidence > 1 else round(confidence, 2)

            return {
                "success": True,
                "current": {
                    "regime": regime,
                    "confidence": confidence_norm,
                    "days_active": regime_data.get("days_active", 1),
                },
                "regime": regime,
                "confidence": confidence_norm,
                "factors": {
                    "breadth_200sma": regime_data.get("breadth_200sma", 50),
                    "bullish_macd_pct": regime_data.get("bullish_macd_pct", 50),
                },
            }
        except Exception as e:
            logger.warning(f"Market regime endpoint failed: {e}")
            return {
                "success": True,
                "current": {"regime": "sideways", "confidence": 0.5, "days_active": 1},
                "regime": "sideways",
                "confidence": 0.5,
            }

    return await _stale_while_revalidate("regime", ttl=300.0, producer=_produce)


@router.get("/api/ai/performance", tags=["AI"])
async def get_ai_performance():
    """Get AI model performance metrics for the dashboard widget.

    Public endpoint (no auth) — returns filtered vs unfiltered win rates
    plus today's scored-signal count.
    """
    try:
        supabase = _get_supabase_admin()
        today = date.today().isoformat()
        thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()

        today_signals = (
            supabase.table("signals")
            .select("id", count="exact")
            .eq("date", today)
            .eq("status", "active")
            .execute()
        )
        signals_today = today_signals.count or len(today_signals.data or [])

        perf = (
            supabase.table("signals")
            .select("id, confidence, model_agreement, status")
            .gte("date", thirty_days_ago)
            .in_("status", ["target_hit", "stop_hit", "expired"])
            .execute()
        )
        closed_signals = perf.data or []
        total = len(closed_signals)

        if total > 0:
            filtered = [s for s in closed_signals if (s.get("model_agreement") or 0) >= 3]
            unfiltered_wins = sum(1 for s in closed_signals if s.get("status") == "target_hit")
            filtered_wins = sum(1 for s in filtered if s.get("status") == "target_hit")
            win_rate_unfiltered = round(unfiltered_wins / total * 100, 1)
            win_rate_filtered = (
                round(filtered_wins / len(filtered) * 100, 1) if filtered else win_rate_unfiltered
            )
            return {
                "win_rate_filtered": win_rate_filtered,
                "win_rate_unfiltered": win_rate_unfiltered,
                "closed_signals": total,
                "signals_scored_today": signals_today,
                "insufficient_data": False,
            }

        # No closed signals yet — NEVER fabricate a track record. The UI shows
        # "insufficient data" until real outcomes accumulate.
        return {
            "win_rate_filtered": None,
            "win_rate_unfiltered": None,
            "closed_signals": 0,
            "signals_scored_today": signals_today,
            "insufficient_data": True,
        }
    except Exception as e:
        logger.warning(f"AI performance endpoint failed: {e}")
        return {
            "win_rate_filtered": None,
            "win_rate_unfiltered": None,
            "closed_signals": 0,
            "signals_scored_today": 0,
            "insufficient_data": True,
        }


@router.get("/api/market/ohlc/{symbol}")
async def get_market_ohlc(
    symbol: str,
    request: Request,
    interval: str = Query("1d", description="Data interval: 1d, 1h, 1wk"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days of data"),
):
    """Get historical OHLCV data for a symbol.

    Path-A: raw NSE OHLC requires an EOD data licence or the user's own
    connected broker. A broker-entitled user is served from THEIR broker feed;
    otherwise honest-empty + broker-lock marker.
    """
    ent, uid = entitlement_and_user(request, DataClass.OHLC)
    if not ent.allowed:
        return entitlement_marker(ent, {"symbol": symbol, "interval": interval, "data": []})

    # Bring-your-own-broker: source charts from the user's own broker feed.
    if ent.source == "broker" and uid:
        from ..services.user_broker_data import historical as _user_hist

        user_h = await asyncio.to_thread(_user_hist, uid, symbol, interval, days)
        if user_h is not None:
            return user_h
        return broker_lock(DataClass.OHLC, {"symbol": symbol, "interval": interval, "data": []})

    try:
        from ..data.market import get_market_data_provider
        provider = get_market_data_provider()

        if days <= 5:
            period = "5d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        else:
            period = "1y"

        df = await asyncio.to_thread(provider.get_historical, symbol, period, interval)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Data not found")

        data = []
        for idx, row in df.iterrows():
            data.append({
                "timestamp": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            })

        return {"symbol": symbol, "interval": interval, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OHLC error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
