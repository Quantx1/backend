"""
================================================================================
WATCHLIST LIVE ROUTES — enriched per-symbol engine snapshots (PR 39)
================================================================================
The base CRUD (``GET/POST/DELETE /api/watchlist``) stays in app.py.
This router adds the *alive* view: every watchlist symbol joined with:

    - Live quote (MarketData.get_quote)
    - Dossier consensus  (Forecast / Alpha / Mood roll-up)
    - Regime warning   (flag when current regime conflicts with a
                          recent LONG signal or vice versa)
    - Upcoming earnings  (EarningsScout, next 14 days)
    - Latest signal      (if one fired in last 7 days)

Tier gate:
    Free → max 5 symbols returned (enforced by watchlist_basic feature)
    Pro+ → unlimited

Routes:
    GET /api/watchlist/live       — enriched snapshot for the user
    GET /api/watchlist/limits     — current tier + symbol count + cap
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ..core.database import get_supabase_admin
from ..core.tiers import Tier, UserTier, tier_rank
from ..middleware.tier_gate import current_user_tier
from ..data.market_calendar import IST

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/watchlist", tags=["watchlist"])

FREE_SYMBOL_CAP = 5


# ============================================================================
# Pydantic
# ============================================================================


class EngineSummary(BaseModel):
    consensus: str                      # bullish | bearish | mixed | neutral
    swing_direction: Optional[str]      # from Forecast quantile p50
    regime: Optional[str]               # bull/sideways/bear (shared)
    regime_warning: bool                # flag mismatch
    sentiment_score: Optional[float]    # Mood mean in [-1,1]


class WatchItem(BaseModel):
    symbol: str
    added_at: Optional[str]
    alert_enabled: bool = False
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None
    notes: Optional[str] = None

    # Live market
    last_price: Optional[float] = None
    change_pct: Optional[float] = None

    # Engines
    engines: Optional[EngineSummary] = None

    # Latest signal (if any in last 7d)
    latest_signal: Optional[Dict[str, Any]] = None

    # Upcoming earnings (if any in next 14d)
    upcoming_earnings: Optional[Dict[str, Any]] = None


class WatchlistLive(BaseModel):
    items: List[WatchItem]
    tier: str
    cap: Optional[int]              # None = unlimited
    count: int
    capped: bool                    # True if Free user's list was truncated


class WatchLimits(BaseModel):
    tier: str
    cap: Optional[int]
    used: int
    remaining: Optional[int]


# ============================================================================
# Helpers
# ============================================================================


def _canon(sym: str) -> str:
    s = (sym or "").upper().strip()
    return s[:-3] if s.endswith(".NS") else s


def _cap_for(user: UserTier) -> Optional[int]:
    if user.is_admin:
        return None
    if tier_rank(user.tier) >= tier_rank(Tier.PRO):
        return None
    return FREE_SYMBOL_CAP


async def _quote_safe(symbol: str) -> Dict[str, Any]:
    """Best-effort live quote. Never raises."""
    def _fetch():
        try:
            from ..data.market import MarketData
            q = MarketData().get_quote(symbol)
            if not q:
                return {}
            return {
                "ltp": float(q.ltp) if q.ltp else None,
                "close": float(q.close) if q.close else None,
            }
        except Exception as exc:
            logger.debug("quote lookup failed %s: %s", symbol, exc)
            return {}
    return await asyncio.to_thread(_fetch)


def _latest_regime(sb) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("regime_history")
            .select("regime, prob_bull, prob_sideways, prob_bear, vix")
            .order("detected_at", desc=True)
            .limit(1)
            .execute()
        )
        return (rows.data or [None])[0] or {}
    except Exception:
        return {}


def _latest_signal(sb, symbol: str, since_iso: str) -> Dict[str, Any]:
    try:
        rows = (
            sb.table("signals")
            .select(
                "id, symbol, direction, entry_price, target, stop_loss, "
                "confidence, status, created_at, tft_p50"
            )
            .eq("symbol", symbol)
            .gte("created_at", since_iso)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        row = (rows.data or [None])[0]
        if not row:
            return {}
        return {
            "id": row.get("id"),
            "direction": row.get("direction"),
            "confidence": row.get("confidence"),
            "status": row.get("status"),
            "created_at": row.get("created_at"),
            "entry_price": row.get("entry_price"),
            "target": row.get("target"),
            "stop_loss": row.get("stop_loss"),
            "swing_p50": row.get("tft_p50"),
        }
    except Exception as exc:
        logger.debug("latest signal lookup failed %s: %s", symbol, exc)
        return {}


def _sentiment(sb, symbol: str) -> Optional[float]:
    try:
        cutoff = (date.today() - timedelta(days=14)).isoformat()
        rows = (
            sb.table("news_sentiment")
            .select("mean_score, headline_count")
            .eq("symbol", symbol)
            .gte("trade_date", cutoff)
            .execute()
        )
        data = rows.data or []
        if not data:
            return None
        total = sum(int(r.get("headline_count") or 0) for r in data)
        if total <= 0:
            return None
        weighted = sum(
            float(r.get("mean_score") or 0) * int(r.get("headline_count") or 0)
            for r in data
        ) / total
        return round(max(-1.0, min(1.0, weighted)), 3)
    except Exception:
        return None


def _upcoming_earnings(sb, symbol: str) -> Dict[str, Any]:
    try:
        today = date.today().isoformat()
        horizon = (date.today() + timedelta(days=14)).isoformat()
        rows = (
            sb.table("earnings_predictions")
            .select("announce_date, beat_prob, confidence")
            .eq("symbol", symbol)
            .gte("announce_date", today)
            .lte("announce_date", horizon)
            .order("announce_date")
            .limit(1)
            .execute()
        )
        return (rows.data or [None])[0] or {}
    except Exception:
        return {}


def _swing_direction(p50: Optional[float], entry: Optional[float]) -> Optional[str]:
    if p50 is None:
        return None
    try:
        p50_f = float(p50)
    except (TypeError, ValueError):
        return None
    if entry:
        try:
            e = float(entry)
            if p50_f > e * 1.005:
                return "bullish"
            if p50_f < e * 0.995:
                return "bearish"
            return "neutral"
        except (TypeError, ValueError):
            pass
    # No entry reference — fall back to raw sign.
    if p50_f > 0.005:
        return "bullish"
    if p50_f < -0.005:
        return "bearish"
    return "neutral"


def _regime_warning(regime: Optional[str], sig_direction: Optional[str]) -> bool:
    """Flag when open signal direction conflicts with current regime."""
    if not regime or not sig_direction:
        return False
    if regime == "bear" and sig_direction == "LONG":
        return True
    if regime == "bull" and sig_direction == "SHORT":
        return True
    return False


def _consensus(
    swing: Optional[str],
    sentiment: Optional[float],
    regime: Optional[str],
) -> str:
    votes: List[str] = []
    if swing in {"bullish", "bearish"}:
        votes.append(swing)
    if sentiment is not None:
        if sentiment > 0.15:
            votes.append("bullish")
        elif sentiment < -0.15:
            votes.append("bearish")
    if regime in {"bull", "bear"}:
        votes.append("bullish" if regime == "bull" else "bearish")
    if not votes:
        return "neutral"
    bulls = votes.count("bullish")
    bears = votes.count("bearish")
    if bulls == bears:
        return "mixed"
    return "bullish" if bulls > bears else "bearish"


# ============================================================================
# Routes
# ============================================================================


@router.get("/limits", response_model=WatchLimits)
async def get_limits(user: UserTier = Depends(current_user_tier)) -> WatchLimits:
    sb = get_supabase_admin()
    try:
        rows = (
            sb.table("watchlist")
            .select("id", count="exact")
            .eq("user_id", user.user_id)
            .execute()
        )
        used = int(getattr(rows, "count", 0) or 0)
    except Exception:
        used = 0
    cap = _cap_for(user)
    remaining = None if cap is None else max(0, cap - used)
    return WatchLimits(tier=user.tier.value, cap=cap, used=used, remaining=remaining)


@router.get("/live", response_model=WatchlistLive)
async def get_live(user: UserTier = Depends(current_user_tier)) -> WatchlistLive:
    """Enriched watchlist snapshot — joins every symbol with its latest
    engine reads, live quote, upcoming earnings, and open-signal row."""
    sb = get_supabase_admin()
    cap = _cap_for(user)

    # 1. Pull raw watchlist rows.
    try:
        rows = (
            sb.table("watchlist")
            .select("symbol, alert_enabled, alert_price_above, alert_price_below, notes, added_at")
            .eq("user_id", user.user_id)
            .order("added_at", desc=False)
            .execute()
        )
        data = rows.data or []
    except Exception as exc:
        logger.error("watchlist pull failed %s: %s", user.user_id, exc)
        data = []

    full_count = len(data)
    capped = False
    if cap is not None and full_count > cap:
        data = data[:cap]
        capped = True

    # 2. Shared context.
    regime_row = _latest_regime(sb)
    regime = regime_row.get("regime")
    since_iso = (datetime.now(IST) - timedelta(days=7)).astimezone(timezone.utc).isoformat()

    # 3. Fan-out per symbol (I/O bound — light threading).
    async def enrich(raw: Dict[str, Any]) -> WatchItem:
        sym_raw = raw.get("symbol") or ""
        sym = _canon(sym_raw)
        quote = await _quote_safe(sym)
        ltp = quote.get("ltp")
        close = quote.get("close")
        change_pct = None
        if ltp and close and close > 0:
            change_pct = round(((ltp - close) / close) * 100, 2)

        signal = _latest_signal(sb, sym, since_iso)
        sentiment = _sentiment(sb, sym)
        earnings = _upcoming_earnings(sb, sym)

        swing_dir = _swing_direction(signal.get("swing_p50"), signal.get("entry_price"))
        consensus = _consensus(swing_dir, sentiment, regime)
        warning = _regime_warning(regime, signal.get("direction"))

        return WatchItem(
            symbol=sym,
            added_at=str(raw.get("added_at")) if raw.get("added_at") else None,
            alert_enabled=bool(raw.get("alert_enabled", False)),
            alert_price_above=raw.get("alert_price_above"),
            alert_price_below=raw.get("alert_price_below"),
            notes=raw.get("notes"),
            last_price=ltp,
            change_pct=change_pct,
            engines=EngineSummary(
                consensus=consensus,
                swing_direction=swing_dir,
                regime=regime,
                regime_warning=warning,
                sentiment_score=sentiment,
            ),
            latest_signal=signal or None,
            upcoming_earnings=earnings or None,
        )

    items: List[WatchItem]
    if data:
        items = await asyncio.gather(*[enrich(r) for r in data])
    else:
        items = []

    return WatchlistLive(
        items=items,
        tier=user.tier.value,
        cap=cap,
        count=full_count,
        capped=capped,
    )


@router.get("/digest")
async def get_digest(
    use_llm: bool = Query(False, description="Include the grounded daily narrative (cached per user/day)"),
    user: UserTier = Depends(current_user_tier),
):
    """Watchlist Daily Digest — deterministic per-symbol bullets + a 'what
    changed today' summary ALWAYS (0 tokens); the grounded narrative only when
    use_llm, metered by the per-tier `watchlist_digest` cap. A capped request
    degrades to deterministic-only (`llm_capped: true`) instead of 402-ing.
    Admins bypass. Honest-empty when the watchlist is empty."""
    llm_capped = False
    if use_llm and not user.is_admin:
        from ..middleware.llm_caps import get_llm_feature_limiter
        try:
            allowed, _used, _cap = get_llm_feature_limiter().consume(
                user.user_id, "watchlist_digest", user.tier)
        except Exception as exc:  # noqa: BLE001 — fail-open, mirrors enforce_llm_cap
            logger.debug("watchlist_digest cap consume skipped (%s)", exc)
            allowed = True
        if not allowed:
            use_llm, llm_capped = False, True
    from ..services.news.watchlist_digest import digest
    res = await asyncio.to_thread(digest, user.user_id, use_llm=use_llm, cap=_cap_for(user))
    return {"success": True, "llm_capped": llm_capped, **res}


__all__ = ["router"]
