"""
================================================================================
QUANT X SCREENER API ROUTES
================================================================================
Complete API for Quant X Screener with 50+ scanners, AI predictions,
ML signals, trend forecasting, and full NSE/BSE coverage.
================================================================================
"""

import httpx as _httpx_for_tv
import asyncio
import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Query, HTTPException, Path, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..data.screener.engine import get_live_screener, SCANNER_MENU
from ..middleware.tier_gate import RequireFeature, current_user_tier
from ..core.tiers import UserTier
from ..core.security import get_current_user
from ..services.entitlement import DataClass, entitlement_for, entitlement_marker

# ── Scanner-result cache (PR-S1.3) ────────────────────────────────────
# `screener.run_scanner(N, …)` evaluates the universe top-to-bottom and
# takes 45+ seconds on a cold call. Trader behaviour is to refresh the
# same scanner repeatedly — cache the result for 60 s so the second
# refresh is sub-100 ms. Keyed on (scanner_id, universe, lookback).
import time as _time
_SCANNER_CACHE_TTL = 60.0
_scanner_cache: Dict[str, tuple] = {}  # key -> (timestamp, payload)


async def _scanner_cached(
    scanner_id: int, universe: str = "N", lookback: str = "12",
) -> Any:
    """Run a scanner with a 60s in-process TTL cache.

    Uses asyncio.Lock per-key (built lazily) so a slow first call
    coalesces N concurrent refreshers into a single backend hit.
    Failures bypass the cache so a transient error doesn't get stuck
    for 60 s.
    """
    key = f"{scanner_id}:{universe}:{lookback}"
    now = _time.monotonic()
    hit = _scanner_cache.get(key)
    if hit and now - hit[0] < _SCANNER_CACHE_TTL:
        return hit[1]
    screener = get_live_screener()
    result = await screener.run_scanner(scanner_id, universe, lookback)
    _scanner_cache[key] = (now, result)
    # Bound the cache so a hostile caller can't blow memory.
    if len(_scanner_cache) > 256:
        oldest = min(_scanner_cache.items(), key=lambda kv: kv[1][0])[0]
        _scanner_cache.pop(oldest, None)
    return result

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/screener", tags=["QuantScan AI"])
quantai_router = APIRouter(
    prefix="/api/quantai",
    tags=["AI Stock Ranker picks"],
    dependencies=[Depends(RequireFeature("scanner_lab"))],  # Pro+; was ungated
)

# ============================================================================
# RESPONSE MODELS
# ============================================================================


class ScannerInfo(BaseModel):
    id: int
    name: str
    description: str
    premium: bool = False


class ScanResult(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    ltp: float
    change_pct: float
    volume: str
    rsi: int
    trend: str
    pattern: str
    signal: str
    ma_signal: Optional[str] = None
    breakout_level: Optional[float] = None
    support_level: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    stop_loss: Optional[float] = None

# ============================================================================
# SCANNER ENDPOINTS
# ============================================================================


@router.get("/info")
async def get_screener_info():
    """Get Quant X Screener capabilities and scanner inventory."""
    screener = get_live_screener()

    return {
        "name": "Quant X Screener",
        "version": "3.0",
        "description": "AI-powered market scanner for NSE/BSE with 50+ scanners and 6 ML models",
        "features": screener.get_all_scanners(),
        "status": "active",
        "data_source": screener._data_source,
    }


@router.get("/scanners")
async def get_all_scanners():
    """
    Get all available scanner categories and individual scanners

    Returns 50+ scanners organized by category with full descriptions
    """
    screener = get_live_screener()
    scanner_data = screener.get_all_scanners()

    return {
        "success": True,
        "total_scanners": scanner_data["total_scanners"],
        "stock_universe": scanner_data["stock_universe"],
        "categories": scanner_data["categories"],
        "ai_ml_features": scanner_data["ai_ml_features"],
        "exchanges": SCANNER_MENU["exchanges"],
    }


@router.get("/menu")
async def get_screener_menu():
    """
    Get screener menu definitions for frontend UI
    """
    return {
        "exchanges": SCANNER_MENU["exchanges"],
        "scan_types": SCANNER_MENU["scan_types"],
    }


@router.get("/scanners/all")
async def get_all_scanner_details():
    """
    Get detailed information for ALL 50+ scanners
    """
    get_live_screener()
    scanner_details = SCANNER_MENU["scan_types"]["X"]["submenu"]

    scanners = []
    for scanner_id, info in scanner_details.items():
        scanners.append({
            "id": scanner_id,
            "name": info["name"],
            "description": info["description"],
            "premium": scanner_id >= 30,  # Advanced scanners are premium
        })

    return {
        "success": True,
        "count": len(scanners),
        "scanners": scanners,
    }


@router.get("/scan/{scanner_id}")
async def run_scanner(
    scanner_id: int = Path(
        ...,
        ge=0,
        le=88,
        description="Scanner ID (0-88; PR-S9 52-61, PR-S17 62-71 bearish, PR-S18 72-86 institutional, PR-S20 87-88 F&O stock)"),
    exchange: str = Query("N", description="Exchange: N=NSE, B=BSE, S=Nifty50, etc."),
    index: str = Query("12", description="Index: 12=Nifty500, 0=Full, etc."),
    # Phase 1.7 audit fix #1.9 — Scanner Lab is Pro-tier per Step 1 §C7.
    # The legacy route was wide-open: any anonymous request could hit
    # /api/screener/scan/{id} and consume the compute. Now we gate via
    # the canonical "scanner_lab" feature key in FEATURE_MATRIX.
    user: UserTier = Depends(RequireFeature("scanner_lab")),
):
    """
    Run a specific scanner by ID

    Scanner IDs:
    - 0: Full Screening (all patterns)
    - 1: Breakout from Consolidation
    - 2: Top Gainers (>2%)
    - 3: Top Losers (>2%)
    - 4: Volume Breakout
    - 5: 52-Week High
    - 6: 10-Day High
    - 7: 52-Week Low
    - 8: Volume Surge (>2.5x)
    - 9: RSI Oversold (<30)
    - 10: RSI Overbought (>70)
    - 11-15: MA Crossover Strategies
    - 16-25: Advanced Patterns (VCP, Cup&Handle, etc.)
    - 26-35: Momentum & Trend
    - 36-42: Smart Money & F&O Analysis
    """
    scanner_info = SCANNER_MENU["scan_types"]["X"]["submenu"].get(scanner_id)

    if not scanner_info:
        raise HTTPException(status_code=404, detail=f"Scanner {scanner_id} not found")

    return await _scanner_cached(scanner_id, exchange, index)


@router.get("/scan/category/{category}")
async def run_category_scan(
    category: str = Path(..., description="Category: breakout, momentum, volume, reversal, patterns, etc."),
    exchange: str = Query("N", description="Exchange code"),
    user: UserTier = Depends(RequireFeature("scanner_lab")),
):
    """
    Run all scanners in a category and combine results
    """
    category_map = {
        "breakout": [0, 1, 4, 5, 6, 7, 20, 33],
        "momentum": [2, 3, 10, 17, 26, 30, 31],
        "volume": [4, 8, 37, 38, 39],
        "reversal": [9, 12, 19, 24, 25, 28],
        "patterns": [12, 13, 14, 21, 22, 23, 24, 25],
        "ma_strategies": [11, 15, 26, 27, 32],
        "smart_money": [36, 37, 38, 39, 40],
        "fo_analysis": [40, 41, 42, 36],
    }

    if category not in category_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Available: {list(category_map.keys())}"
        )

    scanner_ids = category_map[category]

    all_results = []
    for scanner_id in scanner_ids[:3]:  # Limit to 3 scanners per request
        result = await _scanner_cached(scanner_id, exchange, "12")
        if result.get("results"):
            all_results.extend(result["results"][:10])

    # Deduplicate by symbol
    seen = set()
    unique_results = []
    for r in all_results:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            unique_results.append(r)

    return {
        "success": True,
        "category": category,
        "scanners_used": scanner_ids,
        "timestamp": datetime.now().isoformat(),
        "results": unique_results[:30],
        "count": len(unique_results[:30]),
    }


# ============================================================================
# AI/ML PREDICTION ENDPOINTS
# ============================================================================

@router.get("/ai/nifty-prediction")
async def get_nifty_prediction():
    """
    Get AI-powered Nifty prediction.

    Combines:
    - **Regime**: 3-state market regime (bull / sideways / bear)
    - **Gate**: 3-class signal verdict (BUY / SELL / HOLD) with probabilities
    - **Technical indicators**: RSI, MACD, SMA, Bollinger Bands, ATR

    Returns prediction with confidence levels, regime, and support/resistance.
    """
    screener = get_live_screener()
    prediction = await screener.get_nifty_prediction()

    # No-fallbacks contract (locked 2026-04-19): if the engines can't
    # produce a real prediction we surface a 503 instead of substituting
    # a heuristic SMA/RSI computation. Heuristic stand-ins look like AI
    # output to users, which is exactly the trust failure we're avoiding.
    if not prediction or prediction.get("error"):
        detail = (
            (prediction or {}).get("error")
            or "AI Nifty prediction temporarily unavailable"
        )
        raise HTTPException(status_code=503, detail=detail)

    return {
        "success": True,
        "feature": "AI Nifty Prediction",
        "source": "Quant X engines (Regime + Gate)",
        "data": prediction,
    }


@router.get("/ai/trend-forecast/{symbol}")
async def get_trend_forecast(
    symbol: str = Path(..., description="Stock symbol (e.g., RELIANCE, TCS, NIFTY)"),
):
    """
    Get multi-timeframe trend forecast for a stock

    Returns:
    - Intraday trend
    - Short-term trend (1-2 weeks)
    - Medium-term trend (1-3 months)
    - Technical indicators
    - Pattern detection
    """
    screener = get_live_screener()
    forecast = await screener.get_trend_forecast(symbol.upper())

    return {
        "success": True,
        "feature": "Trend Forecasting",
        "data": forecast,
    }


@router.get("/ai/ml-signals")
async def get_ml_signals(
    limit: int = Query(20, ge=5, le=50, description="Number of signals to return"),
):
    """
    Get today's AI ensemble trading signals.

    Reads from the ``signals`` table where the model-first SignalGenerator
    pipeline writes its output (LGBM + TFT + Qlib + FinBERT + HMM voter
    consensus). Returns at most ``limit`` BUY signals ranked by ensemble
    confidence.

    No-fallbacks contract: if no signals qualified today, returns an
    empty list — never a heuristic momentum-scanner approximation.
    """
    from ..core.database import get_supabase_admin

    today = date.today().isoformat()
    sb = get_supabase_admin()

    try:
        rows = (
            sb.table("signals")
            .select(
                "symbol, direction, confidence, entry_price, stop_loss, "
                "target_1, risk_reward, model_agreement, regime_at_signal, "
                "is_premium"
            )
            .eq("date", today)
            .eq("direction", "LONG")
            .in_("status", ["active", "triggered"])
            .order("confidence", desc=True)
            .limit(limit)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.error("ml-signals signals-table query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Signals store unavailable")

    signals = [
        {
            "symbol": row["symbol"],
            "name": row["symbol"],
            "signal_type": "BUY",
            "strength": "Strong" if (row.get("confidence") or 0) >= 75 else "Moderate",
            "confidence": round((row.get("confidence") or 0) / 100.0, 4),
            "entry_price": row.get("entry_price"),
            "target": row.get("target_1"),
            "stop_loss": row.get("stop_loss"),
            "risk_reward": row.get("risk_reward"),
            "model_agreement": row.get("model_agreement"),
            "regime_at_signal": row.get("regime_at_signal"),
            "is_premium": row.get("is_premium", False),
        }
        for row in rows
    ]

    return {
        "success": True,
        "feature": "AI ensemble signals",
        "source": "Quant X model-first pipeline",
        "timestamp": datetime.now().isoformat(),
        "signals": signals,
        "count": len(signals),
    }


# ============================================================================
# SPECIAL SCAN ENDPOINTS
# ============================================================================

def _persisted_alpha_picks(limit: int):
    """Most-recent persisted Alpha ranking from the nightly job (alpha_scores).

    Same Qlib Alpha158 model output — just the last computed session — used so
    the discovery leaderboard works even when the live engine isn't loaded.
    Real model output, NOT a heuristic stand-in (no-fallbacks contract intact)."""
    try:
        from ..core.database import get_supabase_admin
        sb = get_supabase_admin()
        latest = (sb.table("alpha_scores").select("trade_date")
                  .order("trade_date", desc=True).limit(1).execute().data or [])
        if not latest:
            return []
        td = latest[0]["trade_date"]
        rows = (sb.table("alpha_scores")
                .select("symbol,qlib_rank,qlib_score_raw,trade_date")
                .eq("trade_date", td).order("qlib_rank").limit(limit).execute().data or [])
        return [{
            "symbol": r["symbol"], "rank": r.get("qlib_rank"),
            "alpha_score": r.get("qlib_score_raw"),
            "trade_date": r.get("trade_date"),
        } for r in rows]
    except Exception as exc:
        logger.debug("persisted alpha picks failed: %s", exc)
        return []


@router.get("/swing-candidates")
async def get_swing_candidates(
    limit: int = Query(20, ge=5, le=50),
):
    """
    Get top swing trading candidates from **Alpha** (cross-sectional
    AI ranker — Qlib Alpha158 v4).

    PR-S1.2 (2026-05-30): rewired from the unbuilt ``quantai_ranker.txt``
    model to PROD ``qlib_alpha158``. The booster + provider dir are
    loaded lazily on first call; the result is cached for 5 minutes via
    the same in-process cache that backs the nightly ranking job.

    No-fallbacks contract: if Qlib isn't initialised (provider dir
    missing, model not trained) we surface 503 — never a heuristic
    momentum-scanner stand-in.
    """
    # 1) Live Qlib engine (freshest). On ANY unavailability, fall through to
    #    the persisted nightly ranking rather than 503.
    try:
        from ..ai.qlib.engine import get_qlib_engine
        engine = get_qlib_engine()
        if not engine.loaded:
            await asyncio.to_thread(engine.load)
        if engine.loaded:
            rows = await asyncio.to_thread(engine.rank_universe, instruments="nse_all")
            if rows:
                picks = [{
                    "symbol": r["symbol"], "rank": r["qlib_rank"],
                    "alpha_score": r["qlib_score_raw"], "trade_date": r["trade_date"],
                } for r in rows[:limit]]
                return {
                    "success": True, "feature": "Alpha Swing Candidates",
                    "source": "qlib_alpha158:v4", "stale": False,
                    "timestamp": datetime.now().isoformat(),
                    "results": picks, "count": len(picks),
                }
    except Exception as exc:
        logger.warning("Qlib live rank unavailable, using persisted alpha_scores: %s", exc)

    # 2) Persisted fallback — same model's most-recent nightly ranking.
    picks = _persisted_alpha_picks(limit)
    if picks:
        return {
            "success": True, "feature": "Alpha Swing Candidates",
            "source": "qlib_alpha158:persisted", "stale": True,
            "as_of": picks[0].get("trade_date"),
            "timestamp": datetime.now().isoformat(),
            "results": picks, "count": len(picks),
        }

    raise HTTPException(
        status_code=503,
        detail="Alpha not ready — no live engine and no persisted ranking",
    )


# ============================================================================
# QUANTAI ALPHA PICKS ENDPOINTS
# ============================================================================

@quantai_router.get("/picks")
async def get_quantai_picks(
    limit: int = Query(15, ge=5, le=50, description="Number of picks to return"),
):
    """AI stock picks — cross-sectional alpha ranking.

    Served by the production Alpha ranker (the same engine behind
    /swing-candidates). The legacy standalone quantai_ranker was retired
    2026-06-17; picks now come from the Alpha engine (no-fallbacks contract
    intact). Kept as a back-compat alias.
    """
    try:
        from ..ai.qlib.engine import get_qlib_engine
        engine = get_qlib_engine()
        if not engine.loaded:
            await asyncio.to_thread(engine.load)
        if engine.loaded:
            rows = await asyncio.to_thread(engine.rank_universe, instruments="nse_all")
            if rows:
                picks = [{
                    "symbol": r["symbol"], "rank": r["qlib_rank"],
                    "alpha_score": r["qlib_score_raw"], "trade_date": r["trade_date"],
                } for r in rows[:limit]]
                return {
                    "success": True, "feature": "AI Stock Ranker picks",
                    "source": "qlib_alpha158:v4", "timestamp": datetime.now().isoformat(),
                    "results": picks, "count": len(picks),
                }
    except Exception as exc:
        logger.warning("quantai picks: live Alpha unavailable, using persisted: %s", exc)

    picks = _persisted_alpha_picks(limit)
    return {
        "success": bool(picks),
        "feature": "AI Stock Ranker picks",
        "source": "qlib_alpha158:persisted",
        "timestamp": datetime.now().isoformat(),
        "results": picks, "count": len(picks),
    }


@quantai_router.get("/status")
async def get_quantai_status():
    """AI picks model status — served by the Alpha (Qlib Alpha158) engine."""
    try:
        from ..ai.qlib.engine import get_qlib_engine
        engine = get_qlib_engine()
        return {"success": True, "model_loaded": engine.loaded, "source": "qlib_alpha158"}
    except Exception as e:
        return {"success": False, "model_loaded": False, "error": str(e)}


# ============================================================================
# DEPRECATED ALIASES (P1-4 consolidation, 2026-05-08)
# ============================================================================
# These named aliases pre-date the canonical /scan/{scanner_id} dispatcher.
# Frontend doesn't call any of them; they're kept functional for any external
# consumer that may still hit the old paths but ``include_in_schema=False``
# hides them from OpenAPI / new-developer discovery. Replace with /scan/{id}
# in any new client. Pinned scanner-id mapping for reference:
#     /breakouts        → /scan/1
#     /momentum         → /scan/17
#     /volume-surge     → /scan/8
#     /vcp              → /patterns/vcp (canonical)
#     /reversals        → /scan/category/reversal (canonical)
#     /institutional    → /scan/category/smart_money (canonical)
#     /bullish-tomorrow → /ai/ml-signals (canonical)
#     /fo/long-buildup  → /scan/41 (universe=F)
#     /fo/short-buildup → /scan/42 (universe=F)
#     /smart-money/fii-dii → /scan/36
# Removal target: 6 months after the OpenAPI hide (~2026-11-08), once
# server-side analytics confirm no third party hits these paths.
# ============================================================================


@router.get("/breakouts", include_in_schema=False)
async def get_breakout_stocks():
    """[DEPRECATED] Use /scan/1. Kept for back-compat."""
    return await _scanner_cached(1, "N", "12")


@router.get("/momentum", include_in_schema=False)
async def get_momentum_stocks():
    """[DEPRECATED] Use /scan/17. Kept for back-compat."""
    return await _scanner_cached(17, "N", "12")


@router.get("/volume-surge", include_in_schema=False)
async def get_volume_surge():
    """[DEPRECATED] Use /scan/8. Kept for back-compat."""
    return await _scanner_cached(8, "N", "12")


# ════════════════════════════════════════════════════════════════════
# PR-S5 — Pattern Scanner v2 (chart-pattern algo + ML + regime + volume)
# ════════════════════════════════════════════════════════════════════
# The /patterns/v2 endpoints below use our 2,166-line chart pattern
# algorithm (ml/features/patterns.py) gated by BreakoutMetaLabeler RF
# probability, regime alignment, and volume confirmation. Returns
# trade-ready matches with composite score, ML probability, and derived
# entry/stop/target — what /patterns/{type} used to fake.


# PR-S5 v2-scan cache — keyed by (universe, limit, direction). 2-min TTL
# since detection is on daily bars; revalidating per session feels live
# enough without burning a full scan each click.
_v2_scan_cache: Dict[str, tuple] = {}
_V2_SCAN_TTL = 120.0


_TIMEFRAME_FETCH_MAP = {
    # timeframe → (yfinance period, yfinance interval). 60d is the yfinance
    # max for sub-daily intervals; 1y for daily gives ~250 bars (>2× the
    # 100 bars MIN_BARS_REQUIRED). 4h/5m intentionally absent — 4h needs
    # downsampling we haven't built; 5m gives too noisy a signal for the
    # rule engine's swing-trade horizon.
    "1d": ("1y", "1d"),
    "1h": ("60d", "1h"),
    "15m": ("60d", "15m"),
}


@router.get("/patterns/v2/scan")
async def patterns_v2_scan(
    universe: str = Query("nifty50", description="nifty50|nifty100|nifty500|nse_all"),
    timeframe: str = Query("1d", description="1d|1h|15m"),
    limit: int = Query(30, ge=5, le=100),
    direction: Optional[str] = Query(None, description="bullish|bearish (filter)"),
    min_quality: float = Query(0.50, ge=0.0, le=1.0),
    min_ml: float = Query(0.55, ge=0.0, le=1.0),
):
    """Scan a universe and return ranked chart-pattern matches.

    PR-S5 (2026-05-30): replaces the old /patterns/{type} flow with a
    trader-grade pipeline:
      1. Detect via our scan_all_patterns() rule engine
      2. Score with BreakoutMetaLabeler RandomForest (500 trees)
      3. Filter to patterns with quality + ML + volume + regime alignment
      4. Return ranked composite score + entry/stop/target levels

    Public — same response shape for every tier. PRO+ may get the
    deep-dive endpoint (/patterns/v2/explain/{symbol}) later in PR-S3.
    """
    from ..services.chart_patterns import scan_universe as _scan
    from ..ai.qlib.data_handler import load_universe
    from ..data.market import get_market_data_provider

    tf = (timeframe or "1d").lower()
    if tf not in _TIMEFRAME_FETCH_MAP:
        raise HTTPException(400, f"unsupported timeframe: {timeframe}. Use 1d, 1h, or 15m.")
    period_str, interval_str = _TIMEFRAME_FETCH_MAP[tf]

    # 2-min cache — keyed by (universe, timeframe, limit, direction).
    cache_key = f"{universe}:{tf}:{limit}:{direction or 'all'}:{min_quality}:{min_ml}"
    cached = _v2_scan_cache.get(cache_key)
    if cached and _time.monotonic() - cached[0] < _V2_SCAN_TTL:
        return cached[1]

    syms = load_universe(universe)
    if not syms:
        raise HTTPException(400, f"unknown universe: {universe}")
    # Cap to first 60 symbols to keep latency under 30s on a cold call.
    # PR-S2 widens this to the full universe with streaming.
    syms = syms[:60]

    mp = get_market_data_provider()

    def _fetch(sym: str):
        try:
            df = mp.get_historical(sym, period=period_str, interval=interval_str)
            if df is not None and not df.empty:
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception:
            return None

    # Pull current regime so the scanner can drop direction-mismatches
    regime: Optional[str] = None
    try:
        from ..services.regime.resolver import resolve_regime_at
        from ..core.database import get_supabase_admin
        sb = get_supabase_admin()
        regime_row = resolve_regime_at(sb, date.today())
        regime = (regime_row or {}).get("regime")
    except Exception as e:
        logger.debug("regime lookup failed (scanner runs unfiltered): %s", e)

    matches = await asyncio.to_thread(
        _scan, syms,
        bars_fetcher=_fetch, regime=regime,
        max_workers=6, limit=limit,
    )

    out = [m.to_dict() for m in matches]
    if direction:
        out = [m for m in out if m["direction"] == direction.lower()]

    response = {
        "success": True,
        "feature": "Chart Pattern Scanner v2",
        "source": "patterns_v2_gated",
        "regime": regime,
        "universe": universe,
        "timeframe": tf,
        "symbols_scanned": len(syms),
        "min_quality": min_quality,
        "min_ml": min_ml,
        "timestamp": datetime.now().isoformat(),
        "matches": out,
        "count": len(out),
    }
    _v2_scan_cache[cache_key] = (_time.monotonic(), response)
    # Bound cache
    if len(_v2_scan_cache) > 64:
        oldest = min(_v2_scan_cache.items(), key=lambda kv: kv[1][0])[0]
        _v2_scan_cache.pop(oldest, None)
    return response


@router.get("/patterns/v2/explain/{symbol}")
async def patterns_v2_explain(
    symbol: str = Path(..., description="NSE tradingsymbol, e.g. RELIANCE"),
    use_llm: bool = Query(True, description="Use the LLM for AI thesis (slower)"),
):
    """Deep-dive explanation for one symbol's chart pattern (PR-S3).

    Returns: rule values that fired (why_matched), AI thesis (LLM-
    composed factual paragraph — never recommendation language), suggested
    entry/stop/target with derivation basis, regime context, ML probability.
    """
    from ..services.chart_patterns import explain_symbol as _explain
    from ..data.market import get_market_data_provider

    sym = symbol.upper().strip()
    mp = get_market_data_provider()

    try:
        bars = mp.get_historical(sym, period="1y", interval="1d")
    except Exception as e:
        raise HTTPException(500, f"data fetch failed: {e}")
    if bars is None or bars.empty:
        raise HTTPException(404, f"no bars for {sym}")
    bars = bars.copy()
    bars.columns = [c.lower() for c in bars.columns]

    # Resolve regime
    regime: Optional[str] = None
    try:
        from ..services.regime.resolver import resolve_regime_at
        from ..core.database import get_supabase_admin
        sb = get_supabase_admin()
        regime_row = resolve_regime_at(sb, date.today())
        regime = (regime_row or {}).get("regime")
    except Exception:
        pass

    result = await asyncio.to_thread(
        _explain, sym, bars, regime=regime, use_llm=use_llm,
    )
    if result is None:
        raise HTTPException(
            404, f"no usable pattern found for {sym}",
        )
    return result.to_dict()


@router.get("/news/scan")
async def news_scan(
    universe: str = Query("nifty50", description="nifty50|nifty100|nifty500|nse_all"),
    sectors: Optional[str] = Query(None, description="Comma-separated canonical sectors"),
    lookback_days: int = Query(1, ge=1, le=5),
    limit: int = Query(30, ge=5, le=100),
    max_symbols: int = Query(60, ge=10, le=300, description="Per-call symbol cap (news scan is slow)"),
):
    """News-driven scanner (PR-S4) — find stocks with material news today.

    Pulls Google News headlines for each symbol, scores sentiment via
    the LLM, compares to today's price move, and surfaces 6 setup tags
    (underreaction / continuation / divergence × +/-).

    PR-S2.1 (2026-05-31): accepts `nse_all` + optional sector pre-filter
    so users can scan e.g. "Banking sector with material news today"
    without paying for a full 2,136-symbol fetch.

    Strictly descriptive — no recommendation language.
    """
    from ..services.chart_patterns.news_scanner import scan_news_universe
    from ..services.chart_patterns import filter_by_sector
    from ..ai.qlib.data_handler import load_universe

    syms = load_universe(universe)
    if not syms:
        raise HTTPException(400, f"unknown universe: {universe}")

    # Sector pre-filter
    if sectors:
        sector_list = [s.strip() for s in sectors.split(",") if s.strip()]
        syms = filter_by_sector(syms, sector_list)
        if not syms:
            raise HTTPException(400, "no symbols match the selected sectors")

    # News scan does RSS + LLM per symbol — cap to keep latency sane.
    # Even nse_all gets capped here; users can sector-filter to reach
    # smallcaps that wouldn't fit in the top 60.
    syms = syms[:max_symbols]

    hits = await scan_news_universe(
        syms, lookback_days=lookback_days, limit=limit,
    )
    return {
        "success": True,
        "feature": "News-Driven Scanner",
        "universe": universe,
        "symbols_scanned": len(syms),
        "lookback_days": lookback_days,
        "timestamp": datetime.now().isoformat(),
        "hits": [h.to_dict() for h in hits],
        "count": len(hits),
    }


# ════════════════════════════════════════════════════════════════════
# PR-S7 — Power Screeners v2 (confluence + deep-dive + news + earnings)
# ════════════════════════════════════════════════════════════════════


async def _computed_data_or_503(screener, timeout: float = 25.0):
    """Load the cached indicator table off-thread with a HARD timeout. A cold or
    slow ``_get_computed_data`` load must fail fast with 503 rather than hang the
    request indefinitely (observed >90s, no response — 2026-06-22). The abandoned
    worker thread keeps running and warms the cache, so a retry succeeds."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(screener._get_computed_data), timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(503, "screener data not ready — try again in a moment")


@router.get("/v2/nl-compile")
async def nl_screen_compile(
    q: str = Query(..., min_length=2, description="Plain-English screen to compile into scanner blocks"),
):
    """Compile-only half of the NL screener: map free text -> scanner blocks
    WITHOUT running the scan. Powers the AI screen generator's editable
    rule-block UI — resolution is rules-first (0 tokens), LLM for nuance,
    cached; no market-data dependency so it never 503s on a cold engine."""
    from ..services.screener_v2.nl_screen import resolve_screen_query, scanner_label

    resolved = resolve_screen_query(q)
    ids = resolved["scanner_ids"]
    return {
        "success": True,
        "query": q,
        "recognized": bool(ids),
        "source": resolved["source"],
        "blocks": [{"id": i, "name": scanner_label(i)} for i in ids],
    }


@router.get("/v2/nl-scan")
async def nl_screen_scan(
    q: str = Query(..., min_length=2, description="Plain-English screen, e.g. 'oversold in an uptrend with rising volume'"),
    min_hits: int = Query(1, ge=1, le=5),
    limit: int = Query(30, ge=5, le=100),
):
    """Natural-language screener (#6/#15): map free text -> scanner IDs (free
    rule fast-path, LLM agent for nuance, cached) -> real confluence scan."""
    from ..services.screener_v2.nl_screen import resolve_screen_query, scanner_label
    from ..services.screener_v2 import confluence_scan
    from ..data.screener.engine import NSE_STOCK_INFO, get_live_screener

    resolved = resolve_screen_query(q)
    ids = resolved["scanner_ids"]
    if not ids:
        return {"success": True, "query": q, "recognized": False,
                "source": resolved["source"], "scanners_used": [], "count": 0, "results": []}

    screener = get_live_screener()
    summary_df, _ = await _computed_data_or_503(screener)
    if summary_df is None or summary_df.empty:
        raise HTTPException(503, "screener data not ready — try again in a moment")

    matches = await asyncio.to_thread(
        confluence_scan, summary_df, scanner_ids=ids, stock_info=NSE_STOCK_INFO,
        min_hits=min(min_hits, len(ids)), limit=limit,
    )
    return {
        "success": True, "query": q, "recognized": True, "source": resolved["source"],
        "scanners_used": [{"id": i, "name": scanner_label(i)} for i in ids],
        "count": len(matches), "results": [m.to_dict() for m in matches],
    }


@router.get("/v2/confluence")
async def power_screeners_confluence(
    scanners: str = Query(
        "1,4,8,17,26",
        description="Comma-separated scanner IDs to combine (default: breakout+vol_breakout+vol_surge+momentum+macd)",
    ),
    min_hits: int = Query(2, ge=1, le=10, description="Min scanners that must match"),
    limit: int = Query(30, ge=5, le=100),
    sectors: Optional[str] = Query(None, description="Comma-separated canonical sectors"),
):
    """Power Screener v2 — confluence scoring (PR-S7).

    Runs N scanner filters in parallel against the cached indicator table,
    aggregates matches by symbol, scores by (hit count × category diversity
    × per-scanner weight + price/volume confirmation). Bearish scanner
    matches drag the score down rather than being silently ignored.

    Returns a ranked list of stocks that fire on multiple independent
    setups simultaneously — the "real" confluence trades.
    """
    from ..services.screener_v2 import confluence_scan
    from ..data.screener.engine import NSE_STOCK_INFO, get_live_screener

    try:
        scanner_ids = [int(s.strip()) for s in scanners.split(",") if s.strip()]
    except ValueError:
        raise HTTPException(400, "scanners must be comma-separated integers")
    if not scanner_ids:
        raise HTTPException(400, "no scanner ids provided")

    # Use the existing screener's cached indicator dataframe
    screener = get_live_screener()
    summary_df, _ = await _computed_data_or_503(screener)
    if summary_df is None or summary_df.empty:
        raise HTTPException(503, "screener data not ready — try again in a moment")

    # Optional sector pre-filter on the dataframe before running scanners
    if sectors:
        wanted = {s.strip() for s in sectors.split(",") if s.strip()}
        sym_in_sector = {
            sym for sym, info in NSE_STOCK_INFO.items() if info.get("sector") in wanted
        }
        if "symbol" in summary_df.columns:
            summary_df = summary_df[summary_df["symbol"].isin(sym_in_sector)]
        if summary_df.empty:
            raise HTTPException(400, "no symbols match the selected sectors")

    matches = await asyncio.to_thread(
        confluence_scan,
        summary_df,
        scanner_ids=scanner_ids,
        stock_info=NSE_STOCK_INFO,
        min_hits=min_hits,
        limit=limit,
    )

    return {
        "success": True,
        "feature": "Confluence Screener v2",
        "scanners_used": scanner_ids,
        "symbols_evaluated": len(summary_df),
        "min_hits": min_hits,
        "timestamp": datetime.now().isoformat(),
        "matches": [m.to_dict() for m in matches],
        "count": len(matches),
    }


@router.get("/v2/explain/{symbol}")
async def power_screeners_explain(
    symbol: str = Path(..., description="NSE tradingsymbol, e.g. RELIANCE"),
    use_llm: bool = Query(True),
    use_news: bool = Query(True),
    use_earnings: bool = Query(True),
):
    """Deep-dive enrichment for one symbol (PR-S7.3).

    Returns: every indicator currently firing + ATR-derived suggested
    levels + sector breadth + news sentiment + earnings nearness +
    LLM-narrated factual thesis. All blocks degrade independently
    — missing news doesn't break the explainer.
    """
    from ..services.screener_v2 import enrich_symbol
    from ..data.screener.engine import NSE_STOCK_INFO, get_live_screener

    sym = symbol.upper().strip()
    screener = get_live_screener()
    summary_df, per_symbol_dfs = await _computed_data_or_503(screener)
    if summary_df is None or summary_df.empty:
        raise HTTPException(503, "screener data not ready — try again in a moment")

    matching = summary_df[summary_df["symbol"] == sym] if "symbol" in summary_df.columns else summary_df.iloc[0:0]
    if matching.empty:
        raise HTTPException(404, f"{sym} not in current scanner universe")

    summary_row = matching.iloc[0]
    per_symbol_df = per_symbol_dfs.get(sym) if per_symbol_dfs else None

    # Current regime
    regime: Optional[str] = None
    try:
        from ..services.regime.resolver import resolve_regime_at
        from ..core.database import get_supabase_admin
        sb = get_supabase_admin()
        rrow = resolve_regime_at(sb, date.today())
        regime = (rrow or {}).get("regime")
    except Exception:
        pass

    result = await enrich_symbol(
        sym, summary_row, per_symbol_df, summary_df, NSE_STOCK_INFO,
        regime=regime, use_llm=use_llm, use_news=use_news, use_earnings=use_earnings,
    )
    return result.to_dict()


# ════════════════════════════════════════════════════════════════════
# PR-S6 — Saved Scans + Alerts
# ════════════════════════════════════════════════════════════════════


class SavedScanCreate(BaseModel):
    name: str
    scanner_ids: List[int]
    universe: str = "nifty500"
    sectors: Optional[List[str]] = None
    min_hits: int = 1
    schedule: str = "hourly"
    notify_channels: List[str] = ["push"]


class SavedScanUpdate(BaseModel):
    name: Optional[str] = None
    scanner_ids: Optional[List[int]] = None
    universe: Optional[str] = None
    sectors: Optional[List[str]] = None
    min_hits: Optional[int] = None
    schedule: Optional[str] = None
    notify_channels: Optional[List[str]] = None
    enabled: Optional[bool] = None


@router.get("/saved-scans")
async def list_saved_scans(user=Depends(get_current_user)):
    """List all saved scans for the current user."""
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    res = (
        sb.table("saved_scans").select("*")
        .eq("user_id", user.id)
        .order("created_at", desc=True)
        .execute()
    )
    return {"scans": res.data or []}


@router.post("/saved-scans", status_code=201)
async def create_saved_scan(payload: SavedScanCreate, user=Depends(get_current_user)):
    """Save a screener configuration for scheduled re-running."""
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    if not (1 <= len(payload.scanner_ids) <= 10):
        raise HTTPException(400, "scanner_ids must have 1-10 entries")
    row = {
        "user_id": user.id,
        "name": payload.name[:120],
        "scanner_ids": payload.scanner_ids,
        "universe": payload.universe,
        "sectors": payload.sectors or [],
        "min_hits": payload.min_hits,
        "schedule": payload.schedule,
        "notify_channels": payload.notify_channels,
    }
    res = sb.table("saved_scans").insert(row).execute()
    if not res.data:
        raise HTTPException(500, "failed to create saved scan")
    return res.data[0]


@router.patch("/saved-scans/{scan_id}")
async def update_saved_scan(
    scan_id: str,
    payload: SavedScanUpdate,
    user=Depends(get_current_user),
):
    """Update enabled/schedule/notify_channels/etc on an existing saved scan."""
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "no fields to update")
    updates["updated_at"] = datetime.now().isoformat()
    res = (
        sb.table("saved_scans").update(updates)
        .eq("id", scan_id).eq("user_id", user.id)
        .execute()
    )
    if not res.data:
        raise HTTPException(404, "saved scan not found")
    return res.data[0]


@router.delete("/saved-scans/{scan_id}")
async def delete_saved_scan(scan_id: str, user=Depends(get_current_user)):
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    sb.table("saved_scans").delete().eq("id", scan_id).eq("user_id", user.id).execute()
    return {"deleted": True, "scan_id": scan_id}


@router.post("/saved-scans/{scan_id}/run")
async def run_saved_scan_now(scan_id: str, user=Depends(get_current_user)):
    """Manually fire a saved scan from the UI. Returns the match diff."""
    from ..core.database import get_supabase_admin
    from ..services.saved_scans import run_saved_scan
    sb = get_supabase_admin()
    res = (
        sb.table("saved_scans").select("*")
        .eq("id", scan_id).eq("user_id", user.id)
        .limit(1).execute()
    )
    if not res.data:
        raise HTTPException(404, "saved scan not found")
    scan = res.data[0]
    result = await run_saved_scan(scan)

    # Persist the run + insert alert if there are new symbols
    now_iso = datetime.now().isoformat()
    sb.table("saved_scans").update({
        "last_run_at": now_iso,
        "last_hit_symbols": result.matched_symbols,
        "last_hit_count": result.total_count,
        "updated_at": now_iso,
    }).eq("id", scan_id).execute()

    if result.new_symbols:
        sb.table("saved_scan_alerts").insert({
            "scan_id": scan_id,
            "user_id": user.id,
            "new_symbols": result.new_symbols,
            "total_match_count": result.total_count,
            "notified": True,
        }).execute()

    return {
        "matched_symbols": result.matched_symbols,
        "new_symbols": result.new_symbols,
        "total_count": result.total_count,
        "error": result.error,
    }


@router.get("/saved-scans/alerts")
async def list_alerts(
    limit: int = Query(30, ge=1, le=100),
    user=Depends(get_current_user),
):
    """Recent alert firings for this user."""
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    res = (
        sb.table("saved_scan_alerts").select("*")
        .eq("user_id", user.id)
        .order("fired_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"alerts": res.data or []}


# ════════════════════════════════════════════════════════════════════
# PR-S10 — Per-scanner historical win-rate stats
# ════════════════════════════════════════════════════════════════════


@router.get("/v2/scanner-stats")
async def power_screeners_stats(
    scanner_id: Optional[int] = Query(None, description="If set, return stats for just this scanner"),
):
    """Per-scanner historical win-rate + average return.

    Reads pre-computed aggregates from `scanner_stats` table. Backfill
    is run by `scripts/data/backfill_scanner_outcomes.py` (manual + nightly).
    Returns empty stats if the backfill hasn't been run yet.
    """
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    q = sb.table("scanner_stats").select("*")
    if scanner_id is not None:
        q = q.eq("scanner_id", scanner_id)
    res = q.execute()
    return {"stats": res.data or []}


# ════════════════════════════════════════════════════════════════════
# PR-S12 — Multi-timeframe agreement scanner
# ════════════════════════════════════════════════════════════════════


@router.get("/v2/mtf-scan")
async def power_screeners_mtf(
    universe: str = Query("nifty100", description="nifty50/100/500"),
    timeframes: str = Query("15m,1h,1d", description="Comma-separated: 15m,1h,1d"),
    direction: Optional[str] = Query(None, description="bullish|bearish filter"),
    limit: int = Query(30, ge=5, le=100),
    sectors: Optional[str] = Query(None),
):
    """Multi-timeframe (MTF) agreement scanner (PR-S12).

    Finds stocks where momentum is aligned across multiple timeframes
    (e.g. bullish on daily AND 1h AND 15m). Much higher conviction than
    single-timeframe filters because intraday + daily confluence rules
    out 2 PM reversal traps.

    Slower than the legacy screener (~5-15s for 100 symbols × 3 tfs)
    since each timeframe needs its own data fetch — capped to nifty100
    by default.
    """
    from ..services.screener_v2 import scan_multi_timeframe
    from ..data.screener.engine import NSE_STOCK_INFO
    from ..ai.qlib.data_handler import load_universe
    from ..services.chart_patterns import filter_by_sector

    syms = load_universe(universe)
    if not syms:
        raise HTTPException(400, f"unknown universe: {universe}")
    if sectors:
        sector_list = [s.strip() for s in sectors.split(",") if s.strip()]
        syms = filter_by_sector(syms, sector_list)
        if not syms:
            raise HTTPException(400, "no symbols match selected sectors")
    # Cap aggressively — MTF is heavy per-symbol
    syms = syms[:60]

    tfs = [t.strip() for t in timeframes.split(",") if t.strip()]
    matches = await asyncio.to_thread(
        scan_multi_timeframe,
        syms, timeframes=tfs, direction=direction,
        stock_info=NSE_STOCK_INFO,
    )
    matches = matches[:limit]

    return {
        "success": True,
        "feature": "Multi-Timeframe Agreement",
        "timeframes": tfs,
        "direction": direction,
        "universe": universe,
        "symbols_scanned": len(syms),
        "matches": [m.to_dict() for m in matches],
        "count": len(matches),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════════════
# PR-S13 — Sector heatmap
# ════════════════════════════════════════════════════════════════════


@router.get("/v2/sector-heatmap")
async def power_screeners_sector_heatmap():
    """Sector × metric grid for the frontend heatmap view.

    Returns one row per canonical sector with:
      avg_change_pct, median_change_pct, breadth_pct (% up),
      volume_surge_pct, rsi_oversold_count, rsi_overbought_count,
      top_movers (top 3 by |change|).

    Cheap — reads the existing cached summary_df. No extra data fetch.
    """
    from ..services.screener_v2 import build_sector_heatmap
    from ..data.screener.engine import NSE_STOCK_INFO, get_live_screener

    screener = get_live_screener()
    summary_df, _ = await _computed_data_or_503(screener)
    if summary_df is None or summary_df.empty:
        raise HTTPException(503, "screener data not ready")

    rows = await asyncio.to_thread(build_sector_heatmap, summary_df, NSE_STOCK_INFO)
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "sectors": [r.to_dict() for r in rows],
        "count": len(rows),
    }


# ════════════════════════════════════════════════════════════════════
# PR-S14 — Comparable historical setups (k-NN)
# ════════════════════════════════════════════════════════════════════


@router.get("/v2/comparable/{scanner_id}/{symbol}")
async def power_screeners_comparable(
    scanner_id: int = Path(..., ge=0, le=88),
    symbol: str = Path(..., description="NSE tradingsymbol"),
    k: int = Query(5, ge=1, le=20),
    since_days: int = Query(180, ge=30, le=730),
):
    """Comparable historical setups (PR-S14) — find the k closest past
    hits of this scanner + return their forward-return distribution.

    Useful for calibrated expectations: instead of "RSI Oversold has
    47% overall WR" you get "47% based on 215 hits over last 180 days,
    median +1.4% in 5d, max DD -2.8%".

    Requires `scanner_outcomes` to be populated (run
    `scripts/data/backfill_scanner_outcomes.py --scanner N` first).
    """
    from ..services.screener_v2 import comparable_setups
    from ..data.screener.engine import get_live_screener

    sym = symbol.upper().strip()
    screener = get_live_screener()
    summary_df, _ = await _computed_data_or_503(screener)
    if summary_df is None or summary_df.empty:
        raise HTTPException(503, "screener data not ready")

    matching = summary_df[summary_df["symbol"] == sym] if "symbol" in summary_df.columns else summary_df.iloc[0:0]
    if matching.empty:
        raise HTTPException(404, f"{sym} not in current scanner universe")

    summary_row = matching.iloc[0]
    result = await asyncio.to_thread(
        comparable_setups,
        scanner_id, sym, summary_row,
        k=k, since_days=since_days,
    )
    return result.to_dict()


# ════════════════════════════════════════════════════════════════════
# PR-S15 — TradingView symbol resolver proxy
# ════════════════════════════════════════════════════════════════════
# TradingView's public symbol_search API requires Origin + Referer
# headers set to tradingview.com — which browsers can't forge. We proxy
# through here, set the headers server-side, return JSON to the client.
# Cached 30 min per (symbol, exchange) since the catalog is stable.


_tv_resolve_cache: Dict[str, tuple] = {}
_TV_RESOLVE_TTL = 1800.0  # 30 min


@router.get("/instruments")
async def search_instruments(
    q: str = Query("", description="Search term (symbol or company name)"),
    sector: str = Query("", description="Filter by sector (substring match)"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
):
    """Fuzzy search the instruments table (full NSE main-board equity universe).

    Case-insensitive match on symbol OR name, optionally narrowed by sector.
    When `q`/`sector` are empty, returns the first `limit` rows ordered by symbol.
    Reads via the service-role client (RLS is REVOKE'd from anon/authenticated).
    """
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    try:
        base = sb.table("instruments").select(
            "symbol,name,sector,mcap_category"
        ).eq("instrument_type", "EQ")
        if q:
            safe_q = q.replace("%", "").replace("_", "")
            base = base.or_(f"symbol.ilike.%{safe_q}%,name.ilike.%{safe_q}%")
        if sector:
            base = base.ilike("sector", f"%{sector.strip()}%")
        if not q and not sector:
            base = base.order("symbol")
        rows = base.limit(limit).execute().data or []
    except Exception as exc:
        logger.debug("instruments search failed: %s", exc)
        rows = []
    return {"success": True, "instruments": rows}


@router.get("/indices")
async def list_indices():
    """List the index catalog (broad-market / sectoral / derivatives) the user
    can browse. Categories come from the curated NSE index map; the constituent
    lists themselves live in `index_constituents` (see /index/{name}/constituents).
    """
    from ..data.reference.nse_reference import INDEX_CSV_MAP, FNO_INDEX_NAME
    out = [{"index_name": name, "category": cat}
           for name, (_fn, cat) in INDEX_CSV_MAP.items()]
    out.append({"index_name": FNO_INDEX_NAME, "category": "derivatives"})
    out.sort(key=lambda x: (x["category"], x["index_name"]))
    return {"success": True, "indices": out}


@router.get("/index/{index_name}/constituents")
async def get_index_constituents(
    index_name: str = Path(..., description="Index, e.g. 'NIFTY 50' or 'NIFTY BANK'"),
    limit: int = Query(800, ge=1, le=1000),
):
    """Constituents of an index, enriched with instrument name / sector / mcap.

    Powers 'browse by index/sector' surfaces. Enrichment is chunked so even the
    widest index (NIFTY TOTAL MARKET, ~755) stays within PostgREST URL limits.
    """
    from ..core.database import get_supabase_admin
    sb = get_supabase_admin()
    name = index_name.strip().upper()
    try:
        mem = (sb.table("index_constituents").select("symbol,industry")
               .eq("index_name", name).limit(limit).execute().data or [])
        syms = [m["symbol"] for m in mem]
        meta = {}
        for i in range(0, len(syms), 150):  # chunked .in_ to bound URL length
            chunk = syms[i:i + 150]
            inst = (sb.table("instruments").select("symbol,name,sector,mcap_category")
                    .eq("instrument_type", "EQ").in_("symbol", chunk).execute().data or [])
            for row in inst:
                meta[row["symbol"]] = row
        out = []
        for m in mem:
            i = meta.get(m["symbol"], {})
            out.append({
                "symbol": m["symbol"],
                "name": i.get("name"),
                "sector": i.get("sector") or m.get("industry"),
                "mcap_category": i.get("mcap_category"),
            })
        out.sort(key=lambda x: x["symbol"])
    except Exception as exc:
        logger.debug("index constituents failed for %s: %s", name, exc)
        out = []
    return {"success": True, "index_name": name, "count": len(out), "constituents": out}


@router.get("/tv-resolve")
async def tv_resolve(
    symbol: str = Query(..., description="Symbol to resolve (e.g. RELIANCE, M&M, 63MOONS)"),
    exchange: str = Query("NSE", description="NSE or BSE"),
):
    """Validate that a symbol exists in TradingView's catalog.

    Frontend cannot call symbol-search.tradingview.com directly because
    the endpoint requires Origin/Referer = tradingview.com which browsers
    can't forge. This proxy sets the headers server-side, returns the
    first matching hit (or null if not found).

    Used by TradingViewWidget BEFORE mounting the chart so the user
    sees a clear "not in catalog" badge instead of a blank chart for
    smallcaps TradingView doesn't index.
    """
    sym = symbol.strip()
    if not sym:
        return {"symbol": None}
    exch = exchange.upper().strip()
    if exch not in ("NSE", "BSE"):
        raise HTTPException(400, "exchange must be NSE or BSE")

    key = f"{exch}:{sym}"
    now = _time.monotonic()
    hit = _tv_resolve_cache.get(key)
    if hit and now - hit[0] < _TV_RESOLVE_TTL:
        return hit[1]

    try:
        async with _httpx_for_tv.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://symbol-search.tradingview.com/symbol_search/",
                params={
                    "text": sym, "exchange": exch,
                    "hl": "0", "lang": "en", "type": "stock",
                },
                headers={
                    "Origin": "https://www.tradingview.com",
                    "Referer": "https://www.tradingview.com/",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                },
            )
            items = resp.json() if resp.status_code == 200 else []
    except Exception as e:
        logger.debug("tv_resolve failed: %s", e)
        items = []

    # Prefer exact-symbol match, else first hit, else None
    resolved: Optional[str] = None
    want = sym.upper()
    if isinstance(items, list) and items:
        exact = next(
            (h for h in items if (h.get("symbol") or "").upper() == want),
            None,
        )
        winner = exact or items[0]
        ex = winner.get("exchange") or exch
        sy = winner.get("symbol") or sym
        resolved = f"{ex}:{sy}"

    result = {"symbol": resolved, "input": f"{exch}:{sym}"}
    _tv_resolve_cache[key] = (now, result)
    if len(_tv_resolve_cache) > 2000:
        oldest = min(_tv_resolve_cache.items(), key=lambda kv: kv[1][0])[0]
        _tv_resolve_cache.pop(oldest, None)
    return result


# ────────────────────────────────────────────────────────────────────
# PR-S19 — F&O scanner routes (index option-chain snapshot + strategy hints)
# ────────────────────────────────────────────────────────────────────


_FNO_INDEX_SYMBOLS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")

# O.3 (2026-05-31) — guardrail for per-stock option chain. NSE has ~190
# F&O-eligible stocks; rather than hard-code the entire list we accept
# any symbol that looks like an NSE ticker (alphanumeric, len ≤ 20) and
# let the broker chain fetch fail gracefully for non-F&O names.


def _is_valid_fno_target(sym: str) -> bool:
    s = (sym or "").upper().strip()
    if not s or len(s) > 20:
        return False
    if s in _FNO_INDEX_SYMBOLS:
        return True
    # Stock symbols: A-Z plus optional & or - (e.g. M&M, BAJAJ-AUTO)
    return all(c.isalnum() or c in "&-" for c in s)


@router.get("/fno/lot-sizes")
async def fno_lot_sizes():
    """Public lot-size reference (Jan 2026 NSE revision)."""
    from ..services.fno_scanner import LOT_SIZES, FUTURE_TICK_SIZES
    return {
        "effective_from": "2026-01",
        "lot_sizes": dict(LOT_SIZES),
        "tick_sizes": dict(FUTURE_TICK_SIZES),
        "note": "Updated per Jan 2026 NSE revision. Stock futures keep per-stock lot sizes managed by NSE.",
    }


@router.get("/fno/snapshot/{symbol}")
async def fno_snapshot(
    request: Request,
    symbol: str = Path(..., description="Index symbol — NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY"),
    include_strategies: bool = Query(True, description="Include rule-based strategy suggestions"),
):
    """Index F&O option-chain snapshot + (optionally) ranked strategy
    suggestions for the current regime.

    Public — uses admin Kite option chain so it doesn't require a connected
    broker. Returns 503 when option-chain provider is offline.
    """
    ent = entitlement_for(request, DataClass.FNO_CHAIN)
    if not ent.allowed:
        return entitlement_marker(ent, {
            "snapshot": None, "volatility": None, "strategies": [],
        })

    # O.3 — accept any F&O-eligible target (indices + stocks).
    # Broker chain fetch fails cleanly for non-F&O names (returns []).
    sym = symbol.upper().strip()
    if not _is_valid_fno_target(sym):
        raise HTTPException(400, f"invalid F&O symbol: {symbol}")

    from ..services.fno_scanner import fetch_index_snapshot, suggest_strategies, teach_snapshot

    snap = await asyncio.to_thread(fetch_index_snapshot, sym)
    if snap is None:
        raise HTTPException(503, f"F&O option-chain provider unavailable for {sym}")

    # India VIX — best-effort from MarketDataProvider; degrade gracefully
    vix_value: Optional[float] = None
    try:
        from ..data.market import get_market_data_provider
        mp = get_market_data_provider()
        q = mp.get_quote("VIX")
        if q and q.ltp:
            vix_value = float(q.ltp)
    except Exception:
        pass

    # IV Rank / IV Percentile — records today's ATM IV (forward accumulation)
    # and reads the trailing window. Deterministic; grounds the vol agent.
    from ..services.fno_scanner.iv_store import iv_rank_percentile
    vol = await asyncio.to_thread(iv_rank_percentile, sym, snap.iv_atm)

    snap_dict = snap.to_dict()
    snap_dict["iv_rank"] = vol.get("iv_rank")
    snap_dict["iv_percentile"] = vol.get("iv_percentile")
    response: Dict[str, Any] = {
        "snapshot": snap_dict,
        "volatility": vol,
        "teach": teach_snapshot(snap_dict),  # deterministic plain-English read (0 tokens)
        "india_vix": vix_value,
    }
    if include_strategies:
        from ..services.fno_scanner import classify_vix_regime
        suggestions = suggest_strategies(snap, vix=vix_value, iv_rank=vol.get("iv_rank"))
        response["regime"] = classify_vix_regime(vix_value)
        response["strategies"] = [s.to_dict() for s in suggestions]
    return response


@router.get("/fno/best-trade/{symbol}")
async def fno_best_trade(
    symbol: str = Path(..., description="Index symbol — NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY"),
    use_llm: bool = Query(False, description="Include the grounded best-trade narrative (cached per symbol/day)"),
):
    """AI Options Copilot — wraps the rule-based F&O suggester. Returns the
    deterministic ranked strategy candidate(s) for the index (ALWAYS, 0 tokens;
    best = strategies[0]) plus a grounded narrative naming the best trade +
    risk/reward, only when use_llm. Honest-empty (strategies=[]) when the option
    chain is unavailable. Public — uses the admin Kite chain (no broker creds).
    """
    sym = symbol.upper().strip()
    if not _is_valid_fno_target(sym):
        raise HTTPException(400, f"invalid F&O symbol: {symbol}")

    from ..services.fno_scanner.options_copilot import best_trade
    res = await asyncio.to_thread(best_trade, sym, use_llm=use_llm)
    return {"success": True, **res}


# ────────────────────────────────────────────────────────────────────
# PR-P2 — Intraday scanner routes (5m/15m setups)
# ────────────────────────────────────────────────────────────────────


@router.get("/intraday/catalog")
async def intraday_catalog():
    """List every intraday setup with id, name, timeframe, direction."""
    from ..services.intraday_scanner import SETUP_CATALOG
    return {"setups": SETUP_CATALOG, "count": len(SETUP_CATALOG)}


@router.get("/intraday/scan")
async def intraday_scan(
    universe: str = Query("nifty50", description="nifty50|nifty100|nifty500"),
    setups: Optional[str] = Query(None, description="Comma-separated setup ids; default = all"),
    limit: int = Query(50, ge=5, le=200),
):
    """Scan intraday universe for 5m/15m setups.

    Returns ranked matches sorted by confidence + R:R. Reads 5m bars
    via the market data provider (Kite primary, yfinance fallback)
    with a 60-day window (yfinance limit for 5m bars).

    Per the deep-research audit, signals fired during lunch lull
    (12:30-13:30 IST) or closing auction (15:20-15:30 IST) are
    suppressed by the detectors themselves.
    """
    from ..services.intraday_scanner import scan_intraday_setups, SETUP_CATALOG
    from ..ai.qlib.data_handler import load_universe
    from ..data.market import get_market_data_provider

    syms = load_universe(universe)
    if not syms:
        raise HTTPException(400, f"unknown universe: {universe}")
    syms = syms[:80]   # cap for latency — full universe goes via cron later

    requested = None
    if setups:
        valid_ids = {s["id"] for s in SETUP_CATALOG} | {"gap_and_go"}
        requested = [s.strip() for s in setups.split(",")
                     if s.strip() in valid_ids]

    mp = get_market_data_provider()

    def _fetch(sym: str):
        try:
            df = mp.get_historical(sym, period="60d", interval="5m")
            if df is None or df.empty:
                return None
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            # The detector module expects an IST-aware index
            if df.index.tz is None:
                try:
                    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
                except Exception:
                    df.index = df.index.tz_localize("Asia/Kolkata")
            return df
        except Exception:
            return None

    # Prior closes for gap_and_go — best-effort, skip on failure
    prior_closes: Dict[str, float] = {}
    if requested is None or "gap_and_go" in requested:
        for sym in syms[:30]:    # gap-and-go is the only setup needing this
            try:
                d = mp.get_historical(sym, period="5d", interval="1d")
                if d is not None and len(d) >= 2:
                    prior_closes[sym] = float(d["close"].iloc[-2])
            except Exception:
                continue

    matches = await asyncio.to_thread(
        scan_intraday_setups, syms,
        bars_fetcher=_fetch, setup_ids=requested,
        prior_closes=prior_closes, max_workers=6,
    )
    out = [m.to_dict() for m in matches[:limit]]
    return {
        "universe": universe,
        "symbols_scanned": len(syms),
        "setups_run": len(requested) if requested else len(SETUP_CATALOG),
        "matches": out,
        "count": len(out),
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/fno/adjustments")
async def fno_adjustments(payload: Dict[str, Any]):
    """O.2 — Strategy adjustment engine. Given an open multi-leg position
    (snapshot of position_row + legs + spot + vix), return ranked
    adjustment suggestions (roll, hedge, defend, close, scale-out).

    Body shape:
        {
          "position": {net_premium, unrealized_pnl, expiry_date, ...},
          "legs":     [{side, option_type, strike, ...}, ...],
          "spot":     24000,
          "vix":      17.5         # optional
        }
    """
    from ..services.fno_scanner import suggest_adjustments
    position = payload.get("position") or {}
    legs = payload.get("legs") or []
    spot = float(payload.get("spot") or 0)
    vix = payload.get("vix")
    vix = float(vix) if vix is not None else None
    if not position or not legs or spot <= 0:
        raise HTTPException(400, "position, legs, and spot are required")
    suggestions = suggest_adjustments(position, legs, spot=spot, vix=vix)
    return {
        "count": len(suggestions),
        "adjustments": [s.to_dict() for s in suggestions],
    }


@router.get("/fno/oi-heatmap/{symbol}")
async def fno_oi_heatmap(
    request: Request,
    symbol: str = Path(..., description="Index symbol — NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY"),
):
    """Strike-wise OI snapshot for the dashboard's heatmap row.

    Returns per-strike CE+PE OI + OI-change so the UI can render the
    classic 'put OI = support, call OI = resistance' band with
    institutional buildup highlights.
    """
    ent = entitlement_for(request, DataClass.FNO_CHAIN)
    if not ent.allowed:
        return entitlement_marker(ent, {
            "symbol": symbol.upper().strip(), "spot": 0.0,
            "rows": [], "strike_count": 0,
        })

    # O.3 — OI heatmap also extends to per-stock chains (RELIANCE, HDFCBANK, etc.)
    sym = symbol.upper().strip()
    if not _is_valid_fno_target(sym):
        raise HTTPException(400, f"invalid F&O symbol: {symbol}")

    try:
        from ..data.market import get_market_data_provider
        mp = get_market_data_provider()
        chain = await asyncio.to_thread(mp.get_option_chain, sym, "")
    except Exception as e:
        raise HTTPException(503, f"option chain unavailable: {e}")
    if not chain:
        raise HTTPException(503, "option chain returned empty")

    # Aggregate per strike
    by_strike: Dict[float, Dict[str, Any]] = {}
    spot = 0.0
    try:
        q = mp.get_quote(sym)
        spot = float(q.ltp) if q and q.ltp else 0.0
    except Exception:
        pass

    for row in chain:
        try:
            strike = float(row.get("strike", 0) or 0)
            if strike <= 0:
                continue
            otype = str(row.get("option_type", "")).upper()
            oi = int(row.get("oi", 0) or 0)
            oi_change = int(row.get("oi_change", 0) or 0)
            entry = by_strike.setdefault(strike, {
                "strike": strike,
                "call_oi": 0, "put_oi": 0,
                "call_oi_change": 0, "put_oi_change": 0,
            })
            if otype == "CE":
                entry["call_oi"] += oi
                entry["call_oi_change"] += oi_change
            elif otype == "PE":
                entry["put_oi"] += oi
                entry["put_oi_change"] += oi_change
        except Exception:
            continue

    rows = sorted(by_strike.values(), key=lambda r: r["strike"])
    # Annotate distance from spot for the UI to colour ATM band
    for r in rows:
        if spot > 0:
            r["distance_pct"] = round((r["strike"] - spot) / spot * 100, 2)
        else:
            r["distance_pct"] = None

    return {"symbol": sym, "spot": spot, "rows": rows, "strike_count": len(rows)}


@router.get("/fno/flow/{symbol}")
async def fno_flow(
    request: Request,
    symbol: str = Path(..., description="Index/stock F&O symbol — NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, RELIANCE, ..."),
):
    """ONE consolidated options-flow summary — call/put writing, net PCR,
    max-pain pull, biggest OI buildup + a deterministic positioning lean.

    Folds metrics today scattered across the snapshot / OI-heatmap / stock
    scanners into a single card payload. Deterministic (0 LLM tokens), public
    (admin chain). Returns 503 when the option-chain provider is offline.
    """
    ent = entitlement_for(request, DataClass.FNO_CHAIN)
    if not ent.allowed:
        return entitlement_marker(ent, {
            "symbol": symbol.upper().strip(), "spot": None,
            "pcr": None, "max_pain": None, "lean": None, "top_buildup": [],
        })

    sym = symbol.upper().strip()
    if not _is_valid_fno_target(sym):
        raise HTTPException(400, f"invalid F&O symbol: {symbol}")

    from ..services.fno_scanner.options_flow import options_flow

    flow = await asyncio.to_thread(options_flow, sym)
    if flow is None:
        raise HTTPException(503, f"F&O option-chain provider unavailable for {sym}")
    return flow


@router.get("/fno/stock-scanners")
async def fno_stock_scanners():
    """Per-stock F&O signal summary — Long/Short Buildup, Long Unwinding,
    Short Covering, OI Spike — in one shot for the F&O dashboard.

    Each scanner gracefully degrades to empty when NSE participant OI
    data is unavailable (no synthetic fallback per project lock).
    """
    from ..data.screener.nse_data import get_nse_data
    nse = get_nse_data()
    fii_dii = nse.get_fii_dii()
    oi_data = nse.get_oi_spurts()
    spurts = oi_data.get("data", [])

    # Classify each row into the appropriate bucket
    long_buildup, short_buildup, long_unwinding, short_covering, oi_spike = [], [], [], [], []
    for s in spurts:
        s.get("symbol", "")
        chg = s.get("change_pct", 0)
        oi_chg = s.get("oi_change_pct", 0)
        if abs(oi_chg) >= 20.0:
            oi_spike.append(s)
        # Classification (deterministic — no overlap)
        if chg > 0.5 and oi_chg > 5:
            long_buildup.append({**s, "classification": "long_buildup"})
        elif chg < -0.5 and oi_chg > 5:
            short_buildup.append({**s, "classification": "short_buildup"})
        elif chg < -0.5 and oi_chg < -5:
            long_unwinding.append({**s, "classification": "long_unwinding"})
        elif chg > 0.5 and oi_chg < -5:
            short_covering.append({**s, "classification": "short_covering"})

    return {
        "fii_dii": fii_dii.to_dict(),
        "oi_source": oi_data.get("source"),
        "oi_last_error": oi_data.get("last_error"),
        "buckets": {
            "long_buildup": long_buildup[:25],
            "short_buildup": short_buildup[:25],
            "long_unwinding": long_unwinding[:25],
            "short_covering": short_covering[:25],
            "oi_spike": oi_spike[:25],
        },
        "counts": {
            "long_buildup": len(long_buildup),
            "short_buildup": len(short_buildup),
            "long_unwinding": len(long_unwinding),
            "short_covering": len(short_covering),
            "oi_spike": len(oi_spike),
        },
    }


@router.get("/fno/snapshot-all")
async def fno_snapshot_all(request: Request):
    """Snapshot every supported index in one call — for the F&O dashboard tab."""
    ent = entitlement_for(request, DataClass.FNO_CHAIN)
    if not ent.allowed:
        return entitlement_marker(ent, {"any_live": False, "indices": {}})

    from ..services.fno_scanner import fetch_index_snapshot

    out: Dict[str, Any] = {}
    for sym in _FNO_INDEX_SYMBOLS:
        try:
            snap = await asyncio.to_thread(fetch_index_snapshot, sym)
            out[sym] = snap.to_dict() if snap else None
        except Exception:
            out[sym] = None

    # Tag whether any index returned data — surfaces a clear "F&O offline"
    # state in the UI when admin Kite is down without 500ing the request.
    any_live = any(v is not None for v in out.values())
    return {"any_live": any_live, "indices": out}


@router.get("/v2/scanner-catalog")
async def power_screeners_catalog():
    """List every scanner with category, weight, direction, horizon, setup_type.
    Used by the frontend to render the scanner picker with multi-axis filters."""
    from ..services.screener_v2.confluence import (
        SCANNER_CATEGORIES, SCANNER_WEIGHTS, BEARISH_SCANNERS,
        SCANNER_HORIZON, SCANNER_SETUP_TYPE,
    )
    from ..data.screener.engine import SCANNER_MENU
    submenu = SCANNER_MENU["scan_types"]["X"]["submenu"]
    out = []
    for sid, info in submenu.items():
        if sid == 0:
            continue
        out.append({
            "id": sid,
            "name": info.get("name", f"Scanner {sid}"),
            "description": info.get("description", ""),
            "category": SCANNER_CATEGORIES.get(sid, "other"),
            "weight": SCANNER_WEIGHTS.get(sid, 1.0),
            "direction": "bearish" if sid in BEARISH_SCANNERS else "bullish",
            "horizon": SCANNER_HORIZON.get(sid, "swing"),
            "setup_type": SCANNER_SETUP_TYPE.get(sid, "other"),
        })
    out.sort(key=lambda x: (x["horizon"], x["setup_type"], -x["weight"]))
    return {"scanners": out, "count": len(out)}


@router.get("/patterns/v2/sectors")
async def patterns_v2_sectors():
    """List canonical sectors + per-sector symbol counts.

    Used by the frontend to render sector filter chips with counts so
    users can see e.g. "IT (42 symbols)" before filtering.
    """
    from ..ai.sector_taxonomy import CANONICAL_SECTORS, sector_for_symbol
    from ..services.chart_patterns import full_nse_universe

    universe = full_nse_universe()
    counts: Dict[str, int] = {s: 0 for s in CANONICAL_SECTORS}
    untagged = 0
    for sym in universe:
        sec = sector_for_symbol(sym)
        if sec and sec in counts:
            counts[sec] += 1
        else:
            untagged += 1

    return {
        "universe_size": len(universe),
        "tagged_count": sum(counts.values()),
        "untagged_count": untagged,
        "sectors": [
            {"sector": s, "count": counts[s]} for s in CANONICAL_SECTORS
        ],
    }


@router.get("/patterns/v2/scan/stream")
async def patterns_v2_scan_stream(
    request: Request,
    universe: str = Query("nifty500", description="nifty50|nifty100|nifty500|nse_all"),
    timeframe: str = Query("1d", description="1d|1h|15m"),
    sectors: Optional[str] = Query(None, description="Comma-separated canonical sectors (e.g. IT,Banking)"),
    direction: Optional[str] = Query(None, description="bullish|bearish (filter)"),
    limit: int = Query(100, ge=10, le=500, description="Max matches to surface (stream stops emitting after this)"),
):
    """Server-Sent Events stream of pattern matches as they're found.

    PR-S2 (2026-05-31): for the full NSE universe (~2,136 symbols) the
    blocking /v2/scan endpoint would take 15+ minutes. This SSE variant
    fans out in batches of 12 with 8 workers, yields each batch's hits
    plus a progress event so the UI can render a live progress bar +
    append-as-you-go results table.

    Event format (each line prefixed `data: `, separated by blank line):
        {"type":"start","universe":"nse_all","total":2136}
        {"type":"progress","processed":24,"total":2136}
        {"type":"match","symbol":"RELIANCE","matches":[…]}
        {"type":"done","total_matches":47,"elapsed_s":94.2}

    Disconnect handling: the generator checks `await request.is_disconnected()`
    every batch so the user closing the tab stops the scan immediately.
    """
    from ..services.chart_patterns import (
        scan_universe_streaming, full_nse_universe, filter_by_sector,
    )
    from ..ai.qlib.data_handler import load_universe
    from ..data.market import get_market_data_provider

    # Resolve symbol pool
    if universe == "nse_all":
        syms = full_nse_universe()
    else:
        syms = load_universe(universe) or []
    if not syms:
        raise HTTPException(400, f"unknown universe: {universe}")

    # Sector pre-filter (drops untagged smallcaps when explicit)
    if sectors:
        sector_list = [s.strip() for s in sectors.split(",") if s.strip()]
        syms = filter_by_sector(syms, sector_list)

    if not syms:
        raise HTTPException(400, "no symbols match the selected sectors")

    # Resolve current regime once up front
    regime_str: Optional[str] = None
    try:
        from ..services.regime.resolver import resolve_regime_at
        from ..core.database import get_supabase_admin
        sb = get_supabase_admin()
        regime_row = resolve_regime_at(sb, date.today())
        regime_str = (regime_row or {}).get("regime")
    except Exception:
        pass

    tf = (timeframe or "1d").lower()
    if tf not in _TIMEFRAME_FETCH_MAP:
        raise HTTPException(400, f"unsupported timeframe: {timeframe}. Use 1d, 1h, or 15m.")
    period_str, interval_str = _TIMEFRAME_FETCH_MAP[tf]

    mp = get_market_data_provider()

    def _fetch(sym: str):
        try:
            df = mp.get_historical(sym, period=period_str, interval=interval_str)
            if df is not None and not df.empty:
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception:
            return None

    async def _event_gen():
        import json as _json
        t0 = _time.monotonic()
        total_matches = 0
        sent = 0
        wanted_direction = (direction or "").lower() or None

        # Start event
        yield (
            "event: start\n"
            f"data: {_json.dumps({'type': 'start', 'universe': universe, 'timeframe': tf, 'total': len(syms), 'regime': regime_str, 'sectors': sectors})}\n\n"
        )

        try:
            async for kind, payload in scan_universe_streaming(
                syms, bars_fetcher=_fetch, regime=regime_str,
                max_workers=8, batch_size=12,
            ):
                # Client cancelled (closed tab) — bail out
                if await request.is_disconnected():
                    logger.info("SSE stream cancelled by client")
                    return

                if kind == "match":
                    sym, matches = payload
                    out = [m.to_dict() for m in matches]
                    if wanted_direction:
                        out = [m for m in out if m["direction"] == wanted_direction]
                    if out:
                        total_matches += len(out)
                        yield (
                            "event: match\n"
                            f"data: {_json.dumps({'type': 'match', 'symbol': sym, 'matches': out})}\n\n"
                        )
                        sent += len(out)
                        if sent >= limit:
                            yield (
                                "event: done\n"
                                f"data: {_json.dumps({'type': 'done', 'total_matches': total_matches, 'elapsed_s': round(_time.monotonic() - t0, 2), 'reason': 'limit_reached'})}\n\n"
                            )
                            return
                elif kind == "progress":
                    yield (
                        "event: progress\n"
                        f"data: {_json.dumps({'type': 'progress', **payload})}\n\n"
                    )
                elif kind == "done":
                    yield (
                        "event: done\n"
                        f"data: {_json.dumps({'type': 'done', 'total_matches': total_matches, 'elapsed_s': round(_time.monotonic() - t0, 2)})}\n\n"
                    )
        except Exception as e:
            logger.exception("SSE stream error: %s", e)
            import json as _json2
            yield (
                "event: error\n"
                f"data: {_json2.dumps({'type': 'error', 'error': str(e)[:200]})}\n\n"
            )

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # disable proxy buffering for SSE
            "Connection": "keep-alive",
        },
    )


@router.get("/patterns/v2/types")
async def patterns_v2_types():
    """List every pattern type the v2 scanner can detect + which direction
    each is biased toward. Used by the frontend to render the filter chips."""
    from ..services.chart_patterns.scanner import _BEARISH_PATTERNS, _BULLISH_PATTERNS
    types = []
    for p in sorted(_BULLISH_PATTERNS):
        types.append({"pattern": p, "direction": "bullish"})
    for p in sorted(_BEARISH_PATTERNS):
        types.append({"pattern": p, "direction": "bearish"})
    return {"types": types, "count": len(types)}


@router.get("/patterns/{pattern_type}")
async def get_pattern_stocks(
    pattern_type: str = Path(..., description="Pattern: vcp, cup_handle, double_bottom, engulfing, etc."),
):
    """Get stocks matching specific chart patterns.

    PR-S1.4 (2026-05-30): the pattern scanner used to be gated behind
    `RequireFeature("scanner_lab")` so Free-tier users got a 401 with no
    feedback. Now public — caching at the route level (60 s TTL) absorbs
    repeated requests, and the response is the same one every other
    tier sees. Tier-based filtering (limit per day, depth of metadata)
    is being moved up into PR-S5 where the BreakoutMetaLabeler RF
    score adds the Pro/Elite value above the free baseline.
    """
    pattern_map = {
        "vcp": 14,
        "cup_handle": 23,
        "double_bottom": 24,
        "head_shoulders": 25,
        "bullish_engulfing": 12,
        "bearish_engulfing": 13,
        "inside_bar": 28,
        "nr4": 21,
        "nr7": 22,
    }

    if pattern_type not in pattern_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pattern. Available: {list(pattern_map.keys())}"
        )

    return await _scanner_cached(pattern_map[pattern_type], "N", "12")


@router.get("/vcp", include_in_schema=False)
async def get_vcp_patterns():
    """[DEPRECATED] Use /patterns/vcp. Kept for back-compat."""
    return await get_pattern_stocks("vcp")


@router.get("/reversals", include_in_schema=False)
async def get_reversal_candidates(exchange: str = Query("N", description="Exchange code")):
    """[DEPRECATED] Use /scan/category/reversal. Kept for back-compat."""
    return await run_category_scan("reversal", exchange)


@router.get("/institutional", include_in_schema=False)
async def get_institutional_picks(exchange: str = Query("N", description="Exchange code")):
    """[DEPRECATED] Use /scan/category/smart_money. Kept for back-compat."""
    return await run_category_scan("smart_money", exchange)


@router.get("/bullish-tomorrow", include_in_schema=False)
async def get_bullish_tomorrow(limit: int = Query(10, ge=1, le=50)):
    """[DEPRECATED] Use /ai/ml-signals. Kept for back-compat."""
    return await get_ml_signals(limit=limit)


@router.get("/fo/long-buildup", include_in_schema=False)
async def get_long_buildup():
    """[DEPRECATED] Use /scan/41. Kept for back-compat."""
    screener = get_live_screener()
    result = await screener.run_scanner(41, "F", "0")
    return result


@router.get("/fo/short-buildup", include_in_schema=False)
async def get_short_buildup():
    """[DEPRECATED] Use /scan/42. Kept for back-compat."""
    screener = get_live_screener()
    result = await screener.run_scanner(42, "F", "0")
    return result


@router.get("/smart-money/fii-dii", include_in_schema=False)
async def get_fii_dii_data(request: Request):
    """[DEPRECATED] Use /scan/36. Kept for back-compat."""
    ent = entitlement_for(request, DataClass.FII_DII)
    if not ent.allowed:
        return entitlement_marker(ent, {
            "success": True, "scanner_id": 36, "results": [], "count": 0,
        })

    screener = get_live_screener()
    result = await screener.run_scanner(36, "N", "12")
    return result


# ============================================================================
# CATEGORY & BATCH SCAN ENDPOINTS (Frontend integration)
# ============================================================================

# TODO(WP-CONSOLIDATE): dedup /pk/* onto canonical /scanners + /scan/{id}.
# Kept functional but hidden from the OpenAPI schema. /scanners returns
# `categories` as a LIST of {id,name,scanners:[int]} (bare scanner IDs),
# whereas the frontend Screeners tab needs an object keyed by category with
# each scanner enriched to {id,name,description}. This adapter does that
# server-side by joining SCANNER_MENU; reconciling on the frontend was out of
# scope for this small cleanup, so the routes stay for now.
@router.get("/pk/categories", include_in_schema=False)
async def get_pk_categories():
    """
    Get all scanner categories with their scanners for the frontend UI.
    Returns categories keyed by ID with name and scanner list.
    """
    screener = get_live_screener()
    scanner_data = screener.get_all_scanners()
    scanner_details = SCANNER_MENU["scan_types"]["X"]["submenu"]

    categories = {}
    for cat in scanner_data.get("categories", []):
        cat_id = cat["id"]
        scanners = []
        for sid in cat.get("scanners", []):
            info = scanner_details.get(sid, {})
            scanners.append({
                "id": sid,
                "name": info.get("name", f"Scanner {sid}"),
                "menu_code": info.get("description", ""),
            })
        categories[cat_id] = {
            "name": cat["name"],
            "scanners": scanners,
        }

    total = sum(len(c["scanners"]) for c in categories.values())
    return {
        "success": True,
        "categories": categories,
        "total_scanners": total,
    }


# TODO(WP-CONSOLIDATE): dedup /pk/* onto canonical /scanners + /scan/{id}.
# Hidden from the OpenAPI schema but kept functional. This wraps the canonical
# scanner run with universe→index mapping + a limit cap the frontend relies on.
@router.post("/pk/scan/batch", include_in_schema=False)
async def run_batch_scan(
    scanner_id: int = Query(..., description="Scanner ID to run"),
    universe: str = Query("nifty500", description="Stock universe"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    user: UserTier = Depends(RequireFeature("scanner_lab")),  # Pro+; was ungated
):
    """
    Run a scanner and return batch results.
    Called by the frontend screener page when a user clicks a scanner.
    """
    index_map = {
        "nifty50": "12",
        "nifty100": "12",
        "nifty200": "12",
        "nifty500": "12",
        "full": "0",
    }
    index = index_map.get(universe, "12")

    screener = get_live_screener()
    result = await screener.run_scanner(scanner_id, "N", index)

    results = result.get("results", [])[:limit]
    return {
        "success": True,
        "scanner_id": scanner_id,
        "universe": universe,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "count": len(results),
    }


# ============================================================================
# LIVE PRICE ENDPOINT
# ============================================================================

@router.get("/prices/live")
async def get_live_prices(
    symbols: str = Query(..., description="Comma-separated symbols"),
):
    """
    Get live/recent prices for multiple symbols.
    Uses Kite Connect for real-time quotes, yfinance fallback.
    """
    from ..data.market import get_market_data_provider

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {"success": True, "prices": []}

    prices = []
    source = "kite"

    # Use configured market data provider (Kite Connect)
    provider = get_market_data_provider()
    batch = await provider.get_quotes_batch_async(symbol_list[:50])

    got_kite = False
    for symbol in symbol_list[:50]:
        quote = batch.get(symbol)
        if quote and quote.ltp > 0:
            prices.append({
                "symbol": symbol,
                "price": round(quote.ltp, 2),
                "change": round(quote.change, 2),
                "change_percent": round(quote.change_percent, 2),
            })
            got_kite = True
        else:
            prices.append({"symbol": symbol, "price": 0, "change": 0, "change_percent": 0})

    # yfinance fallback for symbols with price=0
    if not got_kite:
        source = "yfinance"
        try:
            import yfinance as yf
            for i, item in enumerate(prices):
                if item["price"] == 0:
                    try:
                        sym = item["symbol"]
                        suffix = "" if "." in sym else ".NS"
                        t = yf.Ticker(f"{sym}{suffix}")
                        fi = t.fast_info
                        p = float(fi.get("lastPrice", 0) or fi.get("last_price", 0) or 0)
                        pc = float(fi.get("previousClose", 0) or fi.get("previous_close", 0) or p)
                        if p > 0:
                            prices[i] = {
                                "symbol": sym,
                                "price": round(p, 2),
                                "change": round(p - pc, 2),
                                "change_percent": round((p - pc) / pc * 100, 2) if pc else 0,
                            }
                    except Exception:
                        pass
        except ImportError:
            pass

    return {
        "success": True,
        "prices": prices,
        "source": source,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/prices/{symbol}")
async def get_stock_price(
    symbol: str = Path(..., description="Stock symbol e.g. RELIANCE"),
):
    """
    Get detailed price data for a single stock.
    Uses Kite Connect for real-time data.
    """
    from ..data.market import get_market_data_provider

    sym = symbol.strip().upper()
    provider = get_market_data_provider()

    try:
        quote = await provider.get_quote_async(sym)
        if quote and quote.ltp > 0:
            # Stock metadata (sector/marketcap from NSE data if available)
            try:
                info = {}
            except Exception:
                info = {}

            return {
                "success": True,
                "symbol": sym,
                "name": info.get("shortName", sym),
                "price": round(quote.ltp, 2),
                "change": round(quote.change, 2),
                "change_percent": round(quote.change_percent, 2),
                "open": round(quote.open, 2),
                "high": round(quote.high, 2),
                "low": round(quote.low, 2),
                "volume": quote.volume,
                "prev_close": round(quote.close, 2),
                "week_52_high": info.get("fiftyTwoWeekHigh", 0),
                "week_52_low": info.get("fiftyTwoWeekLow", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
            }

        # yfinance fallback — the SDK call is blocking, so run it off the
        # event loop rather than directly inside this async handler.
        def _yf_fallback():
            import yfinance as yf
            suffix = "" if "." in sym else ".NS"
            ticker = yf.Ticker(f"{sym}{suffix}")
            fi = ticker.fast_info
            p = float(fi.get("lastPrice", 0) or fi.get("last_price", 0) or 0)
            pc = float(fi.get("previousClose", 0) or fi.get("previous_close", 0) or p)
            if p <= 0:
                return None
            return {
                "success": True,
                "symbol": sym,
                "name": sym,
                "price": round(p, 2),
                "change": round(p - pc, 2),
                "change_percent": round((p - pc) / pc * 100, 2) if pc else 0,
                "open": round(float(fi.get("open", 0) or 0), 2),
                "high": round(float(fi.get("dayHigh", 0) or fi.get("day_high", 0) or 0), 2),
                "low": round(float(fi.get("dayLow", 0) or fi.get("day_low", 0) or 0), 2),
                "volume": int(fi.get("lastVolume", 0) or fi.get("last_volume", 0) or 0),
                "prev_close": round(pc, 2),
                "week_52_high": round(float(fi.get("yearHigh", 0) or fi.get("year_high", 0) or 0), 2),
                "week_52_low": round(float(fi.get("yearLow", 0) or fi.get("year_low", 0) or 0), 2),
                "market_cap": int(fi.get("marketCap", 0) or fi.get("market_cap", 0) or 0),
                "pe_ratio": 0,
                "sector": "",
                "industry": "",
            }

        try:
            data = await asyncio.to_thread(_yf_fallback)
            if data:
                return data
        except Exception as yf_err:
            logger.warning(f"yfinance fallback failed for {sym}: {yf_err}")

        return {"success": False, "symbol": sym, "error": "No price data available"}
    except Exception as e:
        logger.error(f"Error fetching price for {sym}: {e}")
        return {"success": False, "symbol": sym, "error": str(e)}


@router.get("/breadth")
async def breadth_route(
    days: int = Query(120, ge=20, le=250),
):
    """Market breadth (#breadth) — true Advance/Decline issue counts today +
    A/D ratio + the cumulative A/D line (replaces the sector-average proxy)."""
    from ..services.scanners.breadth import breadth
    return {"success": True, **await asyncio.to_thread(breadth, days)}


@router.get("/alerts/live")
async def live_alerts_route(
    limit: int = Query(60, ge=5, le=200),
):
    """Smart Alerts (#8) — conditions firing right now across the universe:
    volume 3×, OI ±15%, 20-day-high breakout, IV-Rank ≥ 80. Deterministic feed
    (the detection layer the scheduler/saved-scan dispatcher can push from)."""
    from ..services.news.live_alerts import scan_live_alerts
    alerts = await asyncio.to_thread(scan_live_alerts, limit)
    return {"success": True, "alerts": alerts, "count": len(alerts)}


@router.get("/setups")
async def setups_route(
    universe: Optional[str] = Query(None, description="Universe knob; 'nse_all'/'all'/'full' widens to the full index, else Nifty500"),
):
    """AI Setup Finder — labeled counts for the 4 canonical swing setups
    (Breakout / Pullback / Trend continuation / Reversal). Reuses the existing
    live-screener scanners (no new detection); deterministic, 0 tokens.
    Honest-empty per category. ``ok`` is False only when every scanner failed."""
    from ..services.scanners.setup_finder import find_setups
    return {"success": True, **await find_setups(universe)}


@router.get("/sector-rotation")
async def sector_rotation_route(
    narrate: bool = Query(False, description="Include a grounded one-line rotation narrative (cached/day)"),
):
    """Multi-period sector rotation (#8) — RRG quadrants (Leading / Weakening /
    Lagging / Improving) from each sector's short (~5d) + long (~20d) return vs
    the market average. Optional grounded narrative of where capital is rotating."""
    from ..services.scanners.sector_rotation import sector_rotation
    rows = await asyncio.to_thread(sector_rotation)
    resp: Dict[str, Any] = {"success": True, "sectors": rows, "count": len(rows)}
    if narrate and rows:
        from ..ai.agents.grounded import grounded_reason
        from datetime import date
        leading = [r["sector"] for r in rows if r["quadrant"] == "leading"][:3]
        lagging = [r["sector"] for r in rows if r["quadrant"] == "lagging"][:3]
        resp["narrative"] = await asyncio.to_thread(
            grounded_reason,
            {"leading": leading, "lagging": lagging, "sectors": rows[:8]},
            "In one or two sentences: what sector rotation is happening — which way is capital rotating?",
            cache_key=f"rotation:{date.today().isoformat()}")
    return resp


@router.get("/market-explainer")
async def market_explainer_route(
    use_llm: bool = Query(False, description="Include the grounded market narrative (cached/day)"),
):
    """AI Market Explainer — index-level plain-English market summary. Returns
    deterministic drivers (NIFTY %, breadth, leading/lagging sectors, regime)
    always (0 tokens); the grounded narrative is produced only when use_llm.
    Honest-empty (no drivers) when no real facts can be assembled."""
    from ..services.explain.market_explainer import explain_market
    return await asyncio.to_thread(explain_market, use_llm=use_llm)


@router.get("/factor-screen")
async def factor_screen_route(
    factors: str = Query(
        "",
        description="Comma-separated factors to compose, e.g. 'momentum,low_volatility'. Empty returns the available-factor list only.",
    ),
    universe: str = Query("", description="Optional index name to scope to (e.g. 'NIFTY 500'); empty = full candle store"),
    top: int = Query(25, ge=5, le=100),
):
    """AI Factor Screener — compose CONTINUOUS factors into one ranking.

    Each requested factor (momentum / low_volatility / trend) is computed per
    symbol from real daily candles, converted to a cross-sectional percentile
    [0..100], and the composite is the mean of the selected-factor percentiles.
    Deterministic, 0 tokens. Honest-empty (results=[]) when universe data is
    thin; an empty `factors` returns just `available_factors` for the picker.
    """
    from ..services.scanners.factor_screener import factor_rank
    sel = [f.strip() for f in factors.split(",") if f.strip()]
    res = await asyncio.to_thread(
        factor_rank, sel, universe.strip() or None, top,
    )
    return {"success": True, **res}


@router.get("/probability/{symbol}")
async def probability_route(
    symbol: str = Path(..., description="NSE symbol"),
    horizon: int = Query(10, ge=3, le=30),
    target: float = Query(2.0, ge=0.5, le=10.0),
):
    """Probability Engine (#17) — empirical setup follow-through rates from the
    symbol's OWN history (20-day breakout / oversold bounce / uptrend
    continuation), each with its sample size + whether it's active now. Real
    historical outcomes, not the old fabricated formulas."""
    from ..services.scanners.probability_engine import setup_probabilities
    return {"success": True, **await asyncio.to_thread(
        setup_probabilities, symbol.strip().upper(), horizon=horizon, target=target)}


@router.get("/market-profile/{symbol}")
async def market_profile_route(
    symbol: str = Path(..., description="NSE symbol"),
    days: int = Query(60, ge=20, le=120),
    bins: int = Query(24, ge=10, le=40),
):
    """Market Profile / TPO (#21) — time-at-price distribution + POC + 70%
    Value Area (VAH/VAL). Daily-bracket approximation."""
    from ..services.market.market_profile import market_profile
    return {"success": True, **await asyncio.to_thread(market_profile, symbol.strip().upper(), days, bins)}


@router.get("/footprint/{symbol}")
async def footprint_route(
    symbol: str = Path(..., description="NSE symbol"),
    days: int = Query(60, ge=20, le=120),
):
    """Footprint / Cumulative Volume Delta (#21) — bar-level proxy (volume ×
    close-location value). Returns the CVD line + latest delta/buy% + trend.
    Honestly a daily-bar approximation (no live tick feed)."""
    from ..services.market.footprint import footprint
    return {"success": True, **await asyncio.to_thread(footprint, symbol.strip().upper(), days)}


@router.get("/interpret/{symbol}")
async def interpret_route(
    symbol: str = Path(..., description="NSE symbol"),
    use_llm: bool = Query(False, description="Include grounded one-line synthesis (cached/day)"),
):
    """AI Indicator Interpreter (#3) — RSI / MACD / ADX / trend / volume in
    plain English + overall bias + an optional grounded synthesis."""
    from ..services.explain.indicator_interpreter import interpret_symbol
    res = await asyncio.to_thread(interpret_symbol, symbol.strip().upper(), use_llm=use_llm)
    return {"success": True, **res}


@router.get("/volume-intel/{symbol}")
async def volume_intel_route(
    symbol: str = Path(..., description="NSE symbol"),
    use_llm: bool = Query(False, description="Include grounded narrative (cached/day)"),
):
    """Volume Intelligence (#9) — spike vs 20d avg + percentile + delivery trend
    + signal (accumulation / churn / high-activity / quiet), with deterministic
    drivers and an optional grounded narrative."""
    from ..services.market.volume_intelligence import volume_intel
    res = await asyncio.to_thread(volume_intel, symbol.strip().upper(), use_llm=use_llm)
    return {"success": True, **res}


@router.get("/rs/{symbol}")
async def relative_strength_route(
    symbol: str = Path(..., description="NSE symbol"),
):
    """True relative strength vs NIFTY (#7) — multi-window (~1m/2.5m/6m)
    benchmark-relative return; positive = outperforming the index."""
    from ..services.scanners.relative_strength import symbol_rs
    res = await asyncio.to_thread(symbol_rs, symbol.strip().upper())
    return {"success": True, **res}


@router.get("/verdict/{symbol}")
async def fusion_verdict_route(
    symbol: str = Path(..., description="NSE symbol"),
    use_llm: bool = Query(False, description="Include grounded narrative (cached/day)"),
):
    """Fusion Verdict — the single, explainable per-symbol setup verdict.

    Deterministically FUSES the existing specialist signals (Alpha rank +
    trend/momentum + smart-money options OI + volume + news mood + market
    regime) into one weighted, ranked verdict with per-factor leans. Event
    risk is a GATE (earnings window caps it to 'Hold off'), never a vote.
    Honest-empty (<2 factors → 'Insufficient data'). LLM only narrates."""
    from ..services.scanners.fusion_verdict import build_verdict
    res = await asyncio.to_thread(build_verdict, symbol.strip().upper(), use_llm=use_llm)
    return {"success": True, **res}


@router.get("/volume-profile/{symbol}")
async def volume_profile_route(
    symbol: str = Path(..., description="NSE symbol"),
    lookback_days: int = Query(60, ge=10, le=365),
    bins: int = Query(24, ge=8, le=60),
):
    """Volume Profile — POC / value area (VAH·VAL) / HVN / LVN computed on the
    backend from real OHLCV (volume distributed across each bar's price range).
    Deterministic; honest-empty when there aren't enough bars."""
    from ..services.market.volume_profile import volume_profile
    res = await asyncio.to_thread(
        volume_profile, symbol.strip().upper(), lookback_days=lookback_days, bins=bins
    )
    return {"success": True, **res}


@router.get("/why-moving/{symbol}")
async def why_moving(
    symbol: str = Path(..., description="NSE symbol"),
    use_llm: bool = Query(True, description="Include the grounded AI narrative (cached per symbol/day)"),
):
    """Why is a stock moving today (#highest-value) — deterministic drivers
    (price / volume vs avg / futures OI build-up / relative strength vs NIFTY /
    regime) PLUS a grounded AI narrative (free-first model, cached per
    symbol/day, only when use_llm)."""
    from ..services.explain.why_moving import explain_move
    res = await asyncio.to_thread(explain_move, symbol.strip().upper(), use_llm=use_llm)
    return {"success": True, **res}


@router.get("/earnings-preview/{symbol}")
async def earnings_preview_route(
    symbol: str = Path(..., description="NSE symbol"),
    use_llm: bool = Query(False, description="Include the grounded preview narrative (cached per symbol/day)"),
):
    """Earnings Preview agent — next confirmed announce date (live consensus
    probe; HONEST-EMPTY when no date is confirmed) + real pre-event facts:
    implied move / ATM IV / IV rank (F&O names), 1-month run-up, RS vs NIFTY.
    Deterministic drivers always (0 tokens); narrative only when use_llm."""
    from ..services.news.earnings_preview import preview
    res = await asyncio.to_thread(preview, symbol.strip().upper(), use_llm=use_llm)
    return {"success": True, **res}


@router.get("/news-digest/{symbol}")
async def news_digest_route(
    symbol: str = Path(..., description="NSE symbol"),
    use_llm: bool = Query(False, description="Include the grounded 'what the news means' narrative (cached per symbol/day)"),
):
    """News digest — deterministic news/sentiment drivers (headline counts,
    mood + trend vs prior day, price reaction vs news) PLUS an optional
    grounded narrative (cached per symbol/day). Honest-empty without news."""
    from ..services.news.news_digest import news_digest
    res = await news_digest(symbol.strip().upper(), use_llm=use_llm)
    return {"success": True, **res}


@router.get("/news-intelligence/{symbol}")
async def news_intelligence_route(
    symbol: str = Path(..., description="NSE symbol"),
    use_narrative: bool = Query(False, description="Include the grounded 'what it means' summary (cached per symbol/day)"),
    direction: Optional[str] = Query(None, description="LONG/SHORT — enables thesis-change alert (news contradicting the position)"),
    user: UserTier = Depends(current_user_tier),
):
    """News Intelligence — de-duplicated unique stories with per-story EVENT
    TYPE + MATERIALITY + URGENCY, a materiality-weighted Mood, an event/impact
    breakdown and the single most-material story. The enrichment runs on the
    free model under the per-tier ``news_intel`` cap; a capped request degrades
    to deterministic clustering only (``llm_capped: true``) rather than 402-ing.
    Cached per symbol/day. Honest-empty without news."""
    from ..ai.agents.response_cache import cache_get, cache_set, seconds_to_ist_eod
    from ..services.news.news_intelligence import analyze

    sym = symbol.strip().upper()
    dir_norm = (direction or "").strip().lower() or None
    cache_key = f"newsintel:{sym}:{date.today().isoformat()}:{int(use_narrative)}:{dir_norm or '-'}"
    cached = cache_get(cache_key)
    if cached:
        return {"success": True, "cached": True, **cached}

    use_llm = True
    llm_capped = False
    if not user.is_admin:
        from ..middleware.llm_caps import get_llm_feature_limiter
        try:
            allowed, _u, _c = get_llm_feature_limiter().consume(user.user_id, "news_intel", user.tier)
        except Exception as exc:  # noqa: BLE001 — fail-open, mirrors enforce_llm_cap
            logger.debug("news_intel cap consume skipped (%s)", exc)
            allowed = True
        if not allowed:
            use_llm, llm_capped = False, True

    res = await analyze(sym, use_llm=use_llm, use_narrative=use_narrative,
                        direction=dir_norm, user_id=user.user_id)
    if res.get("available"):
        # Only cache real results; never cache honest-empty (self-heal).
        cache_set(cache_key, res, ttl_seconds=seconds_to_ist_eod(), surface="news_intel")
    return {"success": True, "cached": False, "llm_capped": llm_capped, **res}


@router.get("/depth/{symbol}")
async def market_depth(
    symbol: str = Path(..., description="NSE symbol"),
):
    """Live L2 order-book depth + deterministic liquidity read (walls, imbalance,
    spread). 0 LLM tokens. Honest-503 when no live depth feed is available
    (requires the admin Kite feed / a connected broker)."""
    from ..data.market import get_market_data_provider
    from ..data.brokers.depth_models import analyze_depth

    sym = symbol.strip().upper()
    depth = await get_market_data_provider().get_depth_async(sym)
    if depth is None or (not depth.bids and not depth.asks):
        raise HTTPException(503, f"No live L2 depth for {sym} — connect a broker for the order book.")
    return {
        "success": True,
        "symbol": sym,
        "depth": depth.to_dict(),
        "analysis": analyze_depth(depth),
    }


@router.get("/prices/{symbol}/history")
async def get_stock_history(
    symbol: str = Path(..., description="Stock symbol"),
    days: int = Query(30, ge=1, le=365, description="Number of days"),
):
    """
    Get historical OHLCV data for a stock.
    Uses Kite Connect.
    """
    from ..data.market import get_market_data_provider

    sym = symbol.strip().upper()
    provider = get_market_data_provider()

    # Map days to period string
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

    try:
        df = await provider.get_historical_async(sym, period=period, interval="1d")

        if df is None or df.empty:
            return {"success": False, "symbol": sym, "error": "No data available"}

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        history = []
        for idx, row in df.iterrows():
            history.append({
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "open": round(float(row.get("open", 0)), 2),
                "high": round(float(row.get("high", 0)), 2),
                "low": round(float(row.get("low", 0)), 2),
                "close": round(float(row.get("close", 0)), 2),
                "volume": int(float(row.get("volume", 0))),
            })

        return {"success": True, "symbol": sym, "history": history[-days:]}
    except Exception as e:
        return {"success": False, "symbol": sym, "error": str(e)}


@router.get("/technicals/{symbol}")
async def get_stock_technicals(
    symbol: str = Path(..., description="Stock symbol"),
):
    """
    Get technical indicators for a stock.
    Uses the configured data provider and compute_all_indicators().
    """
    from ..data.market import get_market_data_provider
    from ml.features.indicators import compute_all_indicators

    sym = symbol.strip().upper()
    provider = get_market_data_provider()

    try:
        df = await provider.get_historical_async(sym, period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 20:
            return {"success": False, "symbol": sym, "error": "Insufficient data"}

        df.columns = [c.lower() for c in df.columns]
        # Heavy pandas/ta computation (PSAR, ADX, BB…) — run off the event loop
        # so one technicals request can't block every other in-flight request.
        indicator_df = await asyncio.to_thread(compute_all_indicators, df)
        last = indicator_df.iloc[-1]

        close_val = float(last.get("close", 0))
        sma_20 = float(last.get("sma_20", 0))
        sma_50 = float(last.get("sma_50", 0))

        if sma_20 > 0 and sma_50 > 0 and close_val > sma_20 > sma_50:
            trend = "Strong Uptrend"
        elif sma_20 > 0 and close_val > sma_20:
            trend = "Uptrend"
        elif sma_20 > 0 and sma_50 > 0 and close_val < sma_20 < sma_50:
            trend = "Strong Downtrend"
        elif sma_20 > 0 and close_val < sma_20:
            trend = "Downtrend"
        else:
            trend = "Sideways"

        return {
            "success": True,
            "symbol": sym,
            "rsi": round(float(last.get("rsi_14", 50)), 2),
            "macd": round(float(last.get("macd", 0)), 2),
            "macd_signal": round(float(last.get("macd_signal", 0)), 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(float(last.get("sma_200", 0)), 2) if float(last.get("sma_200", 0)) > 0 else None,
            "ema_21": round(float(last.get("ema_21", 0)), 2),
            "adx": round(float(last.get("adx", 0)), 2),
            "atr": round(float(last.get("atr_14", 0)), 2),
            "bb_upper": round(float(last.get("bb_upper", 0)), 2),
            "bb_lower": round(float(last.get("bb_lower", 0)), 2),
            "volume_ratio": round(float(last.get("volume_ratio", 1)), 2),
            "trend": trend,
        }
    except Exception as e:
        logger.error(f"Error computing technicals for {sym}: {e}")
        return {"success": False, "symbol": sym, "error": str(e)}


# ============================================================================
# ADDITIONAL AI ENDPOINTS
# ============================================================================

@router.get("/ai/market-regime")
async def get_market_regime():
    """
    Detect current market regime (Bull / Bear / Sideways)
    using multi-factor quantitative analysis with real breadth data.
    """
    screener = get_live_screener()
    regime_data = await screener.get_market_regime()

    # Also fetch Nifty level for the response
    prediction = await screener.get_nifty_prediction()
    nifty_level = prediction.get("current_level", 0)

    regime = regime_data.get("regime", "SIDEWAYS")
    confidence = regime_data.get("confidence", 50)

    return {
        "success": True,
        "feature": "Market Regime Detection",
        "regime": regime,
        "description": regime_data.get("description", ""),
        "confidence": confidence,
        "nifty_level": nifty_level,
        "factors": {
            "trend": "BULLISH" if regime == "BULL" else "BEARISH" if regime == "BEAR" else "NEUTRAL",
            "breadth": f"{regime_data.get('breadth_200sma', 50):.0f}% above 200 SMA",
            "volatility": "Low" if regime == "BULL" else "High" if regime == "BEAR" else "Moderate",
            "momentum": f"{regime_data.get('bullish_macd_pct', 50):.0f}% bullish MACD",
        },
        "stocks_analyzed": regime_data.get("stocks_analyzed", 0),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ai/momentum-radar")
async def get_momentum_radar(
    universe: str = Query("nifty500", description="Stock universe"),
    limit: int = Query(20, ge=1, le=50, description="Max results"),
):
    """
    High momentum stocks detected by AI pattern recognition.
    """
    screener = get_live_screener()
    result = await screener.run_scanner(17, "N", "12")  # Bull momentum scanner

    stocks = []
    for stock in result.get("results", [])[:limit]:
        change = stock.get("change_pct", 0)
        rsi = stock.get("rsi", 50)
        momentum_score = round(min(abs(change) * 8 + (rsi - 40) * 0.3, 100))
        stocks.append({
            **stock,
            "current_price": stock.get("ltp", 0),
            "change_percent": change,
            "momentum_score": momentum_score,
            "signal_reason": stock.get("pattern", "") or stock.get("trend", "Momentum signal"),
        })

    return {
        "success": True,
        "results": stocks,
        "count": len(stocks),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ai/breakout-scanner")
async def get_breakout_scanner(
    universe: str = Query("nifty500", description="Stock universe"),
    limit: int = Query(20, ge=1, le=50, description="Max results"),
):
    """
    Stocks near or at breakout levels detected by AI.
    """
    screener = get_live_screener()
    result = await screener.run_scanner(1, "N", "12")  # Breakout scanner

    stocks = []
    for stock in result.get("results", [])[:limit]:
        change = stock.get("change_pct", 0)
        breakout_prob = round(min(50 + abs(change) * 6 + (stock.get("rsi", 50) - 40) * 0.4, 95))
        stocks.append({
            **stock,
            "current_price": stock.get("ltp", 0),
            "change_percent": change,
            "breakout_score": round(breakout_prob * 0.8),
            "breakout_probability": breakout_prob,
            "signal_reason": stock.get("pattern", "") or "Breakout from consolidation",
        })

    return {
        "success": True,
        "results": stocks,
        "count": len(stocks),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ai/reversal-scanner")
async def get_reversal_scanner(
    universe: str = Query("nifty500", description="Stock universe"),
    limit: int = Query(20, ge=1, le=50, description="Max results"),
):
    """
    Stocks showing reversal patterns detected by AI.
    """
    screener = get_live_screener()
    result = await screener.run_scanner(9, "N", "12")  # RSI oversold scanner

    stocks = []
    for stock in result.get("results", [])[:limit]:
        rsi = stock.get("rsi", 50)
        reversal_prob = round(min(90 - rsi + abs(stock.get("change_pct", 0)) * 3, 95))
        stocks.append({
            **stock,
            "current_price": stock.get("ltp", 0),
            "change_percent": stock.get("change_pct", 0),
            "reversal_score": round(reversal_prob * 0.75),
            "reversal_probability": reversal_prob,
            "signal_reason": stock.get("pattern", "") or f"RSI oversold at {rsi:.0f}",
        })

    return {
        "success": True,
        "results": stocks,
        "count": len(stocks),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ai/trend-analysis")
async def get_trend_analysis():
    """
    Multi-timeframe trend analysis across market segments.
    Uses real breadth data and sector-wise analysis.
    """
    screener = get_live_screener()
    analysis = await screener.get_trend_analysis()

    if "error" in analysis:
        return {"success": False, "error": analysis["error"]}

    return {
        "success": True,
        "feature": "Trend Analysis",
        "summary": analysis.get("summary", {}),
        "sectors": analysis.get("sectors", {}),
        "stocks_analyzed": analysis.get("stocks_analyzed", 0),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# SWINGLENS FORECAST ENDPOINT
# ============================================================================

@router.get("/ai/swinglens-forecast/{symbol}")
async def get_swinglens_forecast(
    symbol: str = Path(..., description="Stock symbol (e.g., RELIANCE, TCS)"),
):
    """
    Get **Forecast** 5-day price forecast for a stock.

    Returns quantile predictions (p10, p50, p90), direction, and score.
    Requires the Forecast engine to be loaded.
    """
    from ..data.market import get_market_data_provider

    sym = symbol.strip().upper()

    # Check if the Forecast engine adapter is importable.
    try:
        pass
    except ImportError:
        return {
            "success": False,
            "error": "Forecast dependencies not installed",
        }

    screener = get_live_screener()
    tft = getattr(screener, "_tft_predictor", None)

    # Try loading from signal generator if screener doesn't have it
    if tft is None:
        try:
            from ..ai.signals import get_signal_generator
            sg = get_signal_generator()
            tft = getattr(sg, "_tft_predictor", None)
        except Exception:
            pass

    if tft is None:
        return {
            "success": False,
            "error": "Forecast not loaded",
        }

    # Fetch historical data for the stock
    provider = get_market_data_provider()
    try:
        df = await provider.get_historical_async(sym, period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 130:
            return {"success": False, "error": f"Insufficient data for {sym} (need 130+ bars)"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch data for {sym}: {e}"}

    # Run Forecast prediction
    try:
        result = tft.predict_for_stock(df, sym)
        if result is None:
            return {"success": False, "error": f"Forecast prediction returned empty for {sym}"}

        return {
            "success": True,
            "symbol": sym,
            "forecast": result,
            "model": "Forecast",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Forecast forecast error for {sym}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/ai/swinglens-forecast-batch")
async def get_swinglens_forecast_batch(
    symbols: str = Query("RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK", description="Comma-separated symbols"),
    limit: int = Query(10, ge=1, le=20),
):
    """
    Get Forecast forecasts for multiple stocks at once.
    Used by the AI Intelligence page Price Forecast tab.
    """
    from ..data.market import get_market_data_provider

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:limit]

    screener = get_live_screener()
    tft = getattr(screener, "_tft_predictor", None)
    if tft is None:
        try:
            from ..ai.signals import get_signal_generator
            sg = get_signal_generator()
            tft = getattr(sg, "_tft_predictor", None)
        except Exception:
            pass

    if tft is None:
        return {
            "success": False,
            "error": "Forecast not loaded",
            "forecasts": [],
        }

    provider = get_market_data_provider()
    forecasts = []

    for sym in symbol_list:
        try:
            df = await provider.get_historical_async(sym, period="6mo", interval="1d")
            if df is None or df.empty or len(df) < 130:
                continue
            result = tft.predict_for_stock(df, sym)
            if result:
                forecasts.append({
                    "symbol": sym,
                    **result,
                })
        except Exception as e:
            logger.debug(f"Forecast batch skip {sym}: {e}")
            continue

    return {
        "success": True,
        "forecasts": forecasts,
        "count": len(forecasts),
        "timestamp": datetime.now().isoformat(),
    }


# ────────────────────────────────────────────────────────────────────
# PR-S21 — Derivatives (F&O) EOD analysis (reads the populated
# derivatives_metrics_eod / options_chain_eod / participant_oi_eod /
# fno_ban tables via the service-role client — RLS blocks anon).
# ────────────────────────────────────────────────────────────────────


def _fno_eod_empty(symbol: str) -> Dict[str, Any]:
    """Honest-empty payload when no EOD F&O data exists for a symbol."""
    return {
        "success": True,
        "symbol": symbol,
        "expiry": None,
        "as_of": None,
        "metrics": None,
        "chain": [],
    }


@router.get("/fno/eod/{symbol}")
async def fno_eod_analysis(
    request: Request,
    symbol: str = Path(..., description="F&O symbol — NIFTY, BANKNIFTY, RELIANCE, etc."),
):
    """EOD derivatives analysis for a symbol's nearest expiry.

    Reads the nightly-populated EOD tables (NOT a live broker feed):
      - derivatives_metrics_eod → PCR, max-pain, total CE/PE OI
      - options_chain_eod      → by-strike CE/PE OI (pivoted)

    Picks the latest available date, then the nearest expiry >= today
    (falling back to the min expiry if every expiry is in the past).
    The chain is the top ~20 strikes by max(ce_oi, pe_oi), sorted by
    strike. Honest-empty (metrics=null, chain=[]) when there's no data.
    """
    ent = entitlement_for(request, DataClass.PARTICIPANT_OI)
    if not ent.allowed:
        return entitlement_marker(ent, _fno_eod_empty((symbol or "").upper().strip()))

    from ..core.database import get_supabase_admin

    sym = (symbol or "").upper().strip()
    if not _is_valid_fno_target(sym):
        raise HTTPException(400, f"invalid F&O symbol: {symbol}")

    sb = get_supabase_admin()

    try:
        # Latest date for which this symbol has metrics.
        latest = (
            sb.table("derivatives_metrics_eod")
            .select("date")
            .eq("symbol", sym)
            .order("date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
        if not latest:
            return _fno_eod_empty(sym)
        as_of = latest[0]["date"]

        metric_rows = (
            sb.table("derivatives_metrics_eod")
            .select("expiry, pcr_oi, pcr_volume, max_pain, total_ce_oi, total_pe_oi")
            .eq("symbol", sym)
            .eq("date", as_of)
            .execute()
            .data
        ) or []
        if not metric_rows:
            return _fno_eod_empty(sym)

        # Nearest expiry >= today, else the earliest expiry available.
        today_iso = date.today().isoformat()
        expiries = sorted(r["expiry"] for r in metric_rows if r.get("expiry"))
        if not expiries:
            return _fno_eod_empty(sym)
        future = [e for e in expiries if e >= today_iso]
        expiry = future[0] if future else expiries[0]

        metric = next((r for r in metric_rows if r.get("expiry") == expiry), None)
        if metric is None:
            return _fno_eod_empty(sym)

        metrics = {
            "pcr_oi": metric.get("pcr_oi"),
            "pcr_volume": metric.get("pcr_volume"),
            "max_pain": metric.get("max_pain"),
            "total_ce_oi": metric.get("total_ce_oi"),
            "total_pe_oi": metric.get("total_pe_oi"),
        }

        # Pivot the option chain (CE/PE → one row per strike).
        chain_rows = (
            sb.table("options_chain_eod")
            .select("strike, option_type, oi")
            .eq("symbol", sym)
            .eq("date", as_of)
            .eq("expiry", expiry)
            .execute()
            .data
        ) or []

        by_strike: Dict[float, Dict[str, float]] = {}
        for row in chain_rows:
            try:
                strike = float(row.get("strike") or 0)
            except (TypeError, ValueError):
                continue
            if strike <= 0:
                continue
            otype = str(row.get("option_type") or "").upper()
            oi = int(row.get("oi") or 0)
            entry = by_strike.setdefault(strike, {"strike": strike, "ce_oi": 0, "pe_oi": 0})
            if otype == "CE":
                entry["ce_oi"] += oi
            elif otype == "PE":
                entry["pe_oi"] += oi

        # Keep the top ~20 strikes by peak OI, then re-sort by strike so the
        # UI renders a contiguous, ascending ladder around the action.
        ranked = sorted(
            by_strike.values(),
            key=lambda r: max(r["ce_oi"], r["pe_oi"]),
            reverse=True,
        )[:20]
        chain = sorted(ranked, key=lambda r: r["strike"])

        return {
            "success": True,
            "symbol": sym,
            "expiry": expiry,
            "as_of": as_of,
            "metrics": metrics,
            "chain": chain,
        }
    except Exception as exc:
        logger.error("fno-eod query failed for %s: %s", sym, exc)
        return _fno_eod_empty(sym)


@router.get("/fno/participants")
async def fno_participants(request: Request):
    """Latest participant-wise OI (Client / DII / FII / Pro).

    Collapses the raw long/short legs into three trader-friendly axes:
      - fut_net   = fut_long − fut_short  (>0 = net long futures = bullish)
      - opt_bull  = opt_call_long + opt_put_short  (bullish option posture)
      - opt_bear  = opt_put_long + opt_call_short   (bearish option posture)

    Honest-empty (participants=[]) when the table has no data.
    """
    ent = entitlement_for(request, DataClass.PARTICIPANT_OI)
    if not ent.allowed:
        return entitlement_marker(ent, {
            "success": True, "as_of": None, "participants": [],
        })

    from ..core.database import get_supabase_admin

    sb = get_supabase_admin()
    try:
        latest = (
            sb.table("participant_oi_eod")
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
        if not latest:
            return {"success": True, "as_of": None, "participants": []}
        as_of = latest[0]["date"]

        rows = (
            sb.table("participant_oi_eod")
            .select(
                "participant, fut_long, fut_short, opt_call_long, "
                "opt_call_short, opt_put_long, opt_put_short"
            )
            .eq("date", as_of)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.error("fno-participants query failed: %s", exc)
        return {"success": True, "as_of": None, "participants": []}

    # Render FII / DII / Pro / Client in the order a trader scans them.
    order = {"fii": 0, "dii": 1, "pro": 2, "client": 3}
    participants = []
    for row in rows:
        name = str(row.get("participant") or "").strip()
        fut_long = int(row.get("fut_long") or 0)
        fut_short = int(row.get("fut_short") or 0)
        participants.append({
            "participant": name.upper() if name.lower() in ("fii", "dii", "pro") else name.title(),
            "fut_net": fut_long - fut_short,
            "opt_bull": int(row.get("opt_call_long") or 0) + int(row.get("opt_put_short") or 0),
            "opt_bear": int(row.get("opt_put_long") or 0) + int(row.get("opt_call_short") or 0),
        })
    participants.sort(key=lambda p: order.get(p["participant"].lower(), 99))

    return {"success": True, "as_of": as_of, "participants": participants}


@router.get("/fno/ban")
async def fno_ban_list(request: Request):
    """Symbols in the F&O ban period today (OI maxed out → square-off only).

    Honest-empty (symbols=[]) when nothing is banned / no data.
    """
    ent = entitlement_for(request, DataClass.PARTICIPANT_OI)
    if not ent.allowed:
        return entitlement_marker(ent, {
            "success": True, "as_of": None, "symbols": [],
        })

    from ..core.database import get_supabase_admin

    sb = get_supabase_admin()
    try:
        latest = (
            sb.table("fno_ban")
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
        if not latest:
            return {"success": True, "as_of": None, "symbols": []}
        as_of = latest[0]["date"]

        rows = (
            sb.table("fno_ban")
            .select("symbol")
            .eq("date", as_of)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.error("fno-ban query failed: %s", exc)
        return {"success": True, "as_of": None, "symbols": []}

    symbols = sorted({str(r.get("symbol") or "").upper() for r in rows if r.get("symbol")})
    return {"success": True, "as_of": as_of, "symbols": symbols}


# ────────────────────────────────────────────────────────────────────
# PR-S22 — Institutional Order-Flow (reads the populated-but-write-only
# fii_dii_flow_eod / bulk_block_deals / short_selling tables via the
# service-role client — RLS blocks anon). Public, honest-empty on error.
# ────────────────────────────────────────────────────────────────────


@router.get("/orderflow/fii-dii")
async def orderflow_fii_dii(request: Request):
    """Latest cash-market FII vs DII net flow (₹ crore).

    Reads the nightly fii_dii_flow_eod table (NOT a live feed), CASH
    segment. Net = buy − sell. FII net < 0 = foreigners sold; DII net
    > 0 = domestics bought. Honest-empty (net=0, as_of=None) on error.
    """
    ent = entitlement_for(request, DataClass.FII_DII)
    if not ent.allowed:
        return entitlement_marker(ent, {
            "success": True, "as_of": None, "segment": "CASH",
            "fii": {"buy": 0.0, "sell": 0.0, "net": 0.0},
            "dii": {"buy": 0.0, "sell": 0.0, "net": 0.0},
        })

    from ..core.database import get_supabase_admin

    sb = get_supabase_admin()
    empty = {
        "success": True,
        "as_of": None,
        "segment": "CASH",
        "fii": {"buy": 0.0, "sell": 0.0, "net": 0.0},
        "dii": {"buy": 0.0, "sell": 0.0, "net": 0.0},
    }
    try:
        latest = (
            sb.table("fii_dii_flow_eod")
            .select("date")
            .eq("segment", "CASH")
            .order("date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
        if not latest:
            return empty
        as_of = latest[0]["date"]

        rows = (
            sb.table("fii_dii_flow_eod")
            .select("fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net")
            .eq("segment", "CASH")
            .eq("date", as_of)
            .limit(1)
            .execute()
            .data
        ) or []
        if not rows:
            return empty
        r = rows[0]
        return {
            "success": True,
            "as_of": as_of,
            "segment": "CASH",
            "fii": {
                "buy": float(r.get("fii_buy") or 0),
                "sell": float(r.get("fii_sell") or 0),
                "net": float(r.get("fii_net") or 0),
            },
            "dii": {
                "buy": float(r.get("dii_buy") or 0),
                "sell": float(r.get("dii_sell") or 0),
                "net": float(r.get("dii_net") or 0),
            },
        }
    except Exception as exc:
        logger.error("orderflow fii-dii query failed: %s", exc)
        return empty


@router.get("/orderflow/deals")
async def orderflow_deals(
    limit: int = Query(15, ge=1, le=100, description="Max deals to return"),
):
    """Largest bulk/block deals on the latest available date.

    Reads bulk_block_deals (NSE-flagged large negotiated trades),
    ordered by notional value (qty × price) descending. value is
    computed server-side since the table only stores qty + price.
    Honest-empty (deals=[]) on error / no data.
    """
    from ..core.database import get_supabase_admin

    sb = get_supabase_admin()
    try:
        latest = (
            sb.table("bulk_block_deals")
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
        if not latest:
            return {"success": True, "as_of": None, "deals": []}
        as_of = latest[0]["date"]

        rows = (
            sb.table("bulk_block_deals")
            .select("symbol, deal_type, buy_sell, qty, price, client_name")
            .eq("date", as_of)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.error("orderflow deals query failed: %s", exc)
        return {"success": True, "as_of": None, "deals": []}

    deals = []
    for row in rows:
        qty = float(row.get("qty") or 0)
        price = float(row.get("price") or 0)
        deals.append({
            "symbol": str(row.get("symbol") or "").upper(),
            "deal_type": str(row.get("deal_type") or "").upper(),
            "side": str(row.get("buy_sell") or "").upper(),
            "qty": qty,
            "price": price,
            "value": qty * price,
            "client": str(row.get("client_name") or "").strip(),
        })
    deals.sort(key=lambda d: d["value"], reverse=True)
    return {"success": True, "as_of": as_of, "deals": deals[:limit]}


@router.get("/orderflow/shorts")
async def orderflow_shorts(
    limit: int = Query(15, ge=1, le=100, description="Max symbols to return"),
):
    """Most heavily short-sold symbols on the latest available date.

    Reads short_selling (NSE daily short-sale qty), ordered by qty
    descending. Honest-empty (shorts=[]) on error / no data.
    """
    from ..core.database import get_supabase_admin

    sb = get_supabase_admin()
    try:
        latest = (
            sb.table("short_selling")
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
        if not latest:
            return {"success": True, "as_of": None, "shorts": []}
        as_of = latest[0]["date"]

        rows = (
            sb.table("short_selling")
            .select("symbol, qty")
            .eq("date", as_of)
            .order("qty", desc=True)
            .limit(limit)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.error("orderflow shorts query failed: %s", exc)
        return {"success": True, "as_of": None, "shorts": []}

    shorts = [
        {"symbol": str(r.get("symbol") or "").upper(), "qty": float(r.get("qty") or 0)}
        for r in rows
        if r.get("symbol")
    ]
    return {"success": True, "as_of": as_of, "shorts": shorts}


# ────────────────────────────────────────────────────────────────────
# PR-S23 — Fundamentals analysis (reads the populated fundamentals_history
# table via the service-role client — RLS blocks anon — with a live
# screener.in fallback for symbols not yet cached). Public, honest-empty.
# ────────────────────────────────────────────────────────────────────


def _fundamentals_empty(symbol: str) -> Dict[str, Any]:
    """Honest-empty payload when no fundamentals exist for a symbol."""
    return {
        "success": True,
        "symbol": symbol,
        "as_of": None,
        "source": None,
        "fundamentals": None,
    }


def _shape_fundamentals(row: Dict[str, Any]) -> Dict[str, Any]:
    """Project a fundamentals_history-shaped row to the public payload."""
    def _f(key: str) -> Optional[float]:
        v = row.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "pe": _f("pe"),
        "roe": _f("roe"),
        "roce": _f("roce"),
        "market_cap_cr": _f("market_cap_cr"),
        "book_value": _f("book_value"),
        "dividend_yield": _f("dividend_yield"),
        "current_price": _f("current_price"),
        "sales_growth": _f("sales_growth"),
        "profit_growth": _f("profit_growth"),
        "promoter_pct": _f("promoter_pct"),
    }


# ── Fundamental screener (Phase 3, 2026-07-11) — screens the fundamentals_history
#    plane (PE/ROE/ROCE/growth/dividend/promoter), separate from the technical
#    confluence engine. Named presets + a transparent 0-5 Quality Score. ──
@router.get("/fundamental/presets")
async def fundamental_presets():
    """List the available fundamental screener presets."""
    from ..services.screener_v2.fundamental_screen import preset_catalog
    return {"success": True, "presets": preset_catalog()}


@router.get("/fundamental")
async def fundamental_screen_route(
    preset: Optional[str] = Query(None, description="Preset key, e.g. low-pe-value, high-roce-quality"),
    limit: int = Query(30, ge=5, le=100),
):
    """Run a named fundamental screen over the latest fundamentals snapshot.
    Honest-empty (with a `note`) when a preset's columns aren't populated yet."""
    from ..core.database import get_supabase_admin
    from ..services.screener_v2.fundamental_screen import run_fundamental_screen
    if not preset:
        from ..services.screener_v2.fundamental_screen import preset_catalog
        return {"success": True, "presets": preset_catalog(), "results": [], "count": 0}
    res = await asyncio.to_thread(
        run_fundamental_screen, get_supabase_admin(), preset=preset, limit=limit,
    )
    return {"success": "error" not in res, **res}


@router.get("/fundamentals/{symbol}")
async def fundamentals_analysis(
    symbol: str = Path(..., description="NSE symbol — RELIANCE, TCS, INFY, etc."),
):
    """Beginner→advanced fundamentals for a symbol.

    Reads the latest ``fundamentals_history`` snapshot (source='cached').
    When the symbol isn't cached yet, falls back to a LIVE screener.in
    fetch (source='live'), mapped through the same column shape. Honest-
    empty (fundamentals=null) when neither path yields data.
    """
    from ..core.database import get_supabase_admin

    sym = (symbol or "").upper().strip()
    if not sym:
        return _fundamentals_empty(symbol)

    sb = get_supabase_admin()

    # ── 1. Cached snapshot (latest by snapshot_date) ──
    try:
        rows = (
            sb.table("fundamentals_history")
            .select(
                "snapshot_date, pe, roe, roce, market_cap_cr, book_value, "
                "dividend_yield, current_price, sales_growth, profit_growth, "
                "promoter_pct"
            )
            .eq("symbol", sym)
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
    except Exception as exc:
        logger.error("fundamentals cached query failed for %s: %s", sym, exc)
        rows = []

    if rows:
        row = rows[0]
        return {
            "success": True,
            "symbol": sym,
            "as_of": row.get("snapshot_date"),
            "source": "cached",
            "fundamentals": _shape_fundamentals(row),
        }

    # ── 2. Live fallback (screener.in → map to the same column shape) ──
    try:
        from ..data.fundamentals.screener_in import get_fundamentals
        from ..data.reference.nse_fundamentals import map_fundamentals_row

        today_iso = date.today().isoformat()
        data = await asyncio.to_thread(get_fundamentals, sym)
        mapped = map_fundamentals_row(sym, data, today_iso)
        if mapped and any(
            mapped.get(k) is not None
            for k in (
                "pe", "roe", "roce", "market_cap_cr", "book_value",
                "dividend_yield", "current_price", "sales_growth",
                "profit_growth", "promoter_pct",
            )
        ):
            return {
                "success": True,
                "symbol": sym,
                "as_of": today_iso,
                "source": "live",
                "fundamentals": _shape_fundamentals(mapped),
            }
    except Exception as exc:
        logger.error("fundamentals live fallback failed for %s: %s", sym, exc)

    return _fundamentals_empty(sym)


# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

def register_screener_routes(app):
    """Register all screener routes with the FastAPI app"""
    app.include_router(router)
    app.include_router(quantai_router)
    logger.info("✅ AI Beta Screener routes registered (50+ scanners + AI Stock Ranker)")
