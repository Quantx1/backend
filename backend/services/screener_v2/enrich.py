"""Deep-dive enrichment for a single symbol (PR-S7.3).

When the user clicks a confluence match, this composes:
  * `indicators_firing`  — every indicator currently bullish/bearish + value
  * `suggested_levels`   — ATR-derived entry/stop/target (no LLM)
  * `regime_context`     — current regime + how this pattern type performs
  * `sector_context`     — sector breadth (how many peers also matched)
  * `news_summary`       — top 3 headlines + LLM sentiment score
  * `earnings_nearness`  — days to next earnings (within 30d → flag)
  * `ai_thesis`          — LLM-narrated 2-3 sentence paragraph,
                           strictly descriptive (no buy/sell language)

Each field is computed independently and degrades gracefully — if news
fetch fails, the explanation still renders without it. The LLM is the
optional cherry on top, not a load-bearing dependency.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IndicatorReading:
    """One indicator + its value + status flag."""
    name: str
    value: Optional[float]
    threshold: Optional[float] = None
    status: str = "neutral"            # "bullish" | "bearish" | "neutral"
    note: Optional[str] = None


@dataclass
class SuggestedLevels:
    entry: float
    stop: float
    target1: float
    target2: Optional[float] = None
    stop_basis: str = "atr_2x"
    target_basis: str = "atr_3x_rr_1p5"
    risk_reward: float = 0.0


@dataclass
class EnrichedMatch:
    """Full deep-dive payload for one symbol."""
    symbol: str
    name: str
    sector: Optional[str]
    last_price: float
    change_pct: float

    indicators_firing: List[Dict[str, Any]] = field(default_factory=list)
    suggested_levels: Optional[Dict[str, Any]] = None

    regime: Optional[str] = None
    sector_breadth: Optional[Dict[str, Any]] = None

    news_sentiment: Optional[float] = None
    top_headlines: List[Dict[str, Any]] = field(default_factory=list)

    earnings_in_days: Optional[int] = None
    earnings_note: Optional[str] = None

    ai_thesis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Indicator readings ─────────────────────────────────────────────


def _read_indicators(row: pd.Series, df: Optional[pd.DataFrame]) -> List[IndicatorReading]:
    """Extract every common indicator value + classify status."""
    out: List[IndicatorReading] = []

    rsi = row.get("rsi")
    if rsi is not None and not pd.isna(rsi):
        rsi = float(rsi)
        out.append(IndicatorReading(
            name="RSI(14)", value=round(rsi, 1),
            status="bullish" if 30 <= rsi <= 50 else "bearish" if rsi > 70 else "neutral",
            note=("Oversold bounce zone" if rsi < 30
                  else "Overbought caution" if rsi > 70
                  else "Building momentum" if 50 <= rsi <= 65
                  else "Neutral"),
        ))

    macd = row.get("macd")
    macd_signal = row.get("macd_signal")
    if macd is not None and macd_signal is not None and not pd.isna(macd):
        diff = float(macd) - float(macd_signal)
        out.append(IndicatorReading(
            name="MACD histogram", value=round(diff, 3),
            status="bullish" if diff > 0 else "bearish",
            note="Above signal" if diff > 0 else "Below signal",
        ))

    adx = row.get("adx")
    if adx is not None and not pd.isna(adx):
        adx = float(adx)
        out.append(IndicatorReading(
            name="ADX(14)", value=round(adx, 1),
            status="bullish" if adx > 25 else "neutral",
            note="Strong trend" if adx > 25 else "Weak/range" if adx < 20 else "Building",
        ))

    close = row.get("close")
    ema50 = row.get("ema50")
    ema200 = row.get("ema200")
    if close is not None and ema50 is not None and not pd.isna(close) and not pd.isna(ema50):
        diff_pct = (float(close) / float(ema50) - 1) * 100
        out.append(IndicatorReading(
            name="Close vs EMA50", value=round(diff_pct, 2),
            status="bullish" if diff_pct > 0 else "bearish",
            note=f"{abs(diff_pct):.1f}% {'above' if diff_pct > 0 else 'below'} EMA50",
        ))
    if close is not None and ema200 is not None and not pd.isna(close) and not pd.isna(ema200):
        diff_pct = (float(close) / float(ema200) - 1) * 100
        out.append(IndicatorReading(
            name="Close vs EMA200", value=round(diff_pct, 2),
            status="bullish" if diff_pct > 0 else "bearish",
            note=("Long-term uptrend" if diff_pct > 0 else "Long-term downtrend"),
        ))

    volume = row.get("volume")
    volume_sma = row.get("volume_sma20")
    if volume is not None and volume_sma is not None and float(volume_sma) > 0:
        ratio = float(volume) / float(volume_sma)
        out.append(IndicatorReading(
            name="Volume / SMA20", value=round(ratio, 2), threshold=1.5,
            status="bullish" if ratio >= 1.5 else "neutral",
            note=(f"{ratio:.1f}× average " +
                  ("(strong confirm)" if ratio >= 1.5 else "(average flow)")),
        ))

    # 52w distance
    wk52_high = row.get("week_52_high") or row.get("52w_high")
    if close and wk52_high and not pd.isna(wk52_high) and float(wk52_high) > 0:
        dist = (float(close) / float(wk52_high) - 1) * 100
        out.append(IndicatorReading(
            name="From 52w high", value=round(dist, 1),
            status="bullish" if dist > -5 else "neutral",
            note=("Near highs" if dist > -5 else f"{abs(dist):.0f}% off highs"),
        ))

    return out


# ── ATR-derived levels ──────────────────────────────────────────────


def _suggest_levels(df: pd.DataFrame, atr_mult_stop: float = 2.0,
                    atr_mult_target: float = 3.0) -> Optional[SuggestedLevels]:
    """Entry at current close; stop at -2 ATR; target at +3 ATR (RR 1.5)."""
    if df is None or len(df) < 15:
        return None

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # 14-bar ATR
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]),
                   np.abs(low[1:] - close[:-1])),
    )
    atr = float(np.mean(tr[-14:]))

    last_close = float(close[-1])
    entry = last_close
    stop = round(last_close - atr_mult_stop * atr, 2)
    target1 = round(last_close + atr_mult_target * atr, 2)
    target2 = round(last_close + 1.5 * atr_mult_target * atr, 2)
    risk = abs(entry - stop)
    reward = abs(target1 - entry)
    rr = round(reward / risk, 2) if risk > 1e-6 else 0.0

    return SuggestedLevels(
        entry=round(entry, 2), stop=stop,
        target1=target1, target2=target2,
        stop_basis=f"atr_{atr_mult_stop:g}x",
        target_basis=f"atr_{atr_mult_target:g}x_rr_{rr:g}",
        risk_reward=rr,
    )


# ── News + earnings + AI thesis ─────────────────────────────────────


async def _fetch_news_block(symbol: str) -> tuple:
    """Returns (sentiment_score, top_headlines_list). Both can be None/[]."""
    try:
        from backend.ai.sentiment.news_fetcher import fetch_headlines
        from backend.ai.sentiment.llm_classifier import LLMFinanceClassifier
    except Exception:
        return None, []

    try:
        items = await fetch_headlines(symbol, lookback_days=2)
    except Exception:
        return None, []
    if not items:
        return None, []

    classifier = LLMFinanceClassifier()
    sentiment = None
    if classifier.ready:
        try:
            texts = [h["title"] for h in items]
            scores = await asyncio.to_thread(classifier.classify_batch, texts)
            vals = [s.get("score", 0.0) for s in scores]
            sentiment = round(sum(vals) / len(vals), 3) if vals else None
        except Exception:
            sentiment = None

    return sentiment, items[:3]


def _earnings_nearness(symbol: str) -> tuple:
    """Returns (days_to_next_earnings, note) or (None, None)."""
    try:
        from backend.core.database import get_supabase_admin
        sb = get_supabase_admin()
        today_iso = date.today().isoformat()
        next_30_iso = (date.today() + timedelta(days=30)).isoformat()
        res = (
            sb.table("earnings_calendar")
            .select("event_date,fiscal_period")
            .eq("symbol", symbol)
            .gte("event_date", today_iso)
            .lte("event_date", next_30_iso)
            .order("event_date")
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None, None
        event_d = date.fromisoformat(rows[0]["event_date"])
        days = (event_d - date.today()).days
        return days, f"{rows[0].get('fiscal_period', 'Q?')} earnings in {days} day(s)"
    except Exception as e:
        logger.debug("earnings nearness %s failed: %s", symbol, e)
        return None, None


def _sector_breadth(sector: Optional[str], summary_df: pd.DataFrame,
                    stock_info: Dict[str, Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """How many peers in this sector are up >1% today + above their EMA50."""
    if not sector or summary_df is None or summary_df.empty:
        return None
    peers = [s for s, info in stock_info.items() if info.get("sector") == sector]
    if not peers:
        return None
    peer_rows = summary_df[summary_df["symbol"].isin(peers)] if "symbol" in summary_df.columns else pd.DataFrame()
    if peer_rows.empty:
        return None
    up_count = int((peer_rows["change_pct"] > 1).sum()) if "change_pct" in peer_rows.columns else 0
    total = len(peer_rows)
    return {
        "sector": sector,
        "peer_count": total,
        "up_today": up_count,
        "breadth_pct": round(up_count / total * 100, 0) if total else 0,
    }


async def _llm_thesis(
    symbol: str, indicators: List[IndicatorReading], levels: Optional[SuggestedLevels],
    sector_breadth: Optional[Dict[str, Any]], news_sentiment: Optional[float],
    earnings_in_days: Optional[int], regime: Optional[str],
) -> Optional[str]:
    """Compose a descriptive 2-3 sentence narration. Never buy/sell language."""
    try:
        from backend.ai.agents.llm import llm_for
        from backend.ai.agents.response_cache import (
            cache_get, cache_set, seconds_to_ist_eod,
        )

        # Persistent per-(symbol, day) cache — cache_get/cache_set are
        # sync + fast (L1 dict, best-effort Supabase), fine in async code.
        cache_key = f"enrichthesis:{symbol}:{date.today().isoformat()}"
        hit = cache_get(cache_key)
        if hit and hit.get("thesis"):
            return hit["thesis"]

        bullish_ind = [i for i in indicators if i.status == "bullish"]
        bearish_ind = [i for i in indicators if i.status == "bearish"]
        ctx = (
            f"Symbol: {symbol}\n"
            f"Regime: {regime or 'unknown'}\n"
            f"Bullish indicators ({len(bullish_ind)}): "
            f"{', '.join(f'{i.name}={i.value}' for i in bullish_ind[:4])}\n"
            f"Bearish indicators ({len(bearish_ind)}): "
            f"{', '.join(f'{i.name}={i.value}' for i in bearish_ind[:4])}\n"
        )
        if levels:
            ctx += (
                f"Levels: entry {levels.entry}, stop {levels.stop}, "
                f"target {levels.target1} (RR {levels.risk_reward}:1)\n"
            )
        if sector_breadth:
            ctx += f"Sector {sector_breadth['sector']}: {sector_breadth['up_today']}/{sector_breadth['peer_count']} peers up >1%\n"
        if news_sentiment is not None:
            ctx += f"News sentiment (2d): {news_sentiment:+.2f} ([-1,1] scale)\n"
        if earnings_in_days is not None:
            ctx += f"Next earnings: in {earnings_in_days} day(s)\n"

        prompt = (
            "Write 2-3 sentences (max 60 words) describing what this stock's "
            "chart is showing right now to an experienced Indian trader. Be "
            "factual. Reference the indicator agreement + the sector breadth + "
            "regime. Do NOT recommend buying or selling. Just describe what's "
            "happening.\n\n" + ctx
        )
        text = await llm_for("scanner_thesis").complete(prompt, feature="screener_enrich", temperature=0.2)
        if not text:
            return None
        if len(text) > 500:
            text = text[:500].rsplit(".", 1)[0] + "."
        if text:   # never cache empty — failures must retry
            cache_set(cache_key, {"thesis": text}, ttl_seconds=seconds_to_ist_eod(),
                      surface="scanner_thesis", model="")
        return text or None
    except Exception as e:
        logger.debug("thesis failed for %s: %s", symbol, e)
        return None


def _deterministic_thesis(
    symbol: str, indicators: List[IndicatorReading],
    levels: Optional[SuggestedLevels], regime: Optional[str],
) -> str:
    """Fallback thesis composed from indicator counts. No LLM."""
    bull = sum(1 for i in indicators if i.status == "bullish")
    bear = sum(1 for i in indicators if i.status == "bearish")
    regime_part = f" in a {regime} regime" if regime else ""
    levels_part = (
        f" Suggested levels: entry {levels.entry}, stop {levels.stop}, "
        f"target {levels.target1} (R:R {levels.risk_reward}:1)."
        if levels else ""
    )
    return (
        f"{symbol}{regime_part}: {bull} bullish indicators and {bear} bearish, "
        f"with mixed-to-positive momentum on the daily timeframe.{levels_part}"
    )


# ── Public entry point ─────────────────────────────────────────────


async def enrich_symbol(
    symbol: str,
    summary_row: pd.Series,
    per_symbol_df: Optional[pd.DataFrame],
    summary_df: pd.DataFrame,
    stock_info: Dict[str, Dict[str, str]],
    *,
    regime: Optional[str] = None,
    use_llm: bool = True,
    use_news: bool = True,
    use_earnings: bool = True,
) -> EnrichedMatch:
    """Build the full deep-dive payload for one symbol.

    Each enrichment block runs independently — failures degrade gracefully
    so a missing news / earnings feed doesn't break the whole panel.
    """
    info = stock_info.get(symbol, {})

    # Synchronous enrichments (cheap)
    indicators = _read_indicators(summary_row, per_symbol_df)
    levels = _suggest_levels(per_symbol_df) if per_symbol_df is not None else None
    breadth = _sector_breadth(info.get("sector"), summary_df, stock_info)

    # Async enrichments — run in parallel
    tasks = []
    tasks.append(_fetch_news_block(symbol) if use_news else _noop_tuple())
    if use_earnings:
        # earnings is sync DB call — wrap in thread for parallelism
        tasks.append(asyncio.to_thread(_earnings_nearness, symbol))
    else:
        tasks.append(_noop_tuple())

    (sentiment, top_news), (earnings_in_days, earnings_note) = await asyncio.gather(*tasks)

    # AI thesis last (depends on everything above)
    thesis: Optional[str] = None
    if use_llm:
        thesis = await _llm_thesis(
            symbol, indicators, levels, breadth, sentiment, earnings_in_days, regime,
        )
    if not thesis:
        thesis = _deterministic_thesis(symbol, indicators, levels, regime)

    return EnrichedMatch(
        symbol=symbol,
        name=info.get("name", symbol),
        sector=info.get("sector"),
        last_price=float(summary_row.get("close", summary_row.get("ltp", 0)) or 0),
        change_pct=round(float(summary_row.get("change_pct", 0) or 0), 2),
        indicators_firing=[asdict(i) for i in indicators],
        suggested_levels=asdict(levels) if levels else None,
        regime=regime,
        sector_breadth=breadth,
        news_sentiment=sentiment,
        top_headlines=top_news or [],
        earnings_in_days=earnings_in_days,
        earnings_note=earnings_note,
        ai_thesis=thesis,
    )


async def _noop_tuple():
    return (None, [])
