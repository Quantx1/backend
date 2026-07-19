"""News Digest — per-symbol news/sentiment synthesis (the news analogue of
`why_moving`). Assembles REAL facts deterministically by REUSING the standalone
SentimentEngine (live Google-News headlines + LLM classifier), the stored
`news_sentiment` daily aggregates (prior-day mood trend), and the market
provider (price reaction). Builds a `drivers` list that is ALWAYS returned
(0 narration tokens), then OPTIONALLY narrates "what the news means" with the
grounded reasoner (cached per symbol/day) only when `use_llm`. Honest-empty
(drivers == []) when there are no recent headlines — never a fabricated score.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MOOD_TTL_S = 600   # mirror /api/market/sentiment's 10-min success cache
_TREND_EPS = 0.1    # |Δ mean_score| below this reads as "steady"


def _label(score: float) -> str:
    """Same thresholds as market_routes.get_stock_sentiment (±0.15)."""
    return "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"


async def _live_mood(sym: str) -> Optional[Dict[str, Any]]:
    """Live headline mood via the standalone SentimentEngine. Successful
    results cached 10 min (shared response_cache); empty/failed NOT cached
    so they self-heal — mirroring the /api/market/sentiment behaviour."""
    from ...ai.agents.response_cache import cache_get, cache_set
    key = f"newsdigest:mood:{sym}"
    hit = cache_get(key)
    if hit:
        return hit
    try:
        from ...ai.sentiment.engine import get_sentiment_engine
        row = await get_sentiment_engine().score_symbol(sym, lookback_days=3)
    except Exception as e:  # noqa: BLE001
        logger.debug("news_digest mood fetch failed for %s: %s", sym, e)
        return None
    if not row:
        return None
    score = float(row.get("mean_score") or 0.0)
    mood = {
        "mean_score": round(score, 3),
        "label": _label(score),
        "headline_count": row.get("headline_count", 0),
        "positive": row.get("positive_count", 0),
        "negative": row.get("negative_count", 0),
        "neutral": row.get("neutral_count", 0),
        "headlines": row.get("sample_headlines", []),
        "sources": row.get("sources", []),
    }
    cache_set(key, mood, ttl_seconds=_MOOD_TTL_S, surface="news_digest", model="")
    return mood


async def assemble_facts(symbol: str) -> Dict[str, Any]:
    """Gather the real news facts for a symbol. Best-effort per factor."""
    sym = symbol.strip().upper().replace(".NS", "")
    facts: Dict[str, Any] = {"symbol": sym}

    mood = await _live_mood(sym)
    if mood:
        facts["mood"] = mood

    # Prior-day stored aggregate (news_sentiment, 16:30 IST nifty500 job) —
    # honest-missing for symbols outside the nightly universe.
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        rows = (sb.table("news_sentiment")
                .select("mean_score,headline_count,trade_date")
                .eq("symbol", sym)
                .gte("trade_date", cutoff)
                .lt("trade_date", date.today().isoformat())
                .order("trade_date", desc=True).limit(1).execute().data or [])
        if rows and rows[0].get("mean_score") is not None:
            facts["mood_prior"] = {
                "mean_score": round(float(rows[0]["mean_score"]), 3),
                "trade_date": rows[0].get("trade_date"),
                "headline_count": rows[0].get("headline_count", 0),
            }
    except Exception as e:  # noqa: BLE001
        logger.debug("news_digest prior-mood facts failed for %s: %s", sym, e)

    # Price reaction — Quote is a DATACLASS: attribute access only.
    try:
        from ...data.market import get_market_data_provider
        q = get_market_data_provider().get_quote(sym)
        chg = getattr(q, "change_percent", None)
        if chg is not None:
            facts["price"] = {"ltp": getattr(q, "ltp", None),
                              "change_pct": round(float(chg), 2)}
    except Exception as e:  # noqa: BLE001
        logger.debug("news_digest price facts failed for %s: %s", sym, e)

    return facts


def build_drivers(facts: Dict[str, Any]) -> List[str]:
    """Deterministic plain bullet drivers — always available, 0 tokens. Pure."""
    out: List[str] = []
    m = facts.get("mood") or {}
    n = m.get("headline_count") or 0
    if not n:
        return out   # no news → honest-empty digest
    out.append(f"{n} headlines in the last 3 days: {m.get('positive', 0)} positive / "
               f"{m.get('neutral', 0)} neutral / {m.get('negative', 0)} negative.")
    sc = m.get("mean_score")
    if sc is not None:
        out.append(f"News mood {m.get('label', 'neutral')} "
                   f"({'+' if sc >= 0 else ''}{sc} on a −1..+1 scale).")
    p = facts.get("mood_prior") or {}
    if sc is not None and p.get("mean_score") is not None:
        delta = round(sc - p["mean_score"], 3)
        if abs(delta) >= _TREND_EPS:
            word = "improving" if delta > 0 else "deteriorating"
            out.append(f"Mood {word} vs {p.get('trade_date', 'the prior day')} "
                       f"({'+' if p['mean_score'] >= 0 else ''}{p['mean_score']} → "
                       f"{'+' if sc >= 0 else ''}{sc}).")
        else:
            out.append("Mood steady vs the prior day.")
    pr = facts.get("price") or {}
    if pr.get("change_pct") is not None and m.get("label") in ("bullish", "bearish"):
        with_news = (m["label"] == "bullish") == (pr["change_pct"] >= 0)
        out.append(f"Price {'+' if pr['change_pct'] >= 0 else ''}{pr['change_pct']}% today — "
                   f"{'trading with' if with_news else 'diverging from'} the news.")
    return out


async def news_digest(symbol: str, *, use_llm: bool = False,
                      user_id: Optional[str] = None) -> Dict[str, Any]:
    """{symbol, facts, drivers, narrative}. Drivers deterministic + always
    returned; narrative is the grounded reasoner, cached per symbol/day,
    only when use_llm. Honest-empty (no drivers) when there's no recent news."""
    sym = symbol.strip().upper().replace(".NS", "")
    facts = await assemble_facts(sym)
    drivers = build_drivers(facts)
    narrative: Optional[str] = None
    if use_llm and drivers:
        from ...ai.agents.grounded import grounded_reason
        narrative = await asyncio.to_thread(
            lambda: grounded_reason(
                facts,
                f"What does the recent news flow mean for {sym}? Synthesize the "
                "headlines and mood into what a trader should take away — and say "
                "whether the price action confirms or diverges from the news. "
                "Describe the sentiment balance qualitatively (e.g. 'mostly "
                "positive', 'mixed') — do NOT state exact counts of positive/"
                "negative/neutral headlines (the UI already shows those).",
                cache_key=f"newsdigest:{sym}:{date.today().isoformat()}",
                user_id=user_id,
            ))
    return {"symbol": sym, "facts": facts, "drivers": drivers, "narrative": narrative}
