"""
News Intelligence — the cutting-edge per-symbol news+sentiment surface.

Composes the existing free stack (Google-News RSS fetch + the free-model
classifier) into a state-of-the-art read that the flat-mean Mood never gave:

  * de-duplicated UNIQUE stories (a wire story counts once; its outlet spread
    is a corroboration signal),
  * per-story EVENT TYPE (earnings / M&A / regulatory / order-win / …),
  * per-story MATERIALITY (high/medium/low) and URGENCY (breaking/recent/…),
  * a MATERIALITY-WEIGHTED mood = Σ(sentiment · impact · source-credibility ·
    recency) / Σ(weights) — so "CEO resigns / SEBI probe" outweighs a routine
    analyst note instead of counting equally,
  * an event-type + impact breakdown and the single most-material story,
  * THESIS-CHANGE detection: high-impact news that contradicts a held
    LONG/SHORT — the "your trade thesis is at risk" alert.

Invariants: deterministic-first (the weighted aggregate + clustering + urgency
work with 0 tokens); the LLM only classifies (free model) and optionally
narrates; honest-empty when there's no news; surfaces as "Mood" (never a model
name). LLM never gates a trade — this is read-only intelligence.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional

from ...ai.sentiment.news_dedup import (
    cluster_headlines, recency_weight, source_weight, urgency,
)
from ...ai.sentiment.news_enrich import IMPACT_WEIGHT, enrich_headlines, event_label

logger = logging.getLogger(__name__)

_LABEL_EPS = 0.15  # shared with market_routes / news_digest label convention


def _label(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if score >= _LABEL_EPS:
        return "bullish"
    if score <= -_LABEL_EPS:
        return "bearish"
    return "neutral"


def _empty(symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol, "available": False, "mood_score": None, "label": "neutral",
        "story_count": 0, "raw_headline_count": 0, "stories": [],
        "event_breakdown": [], "impact_counts": {"high": 0, "medium": 0, "low": 0},
        "top_story": None, "thesis": None, "narrative": None,
        "as_of": date.today().isoformat(),
    }


def _materiality_weighted_mood(stories: List[Dict[str, Any]]) -> Optional[float]:
    """Σ(score · impact · source-credibility · recency) / Σ(weights)."""
    num = den = 0.0
    for s in stories:
        w = (
            IMPACT_WEIGHT.get(s["impact"], 0.25)
            * source_weight(s.get("source"))
            * recency_weight(s.get("published"))
        )
        num += float(s.get("score", 0.0)) * w
        den += w
    if den <= 0:
        return None
    return round(max(-1.0, min(1.0, num / den)), 4)


async def analyze(
    symbol: str,
    *,
    lookback_days: int = 3,
    use_llm: bool = True,
    use_narrative: bool = False,
    direction: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The full news-intelligence read for one symbol. Async (awaits the RSS
    fetch); the blocking LLM calls run off-thread so the event loop stays free."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return _empty(sym)

    # Multi-source: fan out across all enabled free providers (Google News RSS
    # + GDELT + Yahoo Finance + publisher RSS); the clusterer dedups overlaps.
    from ...ai.sentiment.news_providers import fetch_all_sources
    try:
        headlines = await fetch_all_sources(sym, lookback_days=lookback_days)
    except Exception as exc:  # noqa: BLE001
        logger.debug("news_intelligence fetch failed for %s: %s", sym, exc)
        return _empty(sym)
    if not headlines:
        return _empty(sym)

    raw_count = len(headlines)
    clusters = cluster_headlines(headlines)  # unique stories (0 tokens)
    titles = [c["title"] for c in clusters]

    # Enrich UNIQUE stories only (dedup already cut the LLM cost). Degrades to
    # sentiment-only inside enrich_headlines when the model is unavailable.
    if use_llm:
        enriched = await asyncio.to_thread(enrich_headlines, titles)
    else:
        enriched = [
            {"label": "neutral", "score": 0.0, "confidence": 0.0, "event_type": "other", "impact": "low"}
            for _ in titles
        ]

    # Multi-model cross-check: corroborate the LLM read with FinBERT (if
    # loaded) + the finance lexicon → per-story model agreement.
    try:
        from ...ai.sentiment.sentiment_ensemble import cross_check, models_available
        agreement = await asyncio.to_thread(cross_check, titles, [e["score"] for e in enriched])
        model_set = models_available()
    except Exception as exc:  # noqa: BLE001
        logger.debug("sentiment ensemble cross-check failed: %s", exc)
        agreement = [None] * len(clusters)
        model_set = ["llm", "lexicon"]

    stories: List[Dict[str, Any]] = []
    for i, (c, e) in enumerate(zip(clusters, enriched)):
        ag = agreement[i] if i < len(agreement) else None
        stories.append({
            "title": c["title"],
            "source": c.get("source"),
            "link": c.get("link"),
            "published": c.get("published"),
            "member_count": c.get("member_count", 1),
            "sources": c.get("sources", []),
            "label": e["label"],
            "score": e["score"],
            "event_type": e["event_type"],
            "event_label": event_label(e["event_type"]),
            "impact": e["impact"],
            "urgency": urgency(c.get("published")),
            "agreement": ag,
        })

    mood = _materiality_weighted_mood(stories)
    impact_counts = Counter(s["impact"] for s in stories)
    event_counts = Counter(s["event_label"] for s in stories if s["event_type"] != "other")

    # Top story = highest impact, then most corroborated, then freshest.
    _impact_rank = {"high": 0, "medium": 1, "low": 2}
    top = sorted(
        stories,
        key=lambda s: (_impact_rank.get(s["impact"], 3), -s["member_count"]),
    )[0] if stories else None

    out = {
        "symbol": sym,
        "available": True,
        "mood_score": mood,
        "label": _label(mood),
        "story_count": len(stories),
        "raw_headline_count": raw_count,
        "stories": stories,
        "event_breakdown": [{"event": k, "count": v} for k, v in event_counts.most_common()],
        "impact_counts": {lvl: impact_counts.get(lvl, 0) for lvl in ("high", "medium", "low")},
        "top_story": top,
        "models": model_set,
        "providers": sorted({h.get("provider") for h in headlines if h.get("provider")}),
        "thesis": None,
        "narrative": None,
        "as_of": date.today().isoformat(),
    }

    if direction:
        out["thesis"] = detect_thesis_change(direction, out)

    if use_narrative and mood is not None:
        out["narrative"] = await asyncio.to_thread(_narrate, sym, out, user_id)
    return out


def _narrate(sym: str, intel: Dict[str, Any], user_id: Optional[str]) -> Optional[str]:
    try:
        from ...ai.agents.grounded import grounded_reason
        facts = {
            "symbol": sym,
            "mood": intel["label"],
            "mood_score": intel["mood_score"],
            "top_story": (intel.get("top_story") or {}).get("title"),
            "stories": [
                {"title": s["title"], "event": s["event_label"], "impact": s["impact"], "sentiment": s["label"]}
                for s in intel["stories"][:6]
            ],
        }
        return grounded_reason(
            facts,
            f"In 2-3 sentences, summarise what today's news means for {sym}. "
            f"Lead with the most material story. Use only the provided facts.",
            cache_key=f"newsintel:{sym}:{date.today().isoformat()}",
            user_id=user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("news_intelligence narrate failed for %s: %s", sym, exc)
        return None


def detect_thesis_change(direction: Optional[str], intel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Flag when high/medium-impact news CONTRADICTS a held position.

    direction: 'LONG'/'bullish' or 'SHORT'/'bearish'. Returns an alert dict
    when a material story leans against the position, else None (honest-empty).
    Pure — operates on an already-built intel payload.
    """
    if not direction or not intel.get("available"):
        return None
    d = direction.strip().lower()
    want_bear = d in ("long", "bullish", "buy")   # a LONG is threatened by BEARISH news
    want_bull = d in ("short", "bearish", "sell")
    if not (want_bear or want_bull):
        return None

    contradicting = []
    for s in intel.get("stories", []):
        if s["impact"] not in ("high", "medium"):
            continue
        if want_bear and s["label"] == "bearish":
            contradicting.append(s)
        elif want_bull and s["label"] == "bullish":
            contradicting.append(s)

    if not contradicting:
        return None
    has_high = any(s["impact"] == "high" for s in contradicting)
    return {
        "at_risk": True,
        "severity": "high" if has_high else "medium",
        "position": "LONG" if want_bear else "SHORT",
        "reason": (
            f"{len(contradicting)} material story(ies) lean against your "
            f"{'LONG' if want_bear else 'SHORT'} on {intel['symbol']}."
        ),
        "stories": [
            {"title": s["title"], "event": s["event_label"], "impact": s["impact"], "link": s.get("link")}
            for s in contradicting[:4]
        ],
    }


__all__ = ["analyze", "detect_thesis_change"]
