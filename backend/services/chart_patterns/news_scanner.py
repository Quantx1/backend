"""News-driven scanner (PR-S4) — find stocks with material news today
that the market hasn't fully priced in.

Pipeline per symbol:
  1. fetch_headlines() — Google News RSS, last 24-48h
  2. Score each headline via the LLM sentiment classifier (existing)
  3. Compute today's price reaction (close vs prev_close)
  4. Flag candidates where sentiment is strongly directional AND the
     price reaction is muted (mispricing) or aligned (momentum follow-on)
  5. Return ranked list with headline + sentiment + price + setup tag

Output is opinion-free — never recommends buy/sell. The setup tag is
descriptive ("positive_news_underreaction" / "negative_news_continuation"
/ etc.) for the user to decide.

LOCKED: this is a *discovery* surface. LLMs classify sentiment + write
narration; they don't gate any trade.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# Minimum absolute sentiment to qualify as "material"
MIN_SENTIMENT_ABS = 0.4

# Underreaction = strong sentiment but small price move
UNDERREACT_PRICE_BAND = 0.5    # ±0.5% intraday

# Continuation = sentiment aligned with current direction + price move
CONTINUATION_MOVE_MIN = 1.5    # >=1.5% in line with sentiment


@dataclass
class NewsHit:
    """One news-driven candidate."""
    symbol: str
    setup_tag: str                      # see _classify_setup() below
    news_sentiment: float               # -1..1, headline-weighted mean
    headline_count: int
    top_headline: Optional[str]
    top_headline_source: Optional[str]
    last_price: float
    change_pct_today: float
    headlines: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _classify_setup(sentiment: float, change_pct: float) -> Optional[str]:
    """Bucket the (sentiment, price-reaction) combo into a descriptive tag.

    Returns None if the combination doesn't qualify (neutral news, etc.).
    """
    if abs(sentiment) < MIN_SENTIMENT_ABS:
        return None
    s_dir = "positive" if sentiment > 0 else "negative"

    # Underreaction — strong sentiment but tiny price move
    if abs(change_pct) <= UNDERREACT_PRICE_BAND:
        return f"{s_dir}_news_underreaction"

    # Continuation — same direction + meaningful move
    sign_match = (sentiment > 0 and change_pct > 0) or (sentiment < 0 and change_pct < 0)
    if sign_match and abs(change_pct) >= CONTINUATION_MOVE_MIN:
        return f"{s_dir}_news_continuation"

    # Divergence — opposite direction
    if not sign_match and abs(change_pct) >= CONTINUATION_MOVE_MIN:
        return f"{s_dir}_news_divergence"

    return None


async def _score_headlines(headlines: List[Dict[str, Any]]) -> float:
    """Run the LLM classifier on all headlines, return weighted mean.

    Recent headlines weighted more heavily (1 / 1+age_hours). Falls back
    to neutral 0.0 if the classifier is unavailable.
    """
    if not headlines:
        return 0.0
    try:
        from backend.ai.sentiment.llm_classifier import LLMFinanceClassifier
        classifier = LLMFinanceClassifier()
        if not classifier.ready:
            return 0.0
        texts = [h["title"] for h in headlines]
        results = await asyncio.to_thread(classifier.classify_batch, texts)
        # Each result has a `score` in [-1, 1]
        scores = [r.get("score", 0.0) for r in results]
        if not scores:
            return 0.0
        # Weight recent more — assume input order is newest-first
        weights = [1.0 / (1 + i) for i in range(len(scores))]
        total_w = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_w
    except Exception as e:
        logger.debug("headline sentiment scoring failed: %s", e)
        return 0.0


async def scan_news_universe(
    symbols: Sequence[str],
    *,
    lookback_days: int = 1,
    min_headlines: int = 2,
    price_fetcher=None,                 # callable(symbol) -> {ltp, prev_close, ...}
    limit: int = 30,
) -> List[NewsHit]:
    """Fan out news + sentiment + price scoring across symbols.

    `price_fetcher` defaults to MarketDataProvider.get_quote. Passing it
    explicitly lets tests inject a fake.
    """
    from backend.ai.sentiment.news_fetcher import fetch_many

    if not symbols:
        return []

    # 1. Fetch news for all symbols (bounded internally)
    t0 = time.monotonic()
    news_by_sym = await fetch_many(list(symbols), lookback_days=lookback_days)
    logger.info(
        "scan_news_universe: fetched headlines for %d symbols in %.1fs",
        len(news_by_sym), time.monotonic() - t0,
    )

    # 2. Score sentiment in parallel (cached per symbol — news is intraday-
    #    volatile, so a SHORT 20-min TTL; never cache a 0.0 classifier-off score)
    from backend.ai.agents.response_cache import cache_get, cache_set

    async def _cached_score(sym: str, items: List[Dict[str, Any]]) -> float:
        ck = f"news:sentiment:{sym}"
        cached = cache_get(ck)
        if cached is not None:
            return float(cached["v"])
        score = await _score_headlines(items)
        if score != 0.0:
            cache_set(ck, {"v": score}, ttl_seconds=1200,
                      surface="news_sentiment", model="")
        return score

    sentiment_by_sym: Dict[str, float] = {}
    score_tasks = []
    sym_order = []
    for sym, items in news_by_sym.items():
        if len(items) < min_headlines:
            continue
        sym_order.append(sym)
        score_tasks.append(_cached_score(sym, items))
    scores = await asyncio.gather(*score_tasks, return_exceptions=True)
    for sym, s in zip(sym_order, scores):
        if isinstance(s, Exception):
            continue
        sentiment_by_sym[sym] = float(s)

    # 3. Pull live prices (sync fetcher off-thread)
    if price_fetcher is None:
        from backend.data.market import get_market_data_provider
        mp = get_market_data_provider()
        price_fetcher = mp.get_quote

    def _safe_price(sym: str):
        try:
            return price_fetcher(sym)
        except Exception:
            return None

    def _gather_prices():
        return {sym: _safe_price(sym) for sym in sentiment_by_sym.keys()}

    prices = await asyncio.to_thread(_gather_prices)

    # 4. Classify setup + build hits
    hits: List[NewsHit] = []
    for sym, sentiment in sentiment_by_sym.items():
        q = prices.get(sym)
        if q is None:
            continue
        # q is a Quote dataclass — accept attribute or dict form
        ltp = getattr(q, "ltp", None) or (q.get("ltp") if isinstance(q, dict) else None)
        change_pct = getattr(q, "change_percent", None) or (
            q.get("change_percent") if isinstance(q, dict) else None
        )
        if ltp is None or change_pct is None:
            continue
        tag = _classify_setup(sentiment, float(change_pct))
        if tag is None:
            continue
        items = news_by_sym[sym]
        top = items[0] if items else {}
        hits.append(NewsHit(
            symbol=sym,
            setup_tag=tag,
            news_sentiment=round(sentiment, 3),
            headline_count=len(items),
            top_headline=top.get("title"),
            top_headline_source=top.get("source"),
            last_price=round(float(ltp), 2),
            change_pct_today=round(float(change_pct), 2),
            headlines=items[:5],         # cap surface payload
        ))

    # Rank by |sentiment| × headline_count — strongest, most-covered first
    hits.sort(key=lambda h: abs(h.news_sentiment) * h.headline_count, reverse=True)
    return hits[:limit]
