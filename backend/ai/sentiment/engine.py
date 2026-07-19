"""
SentimentEngine — batch per-symbol sentiment for the ``news_sentiment``
table. Used by the nightly sentiment refresh job + on-demand refreshes
for the Portfolio Doctor (F7), Earnings Predictor (F9), Daily Digest
(F12), and intraday F1 enrichment.

Pipeline per symbol:

    fetch_headlines(symbol, lookback=2d)
        → classifier.classify_batch(titles)
        → aggregate mean_score + label counts
        → row ready for news_sentiment upsert

Classifier selection (per locked deep-research decision 2026-05-10):
    Primary: open-model zero-shot classifier (``LLMFinanceClassifier``)
        routed through the OpenRouter gateway
        - FinDPO paper showed FinBERT goes negative at 5bps trading costs
        - Vansh180/FinBERT-India-v1 is a 7K-sample hobby model
        - LLM zero-shot ≥ FinBERT on FPB per multiple 2025 studies
    Fallback: FinBERT-India (``USE_FINBERT_FALLBACK=1`` env override)
        - Used only when the LLM key is missing or its client fails to init
        - Useful for dev environments without network access to the gateway

If neither classifier is ready (dev env without keys or model files) the
engine returns an empty row-list — scheduler treats that as
``status=skipped reason=classifier_not_ready``.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import date
from typing import Any, Dict, List, Optional, Protocol

from .news_fetcher import fetch_many

logger = logging.getLogger(__name__)


class _ClassifierProto(Protocol):
    """Common surface between LLMFinanceClassifier + FinBERTIndia."""

    @property
    def ready(self) -> bool: ...
    def load(self) -> bool: ...
    def classify_batch(self, texts: List[str], *, max_length: int = 128,
                       batch_size: int = 32) -> List[Dict[str, float]]: ...


def _select_classifier() -> _ClassifierProto:
    """Pick the primary open-model LLM classifier, fall back to
    FinBERT-India when set or when the LLM isn't usable.

    Priority:
        1. ``USE_FINBERT_FALLBACK=1`` env → FinBERT-India (legacy / dev)
        2. LLMFinanceClassifier (OpenRouter gateway) if usable
        3. FinBERT-India as a last resort
    """
    force_finbert = os.environ.get("USE_FINBERT_FALLBACK", "").strip() in ("1", "true", "yes")
    if force_finbert:
        from .finbert_india import get_finbert  # noqa: PLC0415
        logger.info("SentimentEngine: USE_FINBERT_FALLBACK=1 → using FinBERT-India")
        return get_finbert()

    from .llm_classifier import get_classifier  # noqa: PLC0415
    clf = get_classifier()
    if clf.load():
        logger.info("SentimentEngine: using LLMFinanceClassifier (primary)")
        return clf

    from .finbert_india import get_finbert  # noqa: PLC0415
    logger.info("SentimentEngine: LLM unavailable, falling back to FinBERT-India")
    return get_finbert()


class SentimentEngine:
    _lock = threading.Lock()

    def __init__(self, *, classifier: Optional[Any] = None):
        # ``classifier`` is duck-typed against _ClassifierProto. Defaults
        # to whichever the selector picks. Accept the legacy ``finbert``
        # kw via a tiny alias so existing callers (tests) keep working.
        self._clf: _ClassifierProto = classifier or _select_classifier()

    # Back-compat property — many callers still ask for ``.engine.finbert``
    @property
    def finbert(self) -> _ClassifierProto:
        return self._clf

    @property
    def ready(self) -> bool:
        return self._clf.ready

    def load(self) -> bool:
        return self._clf.load()

    # ------------------------------------------------------------ batch ops

    async def score_universe(
        self,
        symbols: List[str],
        *,
        lookback_days: int = 2,
    ) -> List[Dict]:
        """Fetch news + classify + aggregate for every symbol.

        Returns one ``news_sentiment``-ready dict per symbol that has at
        least one headline. Symbols with no news are silently skipped
        (we don't write zero-headline rows — sparse table beats noise).
        """
        if not self._clf.ready:
            self.load()
        if not self._clf.ready:
            logger.info("SentimentEngine: classifier not ready, skipping")
            return []

        news_by_symbol = await fetch_many(symbols, lookback_days=lookback_days)
        logger.info(
            "SentimentEngine: fetched news for %d/%d symbols",
            len([v for v in news_by_symbol.values() if v]), len(symbols),
        )

        # Build one flat list of headlines with symbol-index tags so we
        # can run a single large classifier pass instead of per-symbol loops.
        flat_titles: List[str] = []
        origins: List[str] = []
        for sym, items in news_by_symbol.items():
            for it in items:
                flat_titles.append(it["title"])
                origins.append(sym)
        if not flat_titles:
            return []

        classifications = self._clf.classify_batch(flat_titles, batch_size=32)
        # Group classifications back by symbol.
        per_symbol: Dict[str, List[dict]] = {s: [] for s in news_by_symbol}
        for origin_sym, cls in zip(origins, classifications):
            per_symbol[origin_sym].append(cls)

        trade_date = date.today().isoformat()
        rows: List[Dict] = []
        for sym, cls_list in per_symbol.items():
            if not cls_list:
                continue
            rows.append(self._aggregate(sym, trade_date, news_by_symbol[sym], cls_list))
        return rows

    # ------------------------------------------------------------ aggregate

    def _aggregate(
        self,
        symbol: str,
        trade_date: str,
        headlines: List[Dict],
        classifications: List[Dict],
    ) -> Dict:
        pos = sum(1 for c in classifications if c["label"] == "positive")
        neg = sum(1 for c in classifications if c["label"] == "negative")
        neu = sum(1 for c in classifications if c["label"] == "neutral")
        n = len(classifications)
        mean_score = round(sum(c["score"] for c in classifications) / n, 4) if n else 0.0

        # Sample headlines we'll show on the signal detail page + F7 doctor.
        sample = []
        for h, c in list(zip(headlines, classifications))[:6]:
            sample.append({
                "title": h.get("title"),
                "source": h.get("source"),
                "published": h.get("published"),
                "label": c["label"],
                "score": c["score"],
            })
        sources = sorted({h.get("source") for h in headlines if h.get("source")})

        return {
            "symbol": symbol.upper(),
            "trade_date": trade_date,
            "mean_score": mean_score,
            "headline_count": n,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "sample_headlines": sample,
            "sources": sources,
        }

    # ------------------------------------------------------ single-symbol

    async def score_symbol(
        self, symbol: str, *, lookback_days: int = 2,
    ) -> Optional[Dict]:
        """Convenience single-symbol wrapper. Used by the stock dossier
        + portfolio doctor when a user wants a fresh score."""
        rows = await self.score_universe([symbol], lookback_days=lookback_days)
        return rows[0] if rows else None


# --------------------------------------------------------------- singleton

_engine: Optional[SentimentEngine] = None
_engine_lock = threading.Lock()


def get_sentiment_engine() -> SentimentEngine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = SentimentEngine()
    return _engine
