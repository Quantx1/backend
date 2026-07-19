"""
Quant X sentiment package — F1/F7/F9/F12 feature source.

The primary classifier is an **open model zero-shot** prompt served through
the OpenRouter gateway via ``LLMFinanceClassifier``. The ``FinBERTIndia``
class remains importable for the ``USE_FINBERT_FALLBACK=1`` shadow path and
dev environments without OpenRouter API access.

This is the **batch** pipeline that feeds the ``news_sentiment`` table.
Real-time single-symbol scoring goes through ``SentimentEngine`` in this
package as well — there is no longer a separate ``services/sentiment_engine``
module.

Public API::

    from backend.ai.sentiment import (
        SentimentEngine, get_sentiment_engine, fetch_headlines,
        LLMFinanceClassifier, get_classifier,             # primary
        FinBERTIndia, get_finbert,                         # fallback
    )

    engine = get_sentiment_engine()
    engine.load()
    rows = engine.score_universe(['RELIANCE', 'TCS', ...])  # news_sentiment rows
"""

from .engine import SentimentEngine, get_sentiment_engine
from .finbert_india import FinBERTIndia, get_finbert
from .llm_classifier import LLMFinanceClassifier, get_classifier
from .news_fetcher import fetch_headlines

__all__ = [
    "FinBERTIndia",
    "LLMFinanceClassifier",
    "SentimentEngine",
    "fetch_headlines",
    "get_classifier",
    "get_finbert",
    "get_sentiment_engine",
]
