"""
Assistant service package — what survives the 2026-07-11 chat unification.

The legacy finance-assistant chat brain (AssistantService, DomainGuard,
MarketContextBuilder, AssistantLLM) was deleted: the Copilot graph
(backend/ai/agents/copilot.py) is the single brain. Two pieces remain in use:

- AssistantCreditLimiter — the daily chat-credit window shared with the
  copilot cap enforcement (middleware/tier_gate.py, /api/assistant/usage).
- NewsContextService — Indian-market news fetch, consumed by the sentiment
  pipeline (backend/ai/sentiment/news_providers.py).
"""

from .credit_limiter import AssistantCreditLimiter, CreditUsage
from .news_context import NewsArticle, NewsContextService

__all__ = [
    "AssistantCreditLimiter",
    "CreditUsage",
    "NewsArticle",
    "NewsContextService",
]
