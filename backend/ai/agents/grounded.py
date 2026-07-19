"""Reusable grounded-reasoning agent — the cheap-but-deep core for AI features.

Pattern: assemble REAL deterministic facts upstream, hand them to a free
reasoning model here, get back a concise, grounded answer, cached by key. The
grounding is what lets a free model produce high-quality, non-hallucinated
output — so the same engine can power "why is X moving", the indicator
interpreter, volume narration, rebalancing rationale, journal mining, etc.,
everywhere, cheaply.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .response_cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_TTL_S = 6 * 3600               # a trading session; keys usually carry the date

_SYSTEM = (
    "You are a sharp Indian-equities trading analyst. You are given REAL, current "
    "market facts as JSON. Reason carefully over them, weigh which factors matter "
    "most, then answer in 3-5 tight sentences a retail trader understands. Ground "
    "EVERY claim in the provided facts — never invent prices, news or numbers. "
    "When you cite a price or a percentage move, use the EXACT figure present in "
    "the facts — never round, approximate, or state a magnitude (e.g. 'nearly 3%') "
    "that is not in the facts. If the facts are thin, say what the data does and "
    "doesn't show. State the directional read in plain words (bullish / bearish "
    "/ neutral) and put the exact figure next to each claim. Institutional desk "
    "voice — NO emoji or decorative symbols anywhere; emphasis via wording only. "
    "No preamble, no markdown headers."
)


def grounded_reason(
    facts: Dict[str, Any],
    question: str,
    *,
    cache_key: Optional[str] = None,
    role: str = "responder",
    system: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """Deep, grounded answer over `facts`. Free-first model, cached by
    `cache_key` (include the date so it expires daily). Returns None on
    failure so callers can fall back to the deterministic drivers."""
    if cache_key:
        hit = cache_get(cache_key)
        if hit and hit.get("answer"):
            return hit["answer"]
    try:
        from .llm import complete_sync
        prompt = (
            f"Question: {question}\n\nFacts (JSON):\n"
            f"{json.dumps(facts, default=str, ensure_ascii=False)}\n\nAnswer:"
        )
        ans = complete_sync(prompt, role=role, system=system or _SYSTEM,
                            temperature=0.3, feature="grounded_reason", user_id=user_id)
        ans = (ans or "").strip()
        # Guard against weak-model garbage — a model occasionally echoes a few
        # characters (e.g. "Rel" for a RELIANCE prompt) instead of the required
        # 3-5 sentences (seen live 2026-06-22 on /why-moving). A real grounded
        # narrative is always well over this floor; anything shorter is treated
        # as a failure so callers fall back to their deterministic drivers
        # instead of surfacing junk under a 200.
        if ans and len(ans) < 40:
            logger.debug("grounded_reason: discarding implausibly short answer %r", ans)
            ans = ""
        if ans and cache_key:
            cache_set(cache_key, {"answer": ans}, ttl_seconds=_TTL_S,
                      surface="grounded_reason", model="")
        return ans or None
    except Exception as e:
        logger.debug("grounded_reason failed: %s", e)
        return None
