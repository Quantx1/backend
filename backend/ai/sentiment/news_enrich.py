"""
Enriched news classification — one free-model pass that adds an EVENT-TYPE
taxonomy + MATERIALITY (impact) to the existing 3-way sentiment.

The base ``LLMFinanceClassifier`` answers only "bullish/bearish/neutral". A
state-of-the-art news desk also answers "what KIND of event is this" and "how
much does it MATTER". This module extends the exact same structured-output
pattern (single batched JSON-array call, fence-stripping, neutral-pad on
failure) so it rides the same FREE fast model, the same $50 budget kill-switch
and the same RPM limiter — no new model, no new cost.

Degrades honestly: if the LLM is unavailable it returns sentiment-only rows
(via the base classifier) with event_type='other', impact='low', so callers
never break — they just lose the taxonomy until the LLM is back.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

# Closed event taxonomy — the model MUST pick one (anything else → 'other').
EVENT_TYPES = (
    "earnings", "guidance", "mna", "rating_change", "management",
    "regulatory", "legal", "order_win", "product", "dividend_buyback",
    "ownership", "macro", "analyst", "other",
)
_EVENT_LABELS = {
    "earnings": "Earnings", "guidance": "Guidance", "mna": "M&A",
    "rating_change": "Rating change", "management": "Management",
    "regulatory": "Regulatory", "legal": "Legal", "order_win": "Order win",
    "product": "Product", "dividend_buyback": "Dividend/Buyback",
    "ownership": "Ownership/Stake", "macro": "Macro", "analyst": "Analyst",
    "other": "General",
}

IMPACT_LEVELS = ("high", "medium", "low")
# Materiality weight per impact level — used in the weighted mood aggregate.
IMPACT_WEIGHT = {"high": 1.0, "medium": 0.55, "low": 0.25}

LABELS = ("positive", "neutral", "negative")


def event_label(event_type: str) -> str:
    return _EVENT_LABELS.get(event_type, "General")


_PROMPT_TEMPLATE = """You are a financial news analyst for Indian NSE/BSE equities.

For EACH headline in the JSON array, output three fields:
  1. "sentiment": "positive" (bullish) | "neutral" | "negative" (bearish) for the named stock.
  2. "event_type": ONE of {event_types} — the kind of corporate/market event.
     earnings=results; guidance=outlook/forecast; mna=merger/acquisition/stake-sale;
     rating_change=credit/broker upgrade-downgrade; management=CEO/CFO/board change;
     regulatory=SEBI/RBI/govt action; legal=lawsuit/probe/penalty; order_win=contract/order;
     product=launch/expansion; dividend_buyback=dividend/buyback/bonus/split;
     ownership=promoter/FII/block-deal stake change; macro=sector/economy-wide;
     analyst=broker note/target; other=anything else.
  3. "impact": "high" (price-moving, material) | "medium" | "low" (routine/noise).

Return ONLY a JSON array of objects, same order as input, no prose, no fences.
Schema per object: {{"sentiment": "positive"|"neutral"|"negative", "event_type": "<one enum>", "impact": "high"|"medium"|"low"}}

Input headlines:
{headlines_json}
"""


def _neutral_row() -> Dict[str, Any]:
    return {
        "label": "neutral", "score": 0.0, "confidence": 0.0,
        "event_type": "other", "impact": "low",
    }


def _coerce_row(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return _neutral_row()
    label = str(item.get("sentiment") or item.get("label") or "neutral").strip().lower()
    if label not in LABELS:
        label = "neutral"
    et = str(item.get("event_type") or "other").strip().lower()
    if et not in EVENT_TYPES:
        et = "other"
    impact = str(item.get("impact") or "low").strip().lower()
    if impact not in IMPACT_LEVELS:
        impact = "low"
    # score: deterministic from label sign × impact magnitude (no fabricated probs).
    sign = 1.0 if label == "positive" else -1.0 if label == "negative" else 0.0
    return {
        "label": label,
        "score": round(sign * IMPACT_WEIGHT[impact], 4),
        "confidence": IMPACT_WEIGHT[impact],
        "event_type": et,
        "impact": impact,
    }


def _strip_to_array(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        s, e = t.find("["), t.rfind("]")
        return json.loads(t[s:e + 1]) if 0 <= s < e else None


def enrich_headlines(titles: Sequence[str], *, batch_size: int = 20) -> List[Dict[str, Any]]:
    """Per-headline {label, score, confidence, event_type, impact}.

    One batched free-model call per ``batch_size`` titles. On ANY failure the
    affected batch falls back to the base sentiment classifier (sentiment-only,
    event_type='other'/impact='low') so the pipeline degrades, never breaks.
    """
    titles = [str(t)[:256] for t in titles if str(t).strip()]
    if not titles:
        return []

    out: List[Dict[str, Any]] = []
    for i in range(0, len(titles), batch_size):
        chunk = titles[i:i + batch_size]
        rows = _enrich_chunk(chunk)
        out.extend(rows)
    return out


def _enrich_chunk(chunk: List[str]) -> List[Dict[str, Any]]:
    try:
        from ..agents.llm import complete_sync  # noqa: PLC0415
        prompt = _PROMPT_TEMPLATE.format(
            event_types=", ".join(EVENT_TYPES),
            headlines_json=json.dumps(chunk, ensure_ascii=False),
        )
        text = complete_sync(prompt, role="sentiment", temperature=0.0, feature="news_enrich")
        parsed = _strip_to_array(text) if text else None
        if not isinstance(parsed, list):
            raise RuntimeError("enrich model returned non-list")
        rows = [_coerce_row(it) for it in parsed[: len(chunk)]]
        while len(rows) < len(chunk):
            rows.append(_neutral_row())
        return rows
    except Exception as exc:  # noqa: BLE001
        logger.debug("news_enrich chunk failed (%s) — falling back to base sentiment", exc)
        return _fallback_sentiment(chunk)


def _fallback_sentiment(chunk: List[str]) -> List[Dict[str, Any]]:
    """Sentiment-only degrade path via the base classifier (or neutral)."""
    try:
        from .llm_classifier import get_classifier  # noqa: PLC0415
        clf = get_classifier()
        if clf.ready:
            base = clf.classify_batch(chunk)
            rows = []
            for b in base:
                rows.append({
                    "label": b.get("label", "neutral"),
                    "score": float(b.get("score", 0.0)),
                    "confidence": abs(float(b.get("score", 0.0))),
                    "event_type": "other",
                    "impact": "medium" if abs(float(b.get("score", 0.0))) >= 0.4 else "low",
                })
            while len(rows) < len(chunk):
                rows.append(_neutral_row())
            return rows
    except Exception:  # noqa: BLE001
        pass
    return [_neutral_row() for _ in chunk]


__all__ = [
    "EVENT_TYPES", "IMPACT_LEVELS", "IMPACT_WEIGHT", "LABELS",
    "enrich_headlines", "event_label",
]
