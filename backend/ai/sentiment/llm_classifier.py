"""
LLM sentiment classifier — primary classifier for Quant X.

Per the locked deep-research decision (2026-05-10), FinBERT-class models
(including Vansh180/FinBERT-India-v1) are demoted because:

  1. FinDPO paper (arXiv:2507.18417, July 2025) — FinBERT Sharpe collapses
     to negative at realistic 5 bps trading costs.
  2. Vansh180/FinBERT-India-v1 is a fine-tune of `yiyanghkust/finbert-tone`
     on only 7,451 LLM-labeled headlines, self-reported F1=76.4%, no
     external benchmark.
  3. Multiple 2025 studies (FinSentLLM arXiv:2509.12638) show LLM zero-shot
     ≥ FinBERT on FPB. Since Quant X already runs an LLM for explanations,
     using it as the classifier consolidates the LLM stack.

Routing: every call goes through the shared OpenRouter gateway
(``ai/agents/llm.py`` → ``complete_sync(role="sentiment")``). No
single-provider SDK is used.

Drop-in API:
    Mirrors ``FinBERTIndia`` so existing callers in ``ai/sentiment``,
    ``ai/digest``, ``ai/agents``, ``ai/earnings/training/features`` switch
    without changing call sites:

        clf = LLMFinanceClassifier()
        clf.load()                                     # True / False
        results = clf.classify_batch(["TCS Q3 beats", "Reliance falls"])
        # [{"label": "positive", "probs": {...}, "score": 0.84}, ...]

Each result row contains:
    label   — "positive" | "neutral" | "negative"
    probs   — {"positive": float, "neutral": float, "negative": float}
    score   — P(positive) - P(negative) in [-1, +1]

Batching:
    The model handles ~20 headlines per call in <2s. We batch by 20
    via structured-output JSON arrays. Rate-limited to 10 req/min by
    default; override via LLM_RPM_LIMIT env.

Fallback:
    When ``OPENROUTER_API_KEY`` is unset or the client fails to init, ``load()``
    returns False and ``classify_batch`` returns []. Callers should
    check ``ready`` before invoking. The existing ``FinBERTIndia`` class
    remains importable for shadow / fallback use cases (see
    ``USE_FINBERT_FALLBACK=1`` env switch in engine.py).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "llm-sentiment"  # label only; actual model resolved by role="sentiment"
DEFAULT_BATCH_SIZE = 20
DEFAULT_RPM_LIMIT = 10

LABELS = ("positive", "neutral", "negative")
NEUTRAL_RESULT = {
    "label": "neutral",
    "probs": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
    "score": 0.0,
}


# Prompt template — explicit structured output. The model is asked to emit a
# JSON array; free models occasionally wrap it in fences, which we strip below.
_PROMPT_TEMPLATE = """You are a financial sentiment classifier for Indian NSE/BSE equities.

Classify each headline in the JSON array below into one of:
  - "positive": bullish for the named stock / sector / market
  - "neutral": informational, no clear directional implication
  - "negative": bearish for the named stock / sector / market

For each headline, also output a confidence in [0.0, 1.0] for the chosen label.

Return ONLY a JSON array of objects with this exact shape, in the same
order as the input. Do not include any text before or after the JSON.

Schema per object: {{"label": "positive"|"neutral"|"negative", "confidence": <float 0..1>}}

Input headlines:
{headlines_json}
"""


class LLMFinanceClassifier:
    """Drop-in replacement for FinBERTIndia using an open model's structured output.

    Thread-safe after ``load()``. Singleton via ``get_classifier()``.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        rpm_limit: Optional[int] = None,
    ):
        self.model_name = model
        self._api_key = api_key or ""  # legacy field; routing now via OpenRouter gateway
        self._rpm_limit = rpm_limit or int(os.environ.get("LLM_RPM_LIMIT", DEFAULT_RPM_LIMIT))
        self._client: Any = None
        self._init_done = False
        self._call_timestamps: List[float] = []

    # ------------------------------------------------------------- lifecycle

    @property
    def ready(self) -> bool:
        if not self._init_done:
            self.load()
        return self._client is not None

    def load(self) -> bool:
        """Ready iff OpenRouter is configured (no client to init — calls go
        through the shared LLM gateway)."""
        if self._init_done:
            return self._client is not None
        with self._lock:
            if self._init_done:
                return self._client is not None
            self._init_done = True
            from ...core.config import settings  # noqa: PLC0415
            self._client = True if settings.OPENROUTER_API_KEY else None
            if self._client is None:
                logger.info("OPENROUTER_API_KEY not set — sentiment classifier disabled")
            return self._client is not None

    # ------------------------------------------------------------- inference

    def classify_batch(
        self,
        texts: List[str],
        *,
        max_length: int = 256,  # accepted for FinBERT API parity; unused
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, float]]:
        """Score a list of headlines. Returns one result dict per input.

        Same output shape as ``FinBERTIndia.classify_batch``:
            [
              {"label": "positive|neutral|negative",
               "probs": {"positive": p_pos, "neutral": p_neu, "negative": p_neg},
               "score": p_pos - p_neg},
              ...
            ]

        On any per-batch failure the affected batch returns neutral results
        so a single bad call doesn't poison the entire pipeline.
        """
        if not texts:
            return []
        if not self.ready:
            return []

        results: List[Dict[str, float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = [str(t)[:max_length] for t in texts[i: i + batch_size]]
            try:
                self._respect_rate_limit()
                parsed = self._call_llm(chunk)
                results.extend(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.debug("sentiment batch %d failed: %s", i, exc)
                results.extend([dict(NEUTRAL_RESULT) for _ in chunk])
        return results

    # ------------------------------------------------------------- internals

    def _respect_rate_limit(self) -> None:
        """Sliding-window 1-minute rate limiter. Sleeps when needed."""
        now = time.time()
        cutoff = now - 60.0
        self._call_timestamps = [t for t in self._call_timestamps if t > cutoff]
        if len(self._call_timestamps) >= self._rpm_limit:
            oldest = min(self._call_timestamps)
            wait = max(0.0, 60.0 - (now - oldest)) + 0.1
            logger.debug("sentiment rate-limit: sleeping %.1fs", wait)
            time.sleep(wait)
        self._call_timestamps.append(time.time())

    def _call_llm(self, headlines: List[str]) -> List[Dict[str, float]]:
        """Single batched call via the open-model gateway. Returns one result
        per headline in order."""
        from ..agents.llm import complete_sync  # noqa: PLC0415

        headlines_json = json.dumps(headlines, ensure_ascii=False)
        prompt = _PROMPT_TEMPLATE.format(headlines_json=headlines_json)
        text = complete_sync(prompt, role="sentiment", temperature=0.0, feature="sentiment")
        if not text:
            raise RuntimeError("sentiment model returned empty response")

        # Free models may wrap the array in code fences or prose — extract it.
        t = text.strip()
        if t.startswith("```"):
            import re as _re  # noqa: PLC0415
            t = _re.sub(r"^```[a-zA-Z]*\n?", "", t)
            t = _re.sub(r"\n?```$", "", t).strip()
        try:
            parsed = json.loads(t)
        except Exception:  # noqa: BLE001
            s, e = t.find("["), t.rfind("]")
            parsed = json.loads(t[s: e + 1]) if 0 <= s < e else None
        if not isinstance(parsed, list):
            raise RuntimeError(f"sentiment model returned non-list response: {type(parsed)}")

        out: List[Dict[str, float]] = []
        for i, item in enumerate(parsed[: len(headlines)]):
            label = str(item.get("label", "neutral")).strip().lower()
            if label not in LABELS:
                label = "neutral"
            conf = float(item.get("confidence", 0.5))
            conf = min(max(conf, 0.0), 1.0)
            # Map (label, confidence) → 3-way probability distribution.
            #   chosen label gets `conf`; other two split the residual evenly.
            other = (1.0 - conf) / 2.0
            probs = {lbl: other for lbl in LABELS}
            probs[label] = conf
            out.append({
                "label": label,
                "probs": probs,
                "score": round(probs["positive"] - probs["negative"], 4),
            })

        # If the model returned fewer items than we sent, pad with neutral.
        while len(out) < len(headlines):
            out.append(dict(NEUTRAL_RESULT))
        return out


# --------------------------------------------------------------- singleton

_instance: Optional[LLMFinanceClassifier] = None
_instance_lock = threading.Lock()


def get_classifier() -> LLMFinanceClassifier:
    """Module-level singleton — share one client across the process."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = LLMFinanceClassifier()
    return _instance


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MODEL",
    "LLMFinanceClassifier",
    "LABELS",
    "NEUTRAL_RESULT",
    "get_classifier",
]
