"""
FinBERT-India adapter — ``Vansh180/FinBERT-India-v1`` from HuggingFace.

- 3-label head: {0: 'positive', 1: 'neutral', 2: 'negative'}
- Fine-tuned on Indian financial news (earnings commentary, RBI /
  SEBI announcements, sector moves). India-native — not a global
  FinBERT generalizing to IN.
- CPU inference: ~30ms per batched headline at max_length=128 on a
  modern Mac; ~3s to score 100 headlines in a batch of 32.

Convention for score aggregation (consumed by ``SentimentEngine`` and
downstream signal-enrichment):

    score_single = P(positive) - P(negative)    # in [-1, +1]
    label_single = argmax(probs)                # positive / neutral / negative

For one symbol on one day we average ``score_single`` across all
headlines → ``mean_score`` in ``news_sentiment``.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

HF_REPO = "Vansh180/FinBERT-India-v1"
LABELS = {0: "positive", 1: "neutral", 2: "negative"}


class FinBERTIndia:
    """Load-once, batch-classify. Thread-safe after ``load()``."""

    _lock = threading.Lock()

    def __init__(self, *, repo: str = HF_REPO, device: Optional[str] = None):
        # PR 211 — auto-detect GPU. Cuts FinBERT inference from ~30ms to
        # ~6ms per headline on RTX 4090; sentiment backfill 12 min → 3 min.
        if device is None:
            try:
                import torch  # noqa: PLC0415
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.repo = repo
        self.device = device
        self._tokenizer = None
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> bool:
        if self.ready:
            return True
        with self._lock:
            if self.ready:
                return True
            try:
                import torch  # noqa: F401
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
            except Exception as e:
                logger.info("transformers/torch missing (%s) — FinBERT-India disabled", e)
                return False
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.repo)  # nosec B615 - fixed public model repo we control
                model = AutoModelForSequenceClassification.from_pretrained(self.repo)  # nosec B615 - fixed public model repo we control
                # PyTorch inference mode — disables dropout + batchnorm training behavior.
                model.eval()  # noqa: S307 — PyTorch method, not builtin eval()
                # PR 211 — move to GPU when available
                try:
                    model = model.to(self.device)
                except Exception as exc:
                    logger.warning("FinBERT model.to(%s) failed: %s; staying on CPU", self.device, exc)
                    self.device = "cpu"
                self._model = model
                logger.info("FinBERT-India loaded from %s (device=%s)", self.repo, self.device)
                return True
            except Exception as e:
                logger.warning("FinBERT-India load failed: %s", e)
                return False

    # ------------------------------------------------------------- inference

    def classify_batch(
        self,
        texts: List[str],
        *,
        max_length: int = 128,
        batch_size: int = 32,
    ) -> List[Dict[str, float]]:
        """Score a list of texts. Returns one dict per input:

            {"label": "positive|neutral|negative",
             "probs": {"positive": 0.87, "neutral": 0.10, "negative": 0.03},
             "score": 0.84}

        Returns empty list if not ready or input is empty.
        """
        if not self.ready or not texts:
            return []

        import torch

        results: List[Dict[str, float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = [str(t) for t in texts[i: i + batch_size]]
            try:
                inputs = self._tokenizer(
                    chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                )
                # PR 211 — move inputs to model device so GPU path works
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            except Exception as e:
                logger.debug("FinBERT batch %d failed: %s", i, e)
                results.extend([_neutral_result() for _ in chunk])
                continue

            for row in probs:
                pos, neu, neg = float(row[0]), float(row[1]), float(row[2])
                best_idx = int(row.argmax())
                results.append({
                    "label": LABELS[best_idx],
                    "probs": {"positive": pos, "neutral": neu, "negative": neg},
                    "score": round(pos - neg, 4),
                })
        return results


def _neutral_result() -> Dict[str, float]:
    return {
        "label": "neutral",
        "probs": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        "score": 0.0,
    }


# --------------------------------------------------------------- singleton

_instance: Optional[FinBERTIndia] = None
_instance_lock = threading.Lock()


def get_finbert() -> FinBERTIndia:
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = FinBERTIndia()
    return _instance
