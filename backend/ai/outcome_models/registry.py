"""Persistence + loading for outcome models — PR-MODELS.

Saves/loads trained XGBoost models per template_slug. v1: local
``artifacts/outcome/`` directory. v1.1: switch to B2 + model_registry table.

Each model is stored as a directory:
  artifacts/outcome/<slug>/
    model.json           — xgboost native format
    metadata.json        — feature_names, training stats, trained_at

Loaded via ``OutcomeModelRegistry.load(slug)`` at process start; caching
in-memory dict keyed by slug.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Local model directory — created on first save. In production this
# becomes the staging area; B2/S3 sync happens on a separate cron.
MODEL_DIR = Path(os.getenv("OUTCOME_MODELS_DIR", "artifacts/outcome"))


def save_outcome_model(slug: str, training_result: Any) -> Optional[Path]:
    """Persist a trained model + metadata to ``MODEL_DIR/<slug>/``.

    Returns the directory path, or None if no model was actually trained.
    """
    if not training_result.trained or training_result.model_blob is None:
        return None

    target_dir = MODEL_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        training_result.model_blob.save_model(str(target_dir / "model.json"))
    except Exception as exc:
        logger.warning("save_outcome_model %s: model.save_model failed: %s", slug, exc)
        return None

    metadata = {
        "slug": slug,
        "trained_at": training_result.trained_at,
        "n_samples": training_result.n_samples,
        "win_rate": float(training_result.win_rate),
        "auc": float(training_result.auc) if training_result.auc is not None else None,
        "feature_names": list(training_result.feature_names),
        "model_format": "xgboost_json",
    }
    try:
        with open(target_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as exc:
        logger.warning("save_outcome_model %s: metadata.json write failed: %s", slug, exc)

    return target_dir


def _load_one_model(slug: str) -> Optional[Dict[str, Any]]:
    """Try to load slug's model. Returns dict or None if missing/corrupt."""
    target_dir = MODEL_DIR / slug
    if not target_dir.exists():
        return None
    model_path = target_dir / "model.json"
    meta_path = target_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
    except Exception as exc:
        logger.warning("load_outcome_model %s: xgb load failed: %s", slug, exc)
        return None
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}
    return {"model": model, "metadata": metadata,
            "feature_names": metadata.get("feature_names", [])}


class OutcomeModelRegistry:
    """Process-wide cache of loaded outcome models. Singleton-ish."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _scanned: bool = False

    @classmethod
    def get(cls, slug: str) -> Optional["OutcomePredictor"]:
        """Get a predictor for ``slug``. Lazy-loads from disk on first call."""
        if slug in cls._cache:
            return OutcomePredictor(cls._cache[slug])
        loaded = _load_one_model(slug)
        if loaded is None:
            return None
        cls._cache[slug] = loaded
        return OutcomePredictor(loaded)

    @classmethod
    def all_loaded_slugs(cls) -> List[str]:
        return list(cls._cache.keys())

    @classmethod
    def scan_disk(cls) -> List[str]:
        """Eager scan: load all available models on disk into cache."""
        if cls._scanned:
            return cls.all_loaded_slugs()
        if not MODEL_DIR.exists():
            cls._scanned = True
            return []
        for sub in MODEL_DIR.iterdir():
            if sub.is_dir():
                loaded = _load_one_model(sub.name)
                if loaded is not None:
                    cls._cache[sub.name] = loaded
        cls._scanned = True
        return cls.all_loaded_slugs()

    @classmethod
    def invalidate(cls, slug: Optional[str] = None) -> None:
        """Drop cached model — caller can re-load with new training data."""
        if slug is None:
            cls._cache.clear()
        elif slug in cls._cache:
            del cls._cache[slug]


class OutcomePredictor:
    """Thin wrapper around a loaded XGBoost model + its feature list."""

    def __init__(self, loaded: Dict[str, Any]):
        self.model = loaded["model"]
        self.feature_names: List[str] = list(loaded.get("feature_names") or [])
        self.metadata: Dict[str, Any] = loaded.get("metadata") or {}

    def predict_win_proba(self, features: Dict[str, Any]) -> float:
        """P(win) for these features. Returns 0.5 on any failure."""
        try:
            import numpy as np
            row = np.array([[float(features.get(k, 0.0)) for k in self.feature_names]])
            return float(self.model.predict_proba(row)[0, 1])
        except Exception:
            return 0.5

    @property
    def auc(self) -> Optional[float]:
        return self.metadata.get("auc")

    @property
    def n_samples(self) -> int:
        return int(self.metadata.get("n_samples") or 0)
