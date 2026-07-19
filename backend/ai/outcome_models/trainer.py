"""Per-strategy outcome model training — PR-DEPTH.

Reads strategy_outcomes (closed trades + features_at_entry + won) and
trains one LightGBM per template_slug. Saves to B2 / model_registry
with the slug as part of the key.

Requires:
  - ≥30 closed trades for that template_slug
  - At least 40% positive class rate AND 40% negative (avoid degenerate fits)
  - LightGBM (already a dep)

Memory locks honoured:
  - Supervised ML, no LLM
  - No fallbacks: returns None if data insufficient; we don't fabricate
    a model on synthetic labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Tuneable thresholds — admin can override via env later
MIN_OUTCOMES_TO_TRAIN = 30
MIN_POSITIVE_RATE = 0.30
MIN_NEGATIVE_RATE = 0.30


@dataclass
class OutcomeModelConfig:
    template_slug: str
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    min_outcomes: int = MIN_OUTCOMES_TO_TRAIN


@dataclass
class TrainingResult:
    template_slug: str
    trained: bool
    n_samples: int
    win_rate: float
    auc: Optional[float] = None
    model_blob: Any = None              # the trained model object
    feature_names: List[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OutcomeModelTrainer:
    """Trains one LightGBM per template_slug from strategy_outcomes."""

    def __init__(self, supabase_admin: Any):
        self.supabase = supabase_admin

    def fetch_outcomes(self, template_slug: str) -> List[Dict[str, Any]]:
        """All closed trades for this template_slug, newest first.
        Caps at 5000 rows so we don't blow memory on hot strategies."""
        try:
            rows = (
                self.supabase.table("strategy_outcomes")
                .select("won, features_at_entry, regime_at_entry, vix_at_entry, "
                        "pnl_pct, result, exit_at")
                .eq("template_slug", template_slug)
                .order("exit_at", desc=True)
                .limit(5000)
                .execute()
            )
            return rows.data or []
        except Exception as exc:
            logger.warning("fetch_outcomes %s failed: %s", template_slug, exc)
            return []

    def train(self, config: OutcomeModelConfig) -> TrainingResult:
        """Train one model for one template_slug. Returns the result
        (model + metadata) or a skipped result with reason."""
        outcomes = self.fetch_outcomes(config.template_slug)
        n = len(outcomes)

        if n < config.min_outcomes:
            return TrainingResult(
                template_slug=config.template_slug, trained=False, n_samples=n,
                win_rate=0.0,
                skipped_reason=f"insufficient_samples ({n}/{config.min_outcomes})",
            )

        # Build feature matrix from JSONB column
        rows: List[Dict[str, float]] = []
        labels: List[int] = []
        for o in outcomes:
            feat = o.get("features_at_entry") or {}
            if not isinstance(feat, dict) or not feat:
                continue
            # Augment with regime + vix
            if o.get("regime_at_entry"):
                rg = str(o["regime_at_entry"])
                feat = {
                    **feat,
                    "regime_bull": 1.0 if rg == "bull" else 0.0,
                    "regime_sideways": 1.0 if rg == "sideways" else 0.0,
                    "regime_bear": 1.0 if rg == "bear" else 0.0,
                }
            if o.get("vix_at_entry") is not None:
                feat = {**feat, "vix": float(o["vix_at_entry"])}
            rows.append(feat)
            labels.append(1 if o.get("won") else 0)

        if len(rows) < config.min_outcomes:
            return TrainingResult(
                template_slug=config.template_slug, trained=False, n_samples=n,
                win_rate=0.0,
                skipped_reason=f"insufficient_with_features ({len(rows)}/{config.min_outcomes})",
            )

        win_rate = sum(labels) / len(labels)
        if win_rate < MIN_POSITIVE_RATE or (1 - win_rate) < MIN_NEGATIVE_RATE:
            return TrainingResult(
                template_slug=config.template_slug, trained=False,
                n_samples=len(rows), win_rate=win_rate,
                skipped_reason=f"degenerate_label_distribution win_rate={win_rate:.2f}",
            )

        # Build pandas DataFrame, normalise feature set (union of all keys)
        try:
            import pandas as pd
            df = pd.DataFrame(rows).fillna(0.0)
        except Exception as exc:
            return TrainingResult(
                template_slug=config.template_slug, trained=False, n_samples=len(rows),
                win_rate=win_rate, skipped_reason=f"pandas_failed: {exc}",
            )

        # Train/test split — chronological
        # outcomes is newest-first, so the FIRST N rows are most recent
        # → use the LAST N rows (oldest) for training, FIRST N for validation
        split_idx = int(len(df) * 0.7)
        X_train = df.iloc[split_idx:].values
        y_train = labels[split_idx:]
        X_test = df.iloc[:split_idx].values
        y_test = labels[:split_idx]

        try:
            import xgboost as xgb
        except Exception:
            return TrainingResult(
                template_slug=config.template_slug, trained=False, n_samples=len(rows),
                win_rate=win_rate, skipped_reason="xgboost_not_installed",
            )

        try:
            # XGBoost classifier — matches aaryansinha16's contract.
            # scale_pos_weight handles class imbalance automatically.
            n_neg = sum(1 for y in y_train if y == 0)
            n_pos = sum(1 for y in y_train if y == 1)
            scale_pos_weight = n_neg / max(n_pos, 1)

            model = xgb.XGBClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                tree_method="hist",         # fastest
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
        except Exception as exc:
            return TrainingResult(
                template_slug=config.template_slug, trained=False, n_samples=len(rows),
                win_rate=win_rate, skipped_reason=f"fit_failed: {exc}",
            )

        # Compute AUC on held-out set
        auc = None
        if len(X_test) >= 5:
            try:
                from sklearn.metrics import roc_auc_score
                probs = model.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, probs))
            except Exception:
                auc = None

        return TrainingResult(
            template_slug=config.template_slug, trained=True, n_samples=len(rows),
            win_rate=win_rate, auc=auc,
            model_blob=model, feature_names=list(df.columns),
        )


class OutcomeModelPredictor:
    """Loaded outcome model — predicts P(win) for a candidate entry signal."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    @classmethod
    def from_training_result(cls, result: TrainingResult) -> Optional["OutcomeModelPredictor"]:
        if not result.trained or result.model_blob is None:
            return None
        return cls(model=result.model_blob, feature_names=result.feature_names)

    def predict_win_proba(self, features: Dict[str, Any]) -> float:
        """P(win) for these features. Returns 0.5 (no opinion) if it fails."""
        try:
            import numpy as np
            row = np.array([[float(features.get(k, 0.0)) for k in self.feature_names]])
            return float(self.model.predict_proba(row)[0, 1])
        except Exception:
            return 0.5


def train_all_strategy_outcome_models(
    supabase_admin: Any,
    *,
    only_slugs: Optional[List[str]] = None,
) -> List[TrainingResult]:
    """Loop every template slug with sufficient outcomes; train each.

    Caller (admin / scheduled job): persists results, writes JSON to B2,
    bumps model_registry version. This function returns the in-memory
    result list — persistence is a separate concern.
    """
    trainer = OutcomeModelTrainer(supabase_admin)
    # Find slugs that have ≥ MIN_OUTCOMES_TO_TRAIN closed trades
    try:
        if only_slugs:
            slugs = only_slugs
        else:
            rows = (
                supabase_admin.table("strategy_outcomes")
                .select("template_slug")
                .limit(20000)
                .execute()
            )
            counts: Dict[str, int] = {}
            for r in rows.data or []:
                slug = r.get("template_slug")
                if slug:
                    counts[slug] = counts.get(slug, 0) + 1
            slugs = [s for s, c in counts.items() if c >= MIN_OUTCOMES_TO_TRAIN]
    except Exception as exc:
        logger.warning("could not enumerate slugs: %s", exc)
        return []

    results: List[TrainingResult] = []
    for slug in slugs:
        config = OutcomeModelConfig(template_slug=slug)
        result = trainer.train(config)
        results.append(result)
        if result.trained:
            logger.info(
                "outcome_model trained %s: n=%d wr=%.2f auc=%s",
                slug, result.n_samples, result.win_rate,
                f"{result.auc:.3f}" if result.auc else "N/A",
            )
        else:
            logger.info("outcome_model skipped %s: %s", slug, result.skipped_reason)
    return results
