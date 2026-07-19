"""
Model registry for PRD pipeline.
Loads LightGBM and TFT models and provides inference helpers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Legacy v1 feature order — for the originally-shipped 15-column LGBM
# model that scripts/train/train_lgbm.py produced. Used only as the fallback
# when no sidecar metadata exists alongside the .txt artifact.
LGBM_LEGACY_FEATURE_ORDER = [
    "close", "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_percent",
    "ema_20", "ema_50", "atr_14",
    "volume_ratio", "obv", "vwap_diff",
    "body_pct", "wick_pct",
]
# Legacy v1 labels: scripts/train/train_lgbm.py used (0=HOLD, 1=BUY, 2=SELL).
# The v2 trainer (ml/training/trainers/lgbm_signal_gate.py) uses
# (0=SELL, 1=HOLD, 2=BUY) because _remap_labels_for_lgbm maps
# (-1,0,+1) → (0,1,2). Sidecar metadata disambiguates which is which.
LGBM_LEGACY_LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Kept as an exported alias for any older imports.
LGBM_FEATURE_ORDER = LGBM_LEGACY_FEATURE_ORDER


class LGBMGate:
    """
    LightGBM 3-class direction classifier.

    Loads label_map + feature_order from a sidecar
    ``<model_basename>.meta.json`` if present; otherwise falls back to the
    legacy v1 mapping (HOLD=0, BUY=1, SELL=2) and 15 features that match
    ``scripts/train/train_lgbm.py``.

    The v2 trainer (ml/training/trainers/lgbm_signal_gate.py) ALWAYS
    writes a sidecar, so any newly-trained model picks up its own schema
    automatically and we never silently invert signals on retrain.
    """

    def __init__(self, model_path: str):
        import lightgbm as lgb

        # Load via model_str (read text in Python) instead of model_file.
        # On Windows the native model_file parser can crash the whole
        # process ("Model format error, expect a tree here") on valid v4
        # models; the string loader is stable.
        with open(model_path, "r", encoding="utf-8") as _fh:
            self.model = lgb.Booster(model_str=_fh.read())
        self._num_classes = self.model.num_model_per_iteration()
        self._num_features = self.model.num_feature()

        # --- Sidecar metadata load ---
        sidecar_path = self._sidecar_path_for(model_path)
        meta = self._load_sidecar(sidecar_path)
        self._meta_source: str

        if meta:
            self._feature_order = list(meta.get("feature_order") or [])
            raw_map = meta.get("label_map") or {}
            try:
                self._label_map = {int(k): str(v).upper() for k, v in raw_map.items()}
            except Exception:
                self._label_map = dict(LGBM_LEGACY_LABEL_MAP)
            self._meta_source = f"sidecar:{sidecar_path}"
        else:
            self._feature_order = list(LGBM_LEGACY_FEATURE_ORDER)
            self._label_map = dict(LGBM_LEGACY_LABEL_MAP)
            self._meta_source = "legacy_fallback"

        # --- Hard runtime guard: schema-vs-artifact mismatch is fatal. ---
        # Pre-Phase-1.7 we silently zero-padded missing features through
        # features.get(k, 0.0) which produced garbage predictions on any
        # mismatched model. Refuse to load instead.
        if len(self._feature_order) != self._num_features:
            raise ValueError(
                f"LGBMGate: feature_order length ({len(self._feature_order)}) "
                f"does not match model.num_feature() ({self._num_features}). "
                f"meta_source={self._meta_source} — refusing to load to avoid "
                f"silent feature drift."
            )
        if not {0, 1, 2}.issubset(set(self._label_map.keys())):
            raise ValueError(
                f"LGBMGate: label_map must define classes 0/1/2, got "
                f"{self._label_map} (meta_source={self._meta_source})"
            )

        # Reverse index for probs_dict naming. Keys are lowercase strings.
        self._direction_to_class: Dict[str, int] = {
            v.lower(): k for k, v in self._label_map.items()
        }

        logger.info(
            "LGBMGate loaded from %s (%d classes, %d features, meta=%s, label_map=%s)",
            model_path, self._num_classes, self._num_features,
            self._meta_source, self._label_map,
        )

    @staticmethod
    def _sidecar_path_for(model_path: str) -> str:
        """`foo/bar/lgbm_signal_gate.txt` → `foo/bar/lgbm_signal_gate.meta.json`."""
        base, _ext = os.path.splitext(model_path)
        return base + ".meta.json"

    @staticmethod
    def _load_sidecar(path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LGBMGate: failed to parse sidecar %s: %s", path, exc)
            return None

    @property
    def feature_order(self) -> List[str]:
        """Feature names in the exact order the model expects."""
        return list(self._feature_order)

    def predict(self, features: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns (direction, confidence, probs_dict).
        direction: "BUY" / "SELL" / "HOLD"
        confidence: 0-100
        probs_dict: {"hold": ..., "buy": ..., "sell": ...} each 0-100

        ``features`` must contain every key in ``self.feature_order``.
        Missing keys raise — silent zero-fill caused the 2026-05-13 audit
        to flag every shipped prediction as undefined behaviour.
        """
        missing = [k for k in self._feature_order if k not in features]
        if missing:
            raise KeyError(
                f"LGBMGate.predict: missing required features {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} "
                f"(model expects {len(self._feature_order)} features)"
            )

        X = np.array(
            [[float(features[k]) if features[k] is not None else 0.0
              for k in self._feature_order]],
            dtype=np.float64,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw = self.model.predict(X)[0]  # shape (num_classes,) for multiclass

        if self._num_classes > 1 and len(raw) == self._num_classes:
            proba = _softmax(raw)
        else:
            proba = np.array(raw) if hasattr(raw, '__len__') else np.array([raw])
            if proba.sum() > 1.5:
                proba = _softmax(proba)

        best_class = int(np.argmax(proba))
        direction = self._label_map.get(best_class, "HOLD")
        confidence = float(proba[best_class]) * 100

        buy_idx = self._direction_to_class.get("buy")
        sell_idx = self._direction_to_class.get("sell")
        hold_idx = self._direction_to_class.get("hold")
        probs_dict = {
            "hold": float(proba[hold_idx]) * 100 if hold_idx is not None else 0.0,
            "buy": float(proba[buy_idx]) * 100 if buy_idx is not None else 0.0,
            "sell": float(proba[sell_idx]) * 100 if sell_idx is not None else 0.0,
        }
        return direction, confidence, probs_dict


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TFTPredictor:
    """
    Temporal Fusion Transformer for 5-bar price forecasting with quantile outputs.

    Loads lazily to avoid importing pytorch/pytorch_forecasting at module level.
    Provides per-stock prediction via ``predict_for_stock(df, symbol)``.
    """

    def __init__(self, model_path: str, config_path: str):
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        self._torch = torch
        self.TimeSeriesDataSet = TimeSeriesDataSet
        self._GroupNormalizer = GroupNormalizer

        # Load config (JSON for metadata, .pt for full dataset params)
        self.config: Dict = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception:
            pass

        # Load full dataset parameters (.pt) needed by from_parameters()
        self._dataset_params = None
        pt_path = config_path.replace(".json", ".pt")
        try:
            self._dataset_params = torch.load(pt_path, map_location="cpu", weights_only=False)  # nosec B614 - our own trusted TFT artifact from the model store
        except Exception as e:
            logger.warning(f"TFT .pt config not found ({e}), will use manual dataset creation")

        # Load the trained model checkpoint
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location="cpu")
        self.model.eval()

        self._features = self.config.get("features", [
            "close", "open", "high", "low", "volume",
            "rsi_14", "macd", "ema_20", "ema_50",
            "atr_14", "volume_ratio", "bb_percent",
        ])
        self._encoder_length = self.config.get("max_encoder_length", 120)
        self._prediction_length = self.config.get("max_prediction_length", 5)
        self._quantiles = self.config.get("quantiles", [0.1, 0.5, 0.9])

        logger.info(
            "TFTPredictor loaded: encoder=%d, horizon=%d, features=%d",
            self._encoder_length, self._prediction_length, len(self._features),
        )

    def predict_for_stock(self, df, symbol: str) -> Optional[Dict]:
        """
        Run TFT inference for a single stock.

        Args:
            df: DataFrame with OHLCV + indicator columns (at least ``self._encoder_length + _prediction_length`` rows).
            symbol: Stock ticker (e.g. "RELIANCE").

        Returns:
            Dict with keys: "p10", "p50", "p90" (each a list of floats for next N bars),
            "direction" ("bullish"/"bearish"/"neutral"), and "score" (0-1).
            Returns None if prediction fails.
        """
        try:
            from backend.ai.feature_engineering import compute_features

            # Compute features if needed
            featured = compute_features(df) if "rsi_14" not in df.columns else df.copy()

            # Check we have all required columns
            missing = [c for c in self._features if c not in featured.columns]
            if missing:
                logger.debug("TFT skip %s: missing %s", symbol, missing)
                return None

            subset = featured[self._features].copy()
            subset = subset.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            min_rows = self._encoder_length + self._prediction_length
            if len(subset) < min_rows:
                return None

            # Take the last chunk needed for one prediction
            subset = subset.tail(min_rows).reset_index(drop=True)
            subset["time_idx"] = np.arange(len(subset))
            subset["symbol"] = symbol

            for col in self._features:
                subset[col] = subset[col].astype(float)
            subset["time_idx"] = subset["time_idx"].astype(int)
            subset["symbol"] = subset["symbol"].astype(str)

            # Build a prediction dataset
            if self._dataset_params is not None:
                dataset = self.TimeSeriesDataSet.from_parameters(
                    self._dataset_params, subset,
                    predict=True, stop_randomization=True,
                )
            else:
                dataset = self.TimeSeriesDataSet(
                    subset,
                    time_idx="time_idx",
                    target="close",
                    group_ids=["symbol"],
                    max_encoder_length=self._encoder_length,
                    max_prediction_length=self._prediction_length,
                    time_varying_unknown_reals=self._features,
                    time_varying_known_reals=[],
                    static_categoricals=["symbol"],
                    target_normalizer=self._GroupNormalizer(
                        groups=["symbol"], transformation="softplus",
                    ),
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                    predict_mode=True,
                )

            loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            # Get quantile predictions — shape [batch, horizon, n_quantiles]
            preds_tensor = self.model.predict(loader, mode="quantiles", return_x=False)
            preds = preds_tensor[0].detach().cpu().numpy()  # [horizon, n_quantiles]

            q_map = {}
            for i, q in enumerate(self._quantiles):
                q_map[f"p{int(q * 100)}"] = [round(float(v), 2) for v in preds[:, i]]

            # Derive direction and score from median forecast
            current_close = float(subset["close"].iloc[-self._prediction_length - 1])
            median_forecast = preds[:, 1]  # p50 column
            final_predicted = float(median_forecast[-1])

            pct_change = (final_predicted - current_close) / current_close if current_close > 0 else 0
            if pct_change > 0.005:
                direction = "bullish"
            elif pct_change < -0.005:
                direction = "bearish"
            else:
                direction = "neutral"

            # Score: 0-1 where 1 = strong bullish
            score = max(0.0, min(1.0, 0.5 + pct_change * 10))

            return {
                **q_map,
                "direction": direction,
                "score": round(score, 4),
                "horizon": self._prediction_length,
                "current_close": round(current_close, 2),
                "predicted_close": round(final_predicted, 2),
            }

        except Exception as e:
            logger.debug("TFT prediction failed for %s: %s", symbol, e)
            return None
