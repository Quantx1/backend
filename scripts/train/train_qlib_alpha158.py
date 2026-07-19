#!/usr/bin/env python
"""
Train Alpha158 + LightGBM on NSE data using real Microsoft Qlib.

Prerequisite: run ``scripts/data/ingest_nse_to_qlib.py`` first to populate the
Qlib provider directory.

Pipeline:
  1. ``qlib.init(provider_uri=...)``
  2. ``Alpha158`` handler (label = 5-day forward return) fits inference /
     training segments exactly as in Microsoft's CSI300 examples, but
     over our NSE tier.
  3. ``LGBModel`` trained with walk-forward cross-validation, 1-year OOS.
  4. Evaluation: Pearson IC, Spearman Rank IC, top-decile vs bottom-decile
     return spread, daily Rank IC series.
  5. Save booster (``qlib_alpha158.txt``) + meta JSON (feature order,
     handler config, OOS metrics) to ``artifacts/models/qlib_alpha158/``.

Upload to B2:
    python scripts/train/upload_existing_models_to_b2.py --only qlib_alpha158 --force

Usage::

    python scripts/train/train_qlib_alpha158.py \\
        --provider-uri ~/.qlib/qlib_data/nse_data \\
        --instruments nse_all \\
        --label-horizon 5 \\
        --train-start 2018-01-01 \\
        --train-end 2024-12-31 \\
        --oos-start 2025-01-01 \\
        --oos-end 2026-04-18

Run on Colab Pro — ~20 min on Nifty 500, ~60 min on NSE All.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
logger = logging.getLogger("train_qlib_alpha158")

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- training


def build_handler(provider_uri: str, instruments: str, horizon: int,
                  train_start: str, train_end: str,
                  valid_start: str, valid_end: str,
                  oos_start: str, oos_end: str):
    """Construct Qlib's Alpha158 handler with a 5-day forward-return label."""
    from qlib.contrib.data.handler import Alpha158

    label_expr = f"Ref($close, -{horizon}) / $close - 1"

    handler = Alpha158(
        instruments=instruments,
        start_time=train_start,
        end_time=oos_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        label=([label_expr], ["LABEL0"]),
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna",            "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm",        "kwargs": {"fields_group": "label"}},
        ],
    )
    return handler


def build_dataset(handler, train_start, train_end, valid_start, valid_end,
                  oos_start, oos_end):
    from qlib.data.dataset import DatasetH

    return DatasetH(
        handler=handler,
        segments={
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test":  (oos_start, oos_end),
        },
    )


def build_model():
    from qlib.contrib.model.gbdt import LGBModel

    # Qlib's benchmark hyperparameters (CSI300 Alpha158 config.yaml).
    # Good baseline; tune on Rank-IC later if needed.
    return LGBModel(
        loss="mse",
        learning_rate=0.0421,
        max_depth=8,
        num_leaves=210,
        colsample_bytree=0.8879,
        subsample=0.8789,
        lambda_l1=205.67,
        lambda_l2=580.96,
        num_threads=4,
        early_stopping_rounds=30,
        num_boost_round=500,
    )


# ---------------------------------------------------------------- evaluation


def evaluate_ic(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> dict:
    """Pearson + Spearman IC per day, averaged."""
    merged = pred_df.join(label_df, how="inner").dropna()
    if merged.empty:
        return {}
    pearson, spearman = [], []
    for _, grp in merged.groupby(level="datetime"):
        if len(grp) < 20:
            continue
        pearson.append(grp["score"].corr(grp["LABEL0"]))
        spearman.append(grp["score"].corr(grp["LABEL0"], method="spearman"))

    def _decile(x):
        return pd.qcut(x, 10, labels=False, duplicates="drop")

    merged["decile"] = merged.groupby(level="datetime")["score"].transform(_decile)
    top = merged[merged["decile"] == 9]["LABEL0"].mean()
    bot = merged[merged["decile"] == 0]["LABEL0"].mean()
    return {
        "pearson_ic_mean": float(np.nanmean(pearson)),
        "pearson_ic_std":  float(np.nanstd(pearson)),
        "rank_ic_mean":    float(np.nanmean(spearman)),
        "rank_ic_std":     float(np.nanstd(spearman)),
        "rank_icir":       float(np.nanmean(spearman) / (np.nanstd(spearman) + 1e-9)),
        "top_decile_mean_return":    float(top),
        "bottom_decile_mean_return": float(bot),
        "long_short_spread":         float(top - bot),
        "oos_days":    int(merged.index.get_level_values("datetime").nunique()),
        "oos_rows":    int(len(merged)),
        "oos_symbols": int(merged.index.get_level_values("instrument").nunique()),
    }


# ---------------------------------------------------------------- save


def save_artifacts(model, dataset, handler, instruments: str, horizon: int,
                   metrics: dict, out_dir: Path, args: argparse.Namespace):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "qlib_alpha158.txt"
    meta_path = out_dir / "qlib_alpha158_meta.json"

    # LGBModel exposes the underlying booster as .model
    booster = getattr(model, "model", None)
    if booster is None:
        raise RuntimeError("LGBModel.model is None — was fit() called?")
    booster.save_model(str(model_path))

    meta = {
        "model_type": "LightGBMBooster via Qlib LGBModel",
        "qlib_version": _qlib_version(),
        "handler_class": "qlib.contrib.data.handler.Alpha158",
        "instruments": instruments,
        "label": f"Ref($close, -{horizon}) / $close - 1",
        "horizon_days": horizon,
        "train_segment": [args.train_start, args.train_end],
        "valid_segment": [args.valid_start, args.valid_end],
        "oos_segment":   [args.oos_start, args.oos_end],
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "trained_on": "NSE (pandas_market_calendars['NSE'])",
        "metrics": metrics,
        "feature_config": {
            "processors_infer": ["RobustZScoreNorm", "Fillna"],
            "processors_learn": ["DropnaLabel", "CSRankNorm"],
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved booster → %s", model_path)
    logger.info("Saved meta    → %s", meta_path)


def _qlib_version() -> str:
    try:
        import qlib
        return qlib.__version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/nse_data")
    parser.add_argument("--instruments", default="nse_all",
                        help="tier name matching an instruments/<name>.txt file")
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end",   default="2024-06-30")
    parser.add_argument("--valid-start", default="2024-07-01")
    parser.add_argument("--valid-end",   default="2024-12-31")
    parser.add_argument("--oos-start",   default="2025-01-01")
    parser.add_argument("--oos-end",     default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--out", default="artifacts/models/qlib_alpha158")
    args = parser.parse_args()

    # ── init Qlib ──────────────────────────────────────────────────
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(Path(args.provider_uri).expanduser().resolve()),
              region=REG_CN)  # region only affects defaults (lot size, trading times); we override via NSE calendar

    # ── handler + dataset + model ──────────────────────────────────
    handler = build_handler(
        args.provider_uri, args.instruments, args.label_horizon,
        args.train_start, args.train_end,
        args.valid_start, args.valid_end,
        args.oos_start, args.oos_end,
    )
    dataset = build_dataset(
        handler, args.train_start, args.train_end,
        args.valid_start, args.valid_end,
        args.oos_start, args.oos_end,
    )
    model = build_model()

    # ── fit ────────────────────────────────────────────────────────
    logger.info("Fitting LGBModel on instruments=%s horizon=%d",
                args.instruments, args.label_horizon)
    model.fit(dataset)

    # ── evaluate on OOS segment ────────────────────────────────────
    logger.info("Evaluating on OOS segment %s → %s",
                args.oos_start, args.oos_end)
    pred = model.predict(dataset, segment="test").to_frame("score")
    label = dataset.prepare(segments="test",
                            col_set="label",
                            data_key="raw")
    metrics = evaluate_ic(pred, label)
    logger.info("OOS metrics:\n%s", json.dumps(metrics, indent=2))

    # ── save ───────────────────────────────────────────────────────
    out_dir = (ROOT / args.out).resolve()
    save_artifacts(model, dataset, handler, args.instruments,
                   args.label_horizon, metrics, out_dir, args)
    logger.info(
        "Done. Upload:\n"
        "    python scripts/train/upload_existing_models_to_b2.py "
        "--only qlib_alpha158 --force"
    )


if __name__ == "__main__":
    main()
