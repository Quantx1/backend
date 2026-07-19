#!/usr/bin/env python
"""
One-shot migration: upload every existing ``artifacts/models/`` artifact to
Backblaze B2 as version 1, and insert ``model_versions`` rows with
correct is_prod / is_shadow flags.

Run once after PR 2 migration + PR 3 deploy::

    python scripts/upload_existing_models_to_b2.py

Idempotent: if a given model_name already has a row, it is skipped.
Passes --force to re-upload and overwrite metadata.

Registration plan (per PR-3 orphan audit in memory + Step 3 §4):

    regime_hmm             v1  is_prod=True   — live Day 4 (sound HMM)
    breakout_meta_labeler  v1  is_prod=True   — Scanner Lab only
    tft_swing              v1  is_shadow=True — under-parameterized, retrain Weeks 1-2
    lgbm_signal_gate       v1  is_shadow=True — inspect then likely retire
    quantai_ranker         v1  is_shadow=True — obsoleted by Qlib, will retire
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
logger = logging.getLogger("upload_existing_models")

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_MODELS = REPO_ROOT / "artifacts" / "models"

# Make ``backend`` importable when this script is run directly.
sys.path.insert(0, str(REPO_ROOT))

from backend.ai.registry import get_registry  # noqa: E402
from backend.ai.registry import versions as _versions  # noqa: E402


# Every entry describes one logical model: its registry name, the files
# (all sit under artifacts/models/), the desired status flag, and any metrics
# we know about the bundled v1 artifact.
MIGRATION_PLAN: List[Dict] = [
    {
        "name": "regime_hmm",
        "files": ["regime_hmm.pkl"],
        "is_prod": True,
        "is_shadow": False,
        "notes": (
            "3-state GaussianHMM (bull/sideways/bear). Mathematically "
            "sound — promotes to prod on Day 4 per PR-3 orphan audit."
        ),
        "metrics": {"model_type": "GaussianHMM", "n_states": 3},
    },
    {
        "name": "breakout_meta_labeler",
        "files": ["breakout_meta_labeler.pkl"],
        "is_prod": True,
        "is_shadow": False,
        "notes": (
            "RF 500x3 meta-labeler for pattern breakouts. Scanner Lab "
            "only — NOT wired into AI signal pipeline (Step 1 §3.1.1)."
        ),
        "metrics": {"model_type": "RandomForest", "n_estimators": 500, "max_depth": 3},
    },
    {
        "name": "tft_swing",
        "files": ["tft_model.ckpt", "tft_config.json", "tft_config.pt"],
        "is_prod": False,
        "is_shadow": True,
        "notes": (
            "TFT v1 — hidden_size=32, 100-stock universe, encoder=120, "
            "horizon=5. Under-parameterized for production; ships as "
            "shadow while v2 trains on Nifty 500 with hidden_size=128."
        ),
        "metrics": {
            "model_type": "TemporalFusionTransformer",
            "hidden_size": 32,
            "encoder_length": 120,
            "prediction_length": 5,
            "universe_size": 100,
        },
    },
    {
        "name": "lgbm_signal_gate",
        "files": ["lgbm_signal_gate.txt"],
        "is_prod": False,
        "is_shadow": True,
        "notes": (
            "LightGBM 3-class gate (HOLD/BUY/SELL) — 15MB booster is "
            "likely overfit. Shadow pending regression audit; expected "
            "to retire after TFT+Qlib ensemble proves alpha."
        ),
        "metrics": {"model_type": "LightGBMBooster", "num_classes": 3},
    },
    {
        "name": "quantai_ranker",
        "files": ["quantai_ranker.txt", "quantai_ranker_meta.json"],
        "is_prod": False,
        "is_shadow": True,
        "notes": (
            "2-week return ranker — obsoleted by Qlib Alpha158 + "
            "LightGBM path. Shadow-only, will retire in PR 7."
        ),
        "metrics": {"model_type": "LightGBMRegressor", "target": "2w_return"},
    },
    # ---- PR 9: Qlib Alpha158 + LightGBM cross-sectional ranker (NSE) ----
    # Artifacts live under artifacts/models/qlib_alpha158/ produced by
    # scripts/train_qlib_alpha158.py. Uploader runs --only qlib_alpha158
    # AFTER Rishi trains on Colab Pro.
    {
        "name": "qlib_alpha158",
        "files": ["qlib_alpha158/qlib_alpha158.txt", "qlib_alpha158/qlib_alpha158_meta.json"],
        "is_prod": False,
        "is_shadow": True,
        "notes": (
            "Alpha158 (158 technical factors, NSE-adapted port of Qlib) + "
            "LightGBM cross-sectional ranker. Trained on Nifty 500, 5y, 1y "
            "walk-forward OOS. Ships shadow; promotes to prod after IC + "
            "rank-IC regression gate (see train_qlib_alpha158.py)."
        ),
        "metrics": {"model_type": "LightGBMBooster", "label": "fwd_return_5d"},
    },
]


def _resolve_files(names: List[str]) -> List[Path]:
    paths: List[Path] = []
    for n in names:
        p = ML_MODELS / n
        if not p.exists():
            raise FileNotFoundError(f"Expected artifact missing: {p}")
        paths.append(p)
    return paths


def migrate_one(entry: Dict, force: bool, git_sha: Optional[str]) -> None:
    name = entry["name"]
    existing = _versions.list_versions(name)
    if existing and not force:
        logger.info(
            "SKIP %s — already has %d version row(s). Use --force to re-upload.",
            name, len(existing),
        )
        return

    files = _resolve_files(entry["files"])
    logger.info("UPLOAD %s ← %s", name, [f.name for f in files])

    registry = get_registry()
    row = registry.register(
        model_name=name,
        local_files=files,
        version=1 if not existing else None,
        metrics=entry.get("metrics"),
        trained_by="pr3-migration",
        git_sha=git_sha,
        notes=entry.get("notes"),
        is_prod=entry.get("is_prod", False),
        is_shadow=entry.get("is_shadow", False),
    )
    logger.info(
        "  ✓ %s v%s registered (prod=%s shadow=%s)",
        name, row["version"], row["is_prod"], row["is_shadow"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-upload even if rows already exist")
    parser.add_argument("--git-sha", default=None,
                        help="git commit sha to stamp on every row")
    parser.add_argument("--only", default=None,
                        help="migrate only this model_name (comma-separated ok)")
    args = parser.parse_args()

    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    plan = [e for e in MIGRATION_PLAN if not only or e["name"] in only]
    if not plan:
        logger.error("No models match --only=%s", args.only)
        sys.exit(2)

    logger.info("Running migration for %d model(s)", len(plan))
    for entry in plan:
        migrate_one(entry, force=args.force, git_sha=args.git_sha)
    logger.info("Done.")


if __name__ == "__main__":
    main()
