#!/usr/bin/env python
"""
Per-trainer accuracy validator.

Runs ONE trainer with a mid-tier universe (default 50 stocks, 5y) and
emits a rich metric report + a clean PASS/FAIL verdict based on the
documented quality thresholds for that trainer. Designed so we can
diagnose accuracy issues one model at a time BEFORE the full RunPod
training run burns money on a broken trainer.

Usage:
    python scripts/train/validate_trainer.py regime_hmm
    python scripts/train/validate_trainer.py lgbm_signal_gate --universe 50
    python scripts/train/validate_trainer.py tft_swing --period 5y --debug
    python scripts/train/validate_trainer.py --list

Workflow:
    1. On RunPod, run each trainer individually via this script
    2. Inspect metrics + verdict
    3. Fix anything red BEFORE committing to the full --all run

Why mid-tier and not full universe?
    Full universe (200 stocks, 8y) takes ~10 min per trainer for LightGBM
    and ~90 min for TFT. Mid-tier (50 stocks, 5y) hits every code path at
    production scale while keeping each trainer to ~2-10 minutes — fast
    iteration on accuracy issues.

Quality thresholds:
    Documented per-trainer in QUALITY_THRESHOLDS below. Each entry maps
    a metric name to (low, high) bounds; None means "no bound on that side".
    A trainer PASSES when every metric in its threshold dict satisfies its
    bounds. Missing metrics are FAILures.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("validate_trainer")


# ============================================================================
# Quality thresholds — per trainer
# ============================================================================
#
# Each entry: { metric_name: (lo, hi) }
# lo=None  → no lower bound;  hi=None → no upper bound.
# A trainer passes iff ALL metrics are within their bounds.
# Missing metrics → FAIL (the trainer didn't even produce them).
#
# Thresholds reflect what "good enough to ship" looks like on a mid-tier
# 50-stock validation run. The full 200-stock run typically tightens these.

QUALITY_THRESHOLDS: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {

    # ---------------- regime + ranker ----------------
    "regime_hmm": {
        "log_likelihood_per_obs_mean": (-15.0, 0.0),   # >-15 sanity, <=0
        "n_folds_succeeded":           (3,     None),  # at least 3 folds
    },
    "lgbm_signal_gate": {
        # Trainer emits 'sharpe', 'accuracy', 'max_drawdown_pct'.
        # W4 (pre-training audit 2026-05-19) — tightened smoke gates:
        # sharpe 0.3 → 0.5 (must beat "just doesn't blow up"),
        # accuracy 0.35 → 0.38 (3-class random = 0.33; 0.38 catches
        # marginal-edge signal even on a 50-symbol smoke universe).
        "sharpe":           (0.5,   None),
        "max_drawdown_pct": (-0.40, 0.0),
        "accuracy":         (0.38,  None),
    },
    "qlib_alpha158": {
        # Cross-sectional ranker — Rank IC is the right metric. 0.02
        # is the published "tradeable" threshold; full target 0.04+.
        # W4 — tightened rank_ic_mean 0.005 → 0.015. Anything below
        # 0.015 on a 5y daily ranker means the data pipeline is broken
        # (Qlib Alpha158 routinely hits 0.04 on CSI300).
        "rank_ic_mean":  (0.015, None),
        "rank_icir":     (0.30,  None),
    },

    # ---------------- forecasters ----------------
    "tft_swing": {
        # Trainer emits 'pinball_loss_mean', 'directional_accuracy',
        # 'backend' (NOT 'best_val_loss'). Pinball loss bound is loose
        # for smoke since universe is tiny.
        "pinball_loss_mean":     (None, 10.0),
    },

    # earnings_xgb removed 2026-05-11 — F9 EarningsScout deferred.
    # No real labeled data pre-launch; synthetic-only smoke wasn't worth
    # the maintenance overhead. Re-add once we have either:
    #   1. >=50 labeled rows in Supabase earnings_predictions, OR
    #   2. an NSE earnings-calendar scraper that labels historical surprises.
}


# ============================================================================
# Trainer execution
# ============================================================================


def _set_validation_env(universe: int, period: str, timesteps: int, epochs: int) -> None:
    """Configure SMOKE_MODE knobs to validation-tier values."""
    os.environ["SMOKE_MODE"] = "1"
    os.environ["SMOKE_UNIVERSE_SIZE"] = str(universe)
    os.environ["SMOKE_YFINANCE_PERIOD"] = period
    os.environ["SMOKE_TIMESTEPS"] = str(timesteps)
    os.environ["SMOKE_EPOCHS"] = str(epochs)
    os.environ["QLIB_USE_LAMBDARANK_IC"] = "1"


def list_trainers() -> None:
    from ml.training.discovery import discover_sorted
    trainers = discover_sorted()
    print(f"{len(trainers)} trainers discovered:\n")
    for t in trainers:
        gpu = "GPU" if t.requires_gpu else "CPU"
        thr = " ".join(QUALITY_THRESHOLDS.get(t.name, {}).keys()) or "(no thresholds)"
        print(f"  {t.name:<22}  {gpu}    thresholds: {thr}")


def run_one_trainer(name: str, debug: bool = False) -> Dict[str, Any]:
    """Train + evaluate one trainer. Return its full metrics dict + status."""
    from ml.training.discovery import discover_sorted

    trainers = {t.name: t for t in discover_sorted()}
    if name not in trainers:
        raise SystemExit(f"Unknown trainer: {name}\nAvailable: {sorted(trainers)}")
    trainer = trainers[name]

    with tempfile.TemporaryDirectory(prefix=f"validate_{name}_") as tmpdir:
        out = Path(tmpdir)
        start = time.time()
        train_result = None
        eval_metrics: Dict[str, Any] = {}
        error: Optional[str] = None
        try:
            train_result = trainer.train(out)
            if hasattr(trainer, "evaluate"):
                eval_metrics = trainer.evaluate(train_result) or {}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if debug:
                traceback.print_exc()
        elapsed = time.time() - start

    if error:
        return {
            "trainer": name,
            "status": "failed",
            "elapsed_s": round(elapsed, 1),
            "error": error,
            "metrics": {},
        }

    artifacts = [p.name for p in (train_result.artifacts or [])] if train_result else []
    metrics = {**(train_result.metrics if train_result else {}), **eval_metrics}
    return {
        "trainer": name,
        "status": "ok",
        "elapsed_s": round(elapsed, 1),
        "artifacts": artifacts,
        "metrics": metrics,
        "notes": (train_result.notes if train_result else None),
    }


# ============================================================================
# Verdict
# ============================================================================


def _format_bound(lo, hi) -> str:
    parts = []
    if lo is not None:
        parts.append(f">={lo}")
    if hi is not None:
        parts.append(f"<={hi}")
    return " ".join(parts) if parts else "any"


def render_verdict(report: Dict[str, Any]) -> int:
    """Pretty-print the report + threshold check. Return exit code."""
    name = report["trainer"]
    print("=" * 76)
    print(f"VALIDATION REPORT — {name}")
    print("=" * 76)
    print(f"  Status:   {report['status']}")
    print(f"  Elapsed:  {report['elapsed_s']}s")
    if report.get("artifacts"):
        print(f"  Artifacts: {report['artifacts']}")
    if report.get("notes"):
        print(f"  Notes:    {report['notes'][:120]}")

    if report["status"] == "failed":
        print(f"\n  ERROR: {report['error']}")
        print(f"\n  VERDICT: ❌ FAILED — fix this trainer before any full run")
        return 1

    metrics = report["metrics"] or {}
    if metrics:
        print(f"\n  Metrics:")
        for k in sorted(metrics):
            v = metrics[k]
            if isinstance(v, float):
                print(f"    {k:35} = {v:.4f}")
            else:
                # truncate long values
                s = str(v)
                if len(s) > 60:
                    s = s[:57] + "..."
                print(f"    {k:35} = {s}")

    # Skipped trainers (rare — earnings_xgb removed; only zero-shot
    # trainers may emit skipped=True if their HF model can't load) — accept
    if metrics.get("skipped"):
        print(f"\n  Trainer skipped: {metrics.get('skip_reason', 'unknown')}")
        print(f"  VERDICT: ⏭ SKIPPED (acceptable for this trainer)")
        return 0

    thresholds = QUALITY_THRESHOLDS.get(name, {})
    if not thresholds:
        print(f"\n  No quality thresholds defined for '{name}' — manual review")
        print(f"  VERDICT: ⚠ MANUAL REVIEW")
        return 0

    print(f"\n  Quality gate:")
    all_pass = True
    for key, (lo, hi) in thresholds.items():
        v = metrics.get(key)
        if v is None:
            mark = "✗"
            note = "MISSING — trainer didn't produce this metric"
            all_pass = False
        else:
            ok_lo = lo is None or v >= lo
            ok_hi = hi is None or v <= hi
            ok = ok_lo and ok_hi
            mark = "✓" if ok else "✗"
            note = "OK" if ok else f"OUT OF RANGE ({_format_bound(lo, hi)})"
            all_pass = all_pass and ok
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {mark} {key:30} = {v_str:<12} bounds {_format_bound(lo, hi):<20} — {note}")

    verdict = "PASS" if all_pass else "FAIL"
    icon = "✅" if all_pass else "❌"
    print(f"\n  VERDICT: {icon} {verdict}")
    return 0 if all_pass else 1


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one trainer on mid-tier data with quality thresholds.",
    )
    parser.add_argument("trainer", nargs="?", help="trainer name, e.g. regime_hmm")
    # Default tier is SMOKE (10 stocks, 2y, 50k timesteps, 2 epochs) so
    # every model can be validated cheaply first. Use --validation for
    # the mid-tier 50-stock / 5y run, or override flags individually.
    parser.add_argument("--validation", action="store_true",
                        help="use mid-tier validation defaults (50 stocks, 5y, 200k steps, 5 epochs)")
    parser.add_argument("--universe", type=int, default=None,
                        help="number of stocks (default 10 smoke / 50 validation)")
    parser.add_argument("--period", default=None,
                        help="yfinance period (default 2y smoke / 5y validation)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="RL timesteps (default 50k smoke / 200k validation; full=1M)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="DL epochs (default 2 smoke / 5 validation; full=12)")
    parser.add_argument("--list", action="store_true",
                        help="list all trainers + threshold keys, then exit")
    parser.add_argument("--debug", action="store_true",
                        help="print full traceback on failure")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the full JSON report to this path")
    args = parser.parse_args()

    if args.list:
        list_trainers()
        return 0

    if not args.trainer:
        parser.error("trainer name required (or pass --list to see options)")

    # Resolve defaults from tier: smoke (10/2y/50k/2) or validation (50/5y/200k/5)
    if args.validation:
        univ = args.universe or 50
        period = args.period or "5y"
        timesteps = args.timesteps or 200_000
        epochs = args.epochs or 5
        tier = "validation"
    else:
        univ = args.universe or 10
        period = args.period or "2y"
        timesteps = args.timesteps or 50_000
        epochs = args.epochs or 2
        tier = "smoke"

    _set_validation_env(
        universe=univ, period=period, timesteps=timesteps, epochs=epochs,
    )

    logger.info("%s env: universe=%d period=%s timesteps=%d epochs=%d",
                tier, univ, period, timesteps, epochs)

    report = run_one_trainer(args.trainer, debug=args.debug)

    if args.report:
        args.report.write_text(json.dumps(report, indent=2, default=str))

    return render_verdict(report)


if __name__ == "__main__":
    sys.exit(main())
