#!/usr/bin/env bash
# Local end-to-end smoke of the FULL swing stack (Chronos + cached TimesFM/
# Kronos + LGBM) on a Mac/CPU — so the whole training pipeline is debugged here
# BEFORE spending RunPod GPU credits. Tiny universe + coarse forecast stride =
# a few minutes of LGBM; the Chronos pass computes ~13 rebalance dates for 6
# symbols on CPU.
#
# Usage:  bash scripts/runpod/smoke_swing_local.sh
#
# Requirements:
#   * artifacts/forecast_cache/momentum_{tsfm,kronos}.parquet MUST exist —
#     swing consumes them READ-ONLY (momentum owns that cache; without them
#     the spine drops every row on the all-NaN tsfm/kronos columns). The
#     script aborts up-front with a clear message if they're missing.
#   * chronos-forecasting: the first run downloads amazon/chronos-2 from
#     HuggingFace (~1 GB) — acceptable one-time cost for the smoke; later runs
#     reuse the swing_chronos.parquet cache and the local HF model cache.
#
# Env (all set below): FORECAST_DEVICE=cpu (MPS segfaults the foundation
# models), KMP_DUPLICATE_LIB_OK=TRUE (torch+chronos each load libomp).

set -euo pipefail
cd "$(dirname "$0")/../.."

CACHE_DIR="${FORECAST_CACHE_DIR:-artifacts/forecast_cache}"
for f in momentum_tsfm.parquet momentum_kronos.parquet; do
    if [ ! -f "$CACHE_DIR/$f" ]; then
        echo "✗ ABORT: $CACHE_DIR/$f is missing." >&2
        echo "  Swing reads momentum's TimesFM/Kronos forecast parquets READ-ONLY and" >&2
        echo "  never recomputes them. Without them every training row is dropped" >&2
        echo "  (all-NaN forecast cols). Run the momentum forecast backfill first" >&2
        echo "  (scripts/runpod/train_momentum_gpu.sh) or restore the parquets into" >&2
        echo "  $CACHE_DIR/." >&2
        exit 1
    fi
done
echo "✓ momentum forecast parquets present in $CACHE_DIR"

# ISOLATION (2026-07-06 incident): the smoke's tiny-universe run must NEVER
# write its 6-symbol swing_chronos stub into the real cache dir — that stub
# poisoned two full GPU runs. Copy the momentum parquets into a throwaway dir;
# reads work, writes are discarded with the tmpdir.
SMOKE_CACHE="$(mktemp -d /tmp/smoke_forecast_cache.XXXX)"
cp "$CACHE_DIR"/momentum_tsfm.parquet "$CACHE_DIR"/momentum_kronos.parquet "$SMOKE_CACHE/"
export FORECAST_CACHE_DIR="$SMOKE_CACHE"
echo "✓ smoke cache isolated in $SMOKE_CACHE (real cache untouched)"

python -c "import chronos" 2>/dev/null || pip install chronos-forecasting

export FORECAST_DEVICE=cpu          # MPS segfaults these foundation models
export KMP_DUPLICATE_LIB_OK=TRUE    # duplicate OpenMP runtime guard (macOS)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "── Full-stack swing smoke (6 symbols, CPU) ──"
python -u - <<'PY'
import json, pathlib, time
from datetime import date
from ml.training.trainers.swing_lambdarank import train_swing, SwingConfig, cached_universe
from ml.training.purged_cv import PurgedCVConfig
from ml.training.serve_smoke import smoke_artifact
syms = cached_universe(limit=6)
assert syms, "no universe (need data/cache CSVs or data/nse_tiers/)"
t = time.time()
out = pathlib.Path("/tmp/swing_smoke_local")
cfg = SwingConfig(with_forecasts=True, forecast_stride=120,
                  start=date(2020, 1, 1), end=date(2026, 2, 1),
                  cv=PurgedCVConfig(n_folds=2, test_days=63, embargo_days=10, train_days=378))
m = train_swing(cfg=cfg, symbols=syms, out_dir=out)
# full set = 55 swing/RS features + 6 forecast cols (TimesFM x2 + Kronos +
# Chronos x2 + ensemble)
assert m["n_features"] >= 55, f"expected the full feature set (>=55), got {m['n_features']}"
# the spine ran all stages, not just training
assert "eda" in m and "feature_audit" in m and "hpo" in m, "spine stages missing from metrics"
# train/serve contract: the artifact's booster names must equal feature_order.json
# (the audit's #1 skew guard) — makes this an END-TO-END train->serve smoke.
ok, why = smoke_artifact(out, "swing_lambdarank.txt")
assert ok, f"serve contract FAILED: {why}"
print(f"\n✓ FULL STACK GREEN in {time.time()-t:.0f}s — serve-contract: {why} — {json.dumps(m)}")
PY
echo "✓ local end-to-end smoke passed — safe to run on RunPod GPU"
