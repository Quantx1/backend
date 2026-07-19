#!/usr/bin/env bash
# Local end-to-end smoke of the FULL momentum stack (TimesFM + Kronos + LGBM)
# on a Mac/CPU — so the whole training pipeline is debugged here BEFORE spending
# RunPod GPU credits. Tiny universe + coarse forecast stride = a few minutes.
#
# Usage:  bash scripts/runpod/smoke_momentum_local.sh
#
# Env (all set below): FORECAST_DEVICE=cpu (MPS segfaults TimesFM/Kronos),
# KMP_DUPLICATE_LIB_OK=TRUE (torch+timesfm+chronos each load libomp), PYTHONPATH
# includes the Kronos clone.

set -euo pipefail
cd "$(dirname "$0")/../.."

KRONOS_DIR="${KRONOS_DIR:-$HOME/.kronos_repo}"
if [ ! -d "$KRONOS_DIR/.git" ]; then
    echo "Cloning Kronos -> $KRONOS_DIR"
    git clone --depth 1 https://github.com/shiyu-coder/Kronos.git "$KRONOS_DIR"
fi
python -c "import einops" 2>/dev/null || pip install --no-deps einops
python -c "import timesfm, importlib.metadata as m; assert tuple(int(x) for x in m.version('timesfm').split('.')[:2]) >= (2,0), 'need timesfm>=2.0.1'" \
    || pip install -U timesfm

export PYTHONPATH="$KRONOS_DIR:${PYTHONPATH:-}"
export FORECAST_DEVICE=cpu          # MPS segfaults these foundation models
export KMP_DUPLICATE_LIB_OK=TRUE
# Isolated cache dir: a smoke MUST NEVER write its tiny-universe parquets
# into the real artifacts/forecast_cache (2026-07-06 incident: a 6-symbol
# smoke stub poisoned two GPU runs). Reads start empty; writes are throwaway.
export FORECAST_CACHE_DIR="$(mktemp -d /tmp/smoke_forecast_cache.XXXX)"    # duplicate OpenMP runtime guard (macOS)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "── Full-stack momentum smoke (6 symbols, CPU) ──"
python -u - <<'PY'
import json, pathlib, time
from datetime import date
from ml.training.trainers.momentum_lambdarank import train_momentum, MomentumConfig, cached_universe
from ml.training.purged_cv import PurgedCVConfig
from ml.training.serve_smoke import smoke_artifact
syms = cached_universe(limit=6)
assert syms, "no universe (need data/cache CSVs or data/nse_tiers/)"
t = time.time()
out = pathlib.Path("/tmp/mom_smoke_local")
cfg = MomentumConfig(with_forecasts=True, forecast_stride=120,
                     start=date(2020, 1, 1), end=date(2026, 2, 1),
                     cv=PurgedCVConfig(n_folds=2, test_days=63, embargo_days=20, train_days=378))
m = train_momentum(cfg=cfg, symbols=syms, out_dir=out)
# full set = 71 stock/RS features + 4 forecast cols (TimesFM x2 + Kronos + ensemble)
assert m["n_features"] >= 70, f"expected the full feature set (>=70), got {m['n_features']}"
# the spine ran all stages, not just training
assert "eda" in m and "feature_audit" in m and "hpo" in m, "spine stages missing from metrics"
# train/serve contract: the artifact's booster names must equal feature_order.json
# (the audit's #1 skew guard) — makes this an END-TO-END train->serve smoke.
ok, why = smoke_artifact(out, "momentum_lambdarank.txt")
assert ok, f"serve contract FAILED: {why}"
print(f"\n✓ FULL STACK GREEN in {time.time()-t:.0f}s — serve-contract: {why} — {json.dumps(m)}")
PY
echo "✓ local end-to-end smoke passed — safe to run on RunPod GPU"
