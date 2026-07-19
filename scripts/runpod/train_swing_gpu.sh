#!/usr/bin/env bash
# Full-stack Swing training on a RunPod GPU pod.
#
# Trains the LGBM LambdaRank ranker WITH forecast features. Only Chronos-2 is
# computed here (the GPU column swing owns); the TimesFM/Kronos columns are
# consumed READ-ONLY from momentum's parquets, which MUST be restored into
# artifacts/forecast_cache/ before this runs (they ship in the cache tarball —
# without them the spine drops every row on the all-NaN cols and fails loud).
# Self-contained otherwise: needs only a GPU + network (HuggingFace model pull
# + yfinance OHLCV). No Supabase/B2 secrets required — the model + metrics are
# saved locally and printed; you download the artifact afterwards.
#
# Usage on a fresh pod (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel):
#   git clone -b feat/mldl-4engine https://github.com/Ri2506/quantx.git
#   cd quantx
#   # restore data/cache + artifacts/forecast_cache from the cache tarball
#   bash scripts/runpod/train_swing_gpu.sh
#
# Optional: FORECAST_STRIDE=5 (default; higher = cheaper/faster, coarser),
#           UNIVERSE_LIMIT=120 (cap symbols for a cheaper first run).

set -euo pipefail
# PEP 668 (Ubuntu 24.04 pods): without this, every pip install dies.
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/../.."
YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
head() { echo; echo -e "${YELLOW}── $* ──${NC}"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

head "GPU sanity"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))" \
    || fail "CUDA not available — this script needs a GPU pod"

head "Momentum forecast parquets (swing reads them READ-ONLY)"
CACHE_DIR="${FORECAST_CACHE_DIR:-artifacts/forecast_cache}"
for f in momentum_tsfm.parquet momentum_kronos.parquet; do
    [ -f "$CACHE_DIR/$f" ] || fail "$CACHE_DIR/$f missing — restore momentum's forecast cache first (swing never recomputes it; every row would be dropped on the all-NaN cols)"
done
ok "momentum forecast parquets present in $CACHE_DIR"

head "Installing deps (swing-only; NOT the full API stack — avoids torch conflicts)"
python -m pip install --upgrade pip wheel
# The swing trainer's data plane falls back to yfinance, so it needs only
# these — not requirements.txt (FastAPI/supabase/etc.), which would fight
# the foundation models over the torch pin.
# optuna = Stage-7 HPO (silently skipped without it); matplotlib = report PNGs;
# safetensors/tqdm = HF inference deps.
# pydantic-settings = backend.core.config (pulled in via ml.data.data_loader's
# provider import); ta = indicator features. Both found missing on a
# clean Ubuntu-24.04 pod (PEP 668: export PIP_BREAK_SYSTEM_PACKAGES=1).
# scikit-learn: lightgbm's sklearn API (LGBMRanker) hard-requires it — its
# absence crashed the first fit on a clean pod (2026-07-06) AFTER the 4h
# backfill. Keep it in this line forever.
python -m pip install lightgbm scikit-learn yfinance scipy pandas numpy pyarrow einops optuna matplotlib safetensors tqdm pydantic-settings ta
# Chronos-2 zero-shot pipeline (the only foundation model swing computes).
python -m pip install chronos-forecasting
ok "chronos-forecasting installed"

head "Training Swing (full stack: LGBM LambdaRank + Chronos, cached TimesFM/Kronos)"
# Duplicate-OpenMP guard (torch + chronos each load libomp) — caught as a
# SIGSEGV in local debugging; harmless on a clean Linux pod.
export KMP_DUPLICATE_LIB_OK=TRUE
LIMIT_ARG=""
[ -n "${UNIVERSE_LIMIT:-}" ] && LIMIT_ARG="--limit ${UNIVERSE_LIMIT}"
HPO_ARG=""
[ -n "${HPO_TRIALS:-}" ] && HPO_ARG="--hpo-trials ${HPO_TRIALS}"
python -m ml.training.trainers.swing_lambdarank \
    --with-forecasts --stride "${FORECAST_STRIDE:-5}" ${LIMIT_ARG} ${HPO_ARG}

head "Done — artifact + metrics + report"
echo "Saved to: artifacts/models/swing_lambdarank/"
echo "Download with:  runpodctl send artifacts/models/swing_lambdarank"
echo "Also download the fresh Chronos cache:  runpodctl send $CACHE_DIR/swing_chronos.parquet"
echo
cat artifacts/models/swing_lambdarank/metrics.json 2>/dev/null | python -c "import json,sys; m=json.load(sys.stdin); [print(f'{k}: {m[k]}') for k in ('model','n_features','n_rows','n_symbols','rank_ic_mean','rank_icir','decile_spread_mean','deflated_sharpe','probability_backtest_overfitting','swing_lambdarank_quality_pass','train_seconds') if k in m]"
echo
[ -f artifacts/models/swing_lambdarank/report.md ] && sed -n '1,6p' artifacts/models/swing_lambdarank/report.md
