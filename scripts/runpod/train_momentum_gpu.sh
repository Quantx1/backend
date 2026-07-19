#!/usr/bin/env bash
# Full-stack Momentum training on a RunPod GPU pod.
#
# Trains the LGBM LambdaRank ranker WITH TimesFM + Kronos forecast features
# (the foundation-model columns that need a GPU). Self-contained: needs only
# a GPU + network (HuggingFace model pulls + yfinance OHLCV). No Supabase/B2
# secrets required — the model + metrics are saved locally and printed; you
# download the artifact afterwards.
#
# Usage on a fresh pod (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel):
#   git clone -b feat/mldl-4engine https://github.com/Ri2506/quantx.git
#   cd quantx
#   bash scripts/runpod/train_momentum_gpu.sh
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

head "Installing deps (momentum-only; NOT the full API stack — avoids torch conflicts)"
python -m pip install --upgrade pip wheel
# The momentum trainer's data plane falls back to yfinance, so it needs only
# these — not requirements.txt (FastAPI/supabase/etc.), which would fight
# timesfm over the torch pin.
# optuna = Stage-7 HPO (silently skipped without it); matplotlib = report PNGs;
# safetensors/tqdm = Kronos inference deps (we do NOT install Kronos's own
# requirements.txt — it pins huggingface_hub==0.33.1 which breaks timesfm>=2).
# pydantic-settings = backend.core.config (pulled in via ml.data.data_loader's
# provider import); ta = ADX in momentum_features. Both found missing on a
# clean Ubuntu-24.04 pod (PEP 668: export PIP_BREAK_SYSTEM_PACKAGES=1).
# scikit-learn: lightgbm's sklearn API (LGBMRanker) hard-requires it — its
# absence crashed the first fit on a clean pod (2026-07-06) AFTER the 4h
# backfill. Keep it in this line forever.
python -m pip install lightgbm scikit-learn yfinance scipy pandas numpy pyarrow einops optuna matplotlib safetensors tqdm pydantic-settings ta
# TimesFM 2.5 (torch backend). >=2.0.1 fixes a huggingface_hub 'proxies' crash
# caught in local debugging.
python -m pip install "timesfm[torch]>=2.0.1" || python -m pip install "timesfm>=2.0.1"
ok "timesfm installed"

head "Kronos (clone + PYTHONPATH)"
KRONOS_DIR="${KRONOS_DIR:-/workspace/Kronos}"
if [ ! -d "$KRONOS_DIR/.git" ]; then
    git clone https://github.com/shiyu-coder/Kronos.git "$KRONOS_DIR"
fi
# Deliberately NOT `pip install -r Kronos/requirements.txt` — its hard pins
# (huggingface_hub==0.33.1, pandas==2.2.2) downgrade + break timesfm. The only
# runtime deps Kronos needs beyond torch are safetensors/tqdm/einops (installed
# above); `from model import Kronos...` works off PYTHONPATH alone.
export PYTHONPATH="$KRONOS_DIR:${PYTHONPATH:-}"
ok "Kronos on PYTHONPATH ($KRONOS_DIR)"

head "Training Momentum (full stack: LGBM LambdaRank + TimesFM + Kronos)"
# Duplicate-OpenMP guard (torch + timesfm + chronos each load libomp) — caught
# as a SIGSEGV in local debugging; harmless on a clean Linux pod.
export KMP_DUPLICATE_LIB_OK=TRUE
LIMIT_ARG=""
[ -n "${UNIVERSE_LIMIT:-}" ] && LIMIT_ARG="--limit ${UNIVERSE_LIMIT}"
HPO_ARG=""
[ -n "${HPO_TRIALS:-}" ] && HPO_ARG="--hpo-trials ${HPO_TRIALS}"
python -m ml.training.trainers.momentum_lambdarank \
    --with-forecasts --stride "${FORECAST_STRIDE:-5}" ${LIMIT_ARG} ${HPO_ARG}

head "Done — artifact + metrics + report"
echo "Saved to: artifacts/models/momentum_lambdarank/"
echo "Download with:  runpodctl send artifacts/models/momentum_lambdarank"
echo
cat artifacts/models/momentum_lambdarank/metrics.json 2>/dev/null | python -c "import json,sys; m=json.load(sys.stdin); [print(f'{k}: {m[k]}') for k in ('model','n_features','n_rows','n_symbols','rank_ic_mean','rank_icir','decile_spread_mean','deflated_sharpe','probability_backtest_overfitting','momentum_lambdarank_quality_pass','train_seconds') if k in m]"
echo
[ -f artifacts/models/momentum_lambdarank/report.md ] && sed -n '1,6p' artifacts/models/momentum_lambdarank/report.md
