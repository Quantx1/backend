#!/usr/bin/env bash
# Weekly forecast-cache refresh on a RunPod GPU pod (Phase 2 deploy plumbing).
#
# Tops up artifacts/forecast_cache/{momentum_tsfm,momentum_kronos,swing_chronos}
# .parquet to the latest trading day using the trainers' cache-first blocks —
# only the NEW rebalance dates since each cache's max date are computed
# (incremental top-up), so a weekly run costs ~minutes of GPU, not the hours
# of the original backfill. Serving reads these parquets on CPU via
# ml/features/forecast_serving.latest_forecasts, so run this WEEKLY (e.g.
# Sunday, before Monday's open) and ship the refreshed parquets back to the
# serving box / artifact store — otherwise forecast_age_days just keeps
# growing and the rankers score on stale forward-looking views.
#
# Usage on a fresh pod (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel):
#   git clone -b feat/xai-redesign https://github.com/Ri2506/quantx.git
#   cd quantx
#   # restore artifacts/forecast_cache from the cache tarball (top-up needs
#   # the existing parquets; without them this becomes a FULL backfill)
#   bash scripts/runpod/refresh_forecast_cache.sh
#
# Optional: FORECAST_STRIDE=5 (default; higher = cheaper/faster, coarser),
#           UNIVERSE_LIMIT=120 (cap symbols for a cheap smoke),
#           FORECAST_CACHE_DIR=artifacts/forecast_cache (override location).

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

head "Installing deps (forecast-only; NOT the full API stack — avoids torch conflicts)"
python -m pip install --upgrade pip wheel
# Same rationale as train_momentum_gpu.sh: the data plane falls back to
# yfinance, so it needs only these — not requirements.txt (FastAPI/supabase/
# etc.), which would fight timesfm over the torch pin.
# safetensors/tqdm = Kronos inference deps (we do NOT install Kronos's own
# requirements.txt — it pins huggingface_hub==0.33.1 which breaks timesfm>=2).
# pydantic-settings = backend.core.config (pulled in via ml.data.data_loader's
# provider import); ta = indicator features (imported by the feature modules).
# scikit-learn: lightgbm's sklearn API hard-requires it — keep it forever.
python -m pip install lightgbm scikit-learn yfinance scipy pandas numpy pyarrow einops optuna matplotlib safetensors tqdm pydantic-settings ta
# TimesFM 2.5 (torch backend). >=2.0.1 fixes a huggingface_hub 'proxies' crash
# caught in local debugging.
python -m pip install "timesfm[torch]>=2.0.1" || python -m pip install "timesfm>=2.0.1"
# Chronos-2 zero-shot pipeline (the swing_chronos column).
python -m pip install chronos-forecasting
ok "timesfm + chronos-forecasting installed"

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

head "Refreshing forecast caches (incremental top-up via the trainers' cache-first blocks)"
# Duplicate-OpenMP guard (torch + timesfm + chronos each load libomp) — caught
# as a SIGSEGV in local debugging; harmless on a clean Linux pod.
export KMP_DUPLICATE_LIB_OK=TRUE
python - <<'PY'
"""Incremental weekly top-up of the three forecast parquets.

Reuses the trainers' build_features cache-first blocks verbatim (the _topup
logic computes only rebalance dates AFTER each cache's max date, min_date =
cache max + 1 day, then saves the combined frame back). Momentum OWNS
momentum_tsfm/momentum_kronos; swing OWNS swing_chronos and reads momentum's
parquets READ-ONLY — so momentum MUST run first.
"""
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from ml.training.trainers.momentum_lambdarank import (
    MomentumConfig, MomentumTrainer, cached_universe)
from ml.training.trainers.swing_lambdarank import SwingConfig, SwingTrainer

limit = int(os.environ["UNIVERSE_LIMIT"]) if os.environ.get("UNIVERSE_LIMIT") else None
stride = int(os.environ.get("FORECAST_STRIDE", "5"))
symbols = cached_universe(limit=limit)
print(f"universe: {len(symbols)} symbols, stride={stride}")

# 1) momentum_tsfm + momentum_kronos (momentum owns these; top-up + save)
mt = MomentumTrainer(
    cfg=MomentumConfig(with_forecasts=True, forecast_stride=stride),
    symbols=symbols)
mt.build_features(mt.load_panel())

# 2) swing_chronos (swing owns it; momentum parquets consumed read-only)
st = SwingTrainer(
    cfg=SwingConfig(with_forecasts=True, forecast_stride=stride),
    symbols=symbols)
st.build_features(st.load_panel())

cache = Path(os.environ.get("FORECAST_CACHE_DIR", "artifacts/forecast_cache"))
print("\n== refreshed caches ==")
for name in ("momentum_tsfm.parquet", "momentum_kronos.parquet", "swing_chronos.parquet"):
    p = cache / name
    if p.exists():
        df = pd.read_parquet(p)
        print(f"{name}: rows={len(df)} symbols={df['symbol'].nunique()} "
              f"max_date={pd.to_datetime(df['date']).max().date()}")
    else:
        print(f"{name}: MISSING")
PY

head "Done — ship the refreshed parquets to serving"
echo "Download with:  runpodctl send ${FORECAST_CACHE_DIR:-artifacts/forecast_cache}"
