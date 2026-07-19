#!/usr/bin/env bash
# Pod bootstrap — run ONCE after `git clone` on a fresh RunPod 4090.
#
# This script:
#   1. Installs all training deps (pip, but cached via uv when available)
#   2. Warms HuggingFace cache for the zero-shot models BEFORE paid training
#   3. Validates that the consumer load paths can find all expected sidecars
#   4. Kicks off the full unified training run inside tmux so SSH disconnect
#      doesn't kill the run
#
# Total bootstrap: ~10-15 min. Then your wallet starts paying for training.
#
# Required env (set these BEFORE running this script):
#   B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET   — for artifact upload
#   SUPABASE_URL, SUPABASE_SERVICE_KEY         — for model_versions row
#   HF_TOKEN (optional)                        — only needed for gated models

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }
head() { echo; echo -e "${YELLOW}── $* ──${NC}"; }

# ── Sanity: required env vars ────────────────────────────────────────
head "Checking required env vars"
MISSING=()
for v in B2_KEY_ID B2_APPLICATION_KEY B2_BUCKET SUPABASE_URL SUPABASE_SERVICE_KEY; do
    if [ -z "${!v:-}" ]; then
        MISSING+=("$v")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    fail "Missing required env vars: ${MISSING[*]}. Set them and rerun."
fi
ok "all required env vars present"

# ── GPU sanity ───────────────────────────────────────────────────────
head "GPU sanity"
if ! command -v nvidia-smi >/dev/null; then
    warn "nvidia-smi not found — CPU-only run will skip RL trainers"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    ok "GPU detected"
fi

# ── pip / deps ───────────────────────────────────────────────────────
head "Installing training deps"
python3 -m pip install --upgrade pip wheel
# Use uv when available (10x faster), otherwise plain pip
if command -v uv >/dev/null; then
    INSTALLER="uv pip install"
else
    INSTALLER="python3 -m pip install"
fi

# Core
$INSTALLER \
    numpy pandas scikit-learn lightgbm hmmlearn \
    torch torchvision pytorch-forecasting onnx onnxruntime \
    yfinance jugaad-data pyqlib \
    stable-baselines3[extra] gymnasium \
    transformers accelerate sentencepiece \
    chronos-forecasting timesfm autogluon.timeseries \
    boto3 supabase pyarrow \
    pytest pytest-mock pytest-asyncio

ok "deps installed"

# ── HF cache warm ────────────────────────────────────────────────────
head "Pre-pulling HF models (warm cache BEFORE paid training)"
python3 - <<'PY'
import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
from huggingface_hub import snapshot_download

models = [
    "google/timesfm-1.0-200m-pytorch",
    "amazon/chronos-bolt-base",
    "ProsusAI/finbert",   # FinBERT-India fallback if Vansh180 errors
]
for m in models:
    try:
        snapshot_download(m, allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
        print(f"  ✓ pulled {m}")
    except Exception as e:
        print(f"  ! {m} failed: {e}")
PY
ok "HF cache warmed (errors above are non-fatal)"

# ── Final pre-train sanity ───────────────────────────────────────────
head "Pytest discovery sanity (NOT a full smoke run)"
python3 -m pytest tests/ml/test_data_integrity.py tests/ml/test_artifact_smoke.py \
    -q --tb=short || fail "Tier 0-1 tests failed on the pod — investigate"
ok "Tier 0-1 green on pod"

# ── Kick off training in tmux ────────────────────────────────────────
head "Launching unified training in tmux"
SESSION="trainall_$(date +%H%M%S)"
LOG="reports/pod_$(date +%Y%m%d_%H%M%S).log"
mkdir -p reports

# Production env for the training run
TRAIN_CMD="export KRONOS_ENABLED=1 LGBM_HISTORY_YEARS=5 INTRADAY_TOP_N=25 \
    FINRL_TIMESTEPS=300000 OPTIONS_RL_TIMESTEPS=200000; \
    python3 scripts/train/train_all_models.py --promote --verbose 2>&1 | tee ${LOG}"

if command -v tmux >/dev/null; then
    tmux new-session -d -s "$SESSION" "${TRAIN_CMD}"
    ok "training launched in tmux session: ${SESSION}"
    echo
    echo "Attach:   tmux attach -t ${SESSION}"
    echo "Detach:   Ctrl+B then D (training keeps running)"
    echo "Tail log: tail -f ${LOG}"
    echo "Resume after crash: python3 scripts/train/train_all_models.py --resume --promote"
else
    warn "tmux not installed — running in foreground (SSH disconnect WILL kill the run)"
    eval "${TRAIN_CMD}"
fi
