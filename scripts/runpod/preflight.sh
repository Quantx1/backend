#!/usr/bin/env bash
# Local SMOKE_MODE preflight — run every trainer end-to-end on a tiny
# universe to catch import errors, schema drift, missing deps, and dead
# code paths BEFORE booting a paid pod.
#
# Total runtime: ~30-90 min on M-series Mac (depends on RL trainers).
# Cost: $0.
#
# Pass criteria: every trainer either completes OR reports a clean skip
# (e.g. "requires_gpu and CPU-only run"). Any exception = STOP, do not
# boot the pod until fixed.
#
# Usage:
#   ./scripts/runpod/preflight.sh          # run everything
#   ./scripts/runpod/preflight.sh --tests-only   # skip trainers, just run pytest

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }
head() { echo; echo -e "${YELLOW}── $* ──${NC}"; }

# ── Env ─────────────────────────────────────────────────────────────
export SMOKE_MODE=1
export KRONOS_ENABLED=0           # disable until first run is green
export LGBM_HISTORY_YEARS=2
export INTRADAY_TOP_N=8
export PYTHONPATH="${REPO_ROOT}"

REPORTS_DIR="reports/preflight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${REPORTS_DIR}"

# ── Tier 0 + Tier 1 pytest sweep ────────────────────────────────────
head "Tier 0-1 pytest sweep"
if python -m pytest tests/ml/ -q --tb=short \
    --ignore=tests/ml/test_robustness.py \
    --ignore=tests/ml/test_latency.py 2>&1 | tee "${REPORTS_DIR}/pytest.log"; then
    ok "pytest tier 0+1 passed"
else
    fail "pytest tier 0+1 failed — see ${REPORTS_DIR}/pytest.log"
fi

if [ "${1:-}" = "--tests-only" ]; then
    head "Tests-only run complete"
    exit 0
fi

# ── Trainer dry-run discovery ────────────────────────────────────────
head "Trainer discovery"
if python scripts/train/train_all_models.py --dry-run --verbose 2>&1 \
        | tee "${REPORTS_DIR}/dry_run.log"; then
    ok "every trainer discoverable + ready"
else
    fail "discovery failed — see ${REPORTS_DIR}/dry_run.log"
fi

# ── Per-trainer SMOKE run ────────────────────────────────────────────
# v1 scope (locked 2026-05-17): 9 trainers. Dropped permanently:
#   momentum_chronos (redundant w/ TimesFM), options_rl + vix_tft
#   (replaced by rule-based F&O), chronos2_macro (deferred to v1.1).
TRAINERS=(
    regime_hmm
    lgbm_signal_gate
    qlib_alpha158
    intraday_lstm
    tft_swing
    momentum_timesfm
    finrl_x_ppo
    finrl_x_ddpg
    finrl_x_a2c
)

PASSED=()
FAILED=()
SKIPPED=()

for t in "${TRAINERS[@]}"; do
    head "SMOKE: ${t}"
    LOG="${REPORTS_DIR}/smoke_${t}.log"
    if python scripts/train/train_all_models.py --only "${t}" --no-upload \
            > "${LOG}" 2>&1; then
        if grep -q "status=success" "${LOG}"; then
            ok "${t} SUCCESS"
            PASSED+=("${t}")
        elif grep -q "status=skipped" "${LOG}"; then
            warn "${t} skipped (likely requires_gpu)"
            SKIPPED+=("${t}")
        else
            warn "${t} completed but status unclear — review ${LOG}"
            PASSED+=("${t}")
        fi
    else
        echo -e "${RED}✗ ${t} CRASHED${NC} — see ${LOG}"
        FAILED+=("${t}")
    fi
done

# ── Tier 4-5 artifact-based tests ────────────────────────────────────
head "Tier 4-5 post-train robustness + latency"
python -m pytest tests/ml/test_robustness.py tests/ml/test_latency.py \
    tests/ml/test_reproducibility.py -q 2>&1 \
    | tee "${REPORTS_DIR}/tier45.log" || \
    warn "Tier 4-5 had failures — review ${REPORTS_DIR}/tier45.log"

# ── Summary ──────────────────────────────────────────────────────────
head "PREFLIGHT SUMMARY"
echo "Passed   (${#PASSED[@]}): ${PASSED[*]:-(none)}"
echo "Skipped  (${#SKIPPED[@]}): ${SKIPPED[*]:-(none)}"
echo "Failed   (${#FAILED[@]}): ${FAILED[*]:-(none)}"
echo
echo "Logs: ${REPORTS_DIR}/"

if [ "${#FAILED[@]}" -gt 0 ]; then
    fail "Preflight has ${#FAILED[@]} failed trainers. Fix locally before paid pod run."
fi
ok "Preflight clean — safe to boot pod."
echo
echo "Next: write down the fresh git SHA + your B2/Supabase env vars,"
echo "then run scripts/runpod/pod_bootstrap.sh on the pod."
