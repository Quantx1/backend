#!/usr/bin/env bash
# PR 202 — Quant X RunPod training driver.
#
# End-to-end script for the $15 RunPod budget. Runs on an RTX 4090
# community pod (image: runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204).
#
# Sequence:
#   1. Install upstream libraries (no custom ports — Step 2 lock)
#   2. Verify discovery (14 trainers expected)
#   3. Smoke test 2 cheap trainers (~10 min, ~$0.06)
#   4. Backfill caches (~50 min, ~$0.28)
#   5. Bootstrap Qlib NSE provider directory (~30 min)
#   6. Full training run with --promote (~5.5 hrs, ~$1.87)
#   7. Extract + display gate decisions
#
# Total expected: ~7 hours, ~$2.50 of the $15 budget. Buffer for
# retries / debugging covers any shortfall.
#
# Required env vars (set BEFORE running):
#   B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET (default: quantx-models)
#   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
#   QLIB_PROVIDER_URI (optional, default: ~/.qlib/qlib_data/nse_data)
#
# Usage:
#   chmod +x scripts/runpod/runpod_train.sh
#   ./scripts/runpod/runpod_train.sh 2>&1 | tee training_session.log

set -euo pipefail

# ---------- color helpers ----------
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
RESET='\033[0m'

log()  { printf "${BLUE}[%s]${RESET} %s\n" "$(date +%H:%M:%S)" "$*"; }
ok()   { printf "${GREEN}[%s] ✓${RESET} %s\n" "$(date +%H:%M:%S)" "$*"; }
warn() { printf "${YELLOW}[%s] !${RESET} %s\n" "$(date +%H:%M:%S)" "$*"; }
fail() { printf "${RED}[%s] ✗${RESET} %s\n" "$(date +%H:%M:%S)" "$*"; exit 1; }

START_TIME=$(date +%s)

# ---------- 0. Sanity checks ----------
log "Phase 0: sanity checks"
[ -f ml/training/runner.py ] || fail "Run from quantx repo root (ml/training/runner.py not found)"
command -v python >/dev/null 2>&1 || fail "python not on PATH"
command -v pip    >/dev/null 2>&1 || fail "pip not on PATH"

if [ -z "${B2_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    warn "B2 credentials missing — uploads will fail; export B2_KEY_ID and B2_APPLICATION_KEY first"
fi
if [ -z "${SUPABASE_URL:-}" ] || [ -z "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
    warn "Supabase credentials missing — model_versions writes will fail"
fi

# ---------- 1. Install upstream libraries ----------
log "Phase 1: installing upstream libraries (real Microsoft Qlib, real pytorch-forecasting, real Amazon Chronos)"
pip install --quiet --upgrade pip wheel
pip install --quiet -r requirements-train.txt
# Step 2 §1.4 — real Microsoft Qlib
pip install --quiet pyqlib
# Step 2 §1.6 — real pytorch-forecasting + Lightning
pip install --quiet pytorch-forecasting lightning
# v1 scope (locked 2026-05-17) — chronos2_macro + momentum_chronos
# dropped. autogluon.timeseries + chronos-forecasting installs removed;
# momentum_timesfm is the only zero-shot forecaster shipping and it
# pulls direct from HuggingFace via the transformers package.
# PR 179 — bhavcopy primary source
pip install --quiet jugaad-data
# PR 193 — fractional-diff stationarity tests
pip install --quiet statsmodels
ok "libraries installed"

# ---------- 2. Verify discovery ----------
log "Phase 2: verifying trainer discovery"
DISCOVERED=$(python -c "from ml.training.discovery import discover_sorted; print(len(discover_sorted()))" 2>&1 | tail -1)
# v1 scope locked 2026-05-17 — 9 trainers (dropped: momentum_chronos,
# options_rl, vix_tft, chronos2_macro).
if [ "$DISCOVERED" != "9" ]; then
    fail "expected 9 trainers, got $DISCOVERED — abort before wasting GPU"
fi
ok "9 trainers discovered (v1 scope)"

# Spot-check that real upstream libraries are importable
python -c "import qlib, pytorch_forecasting, lightning; print('upstream libs OK')" || fail "upstream lib import failed"

# ---------- 3. Smoke test ----------
log "Phase 3: smoke test on regime_hmm + momentum_timesfm (cheap trainers, ~10 min)"
python -m ml.training.runner --only regime_hmm,momentum_timesfm --promote 2>&1 | tee smoke.log
if grep -q "FAILED" smoke.log; then
    fail "smoke test had failures — review smoke.log before continuing"
fi
ok "smoke test passed"

# ---------- 4. Backfills ----------
log "Phase 4: backfilling caches (fundamentals + FII/DII + sentiment)"
python scripts/data/backfill_fundamentals.py || warn "fundamentals backfill had issues"
python scripts/data/backfill_fii_dii.py        || warn "FII/DII backfill had issues"
python scripts/data/backfill_sentiment.py      || warn "sentiment backfill had issues"
ok "backfills complete"

# ---------- 5. Qlib provider bootstrap ----------
log "Phase 5: bootstrapping Qlib NSE provider directory (required for qlib_alpha158)"
QLIB_DIR="${QLIB_PROVIDER_URI:-$HOME/.qlib/qlib_data/nse_data}"
if [ ! -d "$QLIB_DIR" ] || [ -z "$(ls -A "$QLIB_DIR" 2>/dev/null)" ]; then
    log "Qlib provider directory missing or empty — running ingest_nse_to_qlib.py"
    python scripts/data/ingest_nse_to_qlib.py || warn "Qlib ingest had issues — qlib_alpha158 may skip"
else
    ok "Qlib provider directory already populated at $QLIB_DIR"
fi

# ---------- 6. Full training run ----------
log "Phase 6: full --all training run (~5.5 hrs of GPU work)"
log "  v1 scope locked 2026-05-17 — 9 trainers."
log "  Dropped: momentum_chronos / options_rl / vix_tft / chronos2_macro."
log "  earnings_xgb removed 2026-05-11 — F9 deferred until real labels."
log ""
log "  Tail this log live in another shell: tail -f runner.log"
log ""

RUNNER_LIST="finrl_x_a2c,finrl_x_ddpg,finrl_x_ppo,intraday_lstm,lgbm_signal_gate,momentum_timesfm,qlib_alpha158,regime_hmm,tft_swing"

python -m ml.training.runner \
    --only "$RUNNER_LIST" \
    --promote \
    --json 2>&1 | tee runner.log

# ---------- 7. Extract gate decisions ----------
log "Phase 7: extracting gate decisions from runner.log"
echo ""
echo "============================================================"
echo "PROMOTION SUMMARY"
echo "============================================================"
python <<'PYEOF'
import json
import re
import sys

try:
    with open("runner.log") as f:
        text = f.read()
    # The --json flag emits a JSON list at the end of stdout. Extract it.
    json_blob = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if json_blob is None:
        print("no JSON report block found in runner.log")
        sys.exit(0)
    reports = json.loads(json_blob.group(0))
    print(f"{'trainer':<28} {'status':<8} {'duration':<10} {'version':<8} {'promoted':<10} {'primary':<25}")
    print("-" * 92)
    for r in reports:
        name = r.get("name", "")
        status = r.get("status", "")
        dur = f"{r.get('duration_sec', 0):.0f}s"
        v = f"v{r['version']}" if r.get("version") is not None else ""
        promoted = "PROD ★" if r.get("promoted") else ("(eval)" if status == "ok" else "")
        m = r.get("metrics") or {}
        primary = f"{m.get('primary_metric', '?')}={m.get('primary_value', '?')}"
        print(f"{name:<28} {status:<8} {dur:<10} {v:<8} {promoted:<10} {primary:<25}")
        if r.get("error"):
            print(f"   ↳ {r['error'][:120]}")
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for r in reports:
        counts[r.get("status", "ok")] = counts.get(r.get("status", "ok"), 0) + 1
    promoted_count = sum(1 for r in reports if r.get("promoted"))
    print("-" * 92)
    print(f"summary: ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']} promoted={promoted_count}")
except Exception as exc:
    print(f"failed to parse runner.log: {exc}")
PYEOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "============================================================"
ok "Training session complete in $((ELAPSED / 60)) min"
log "Estimated cost @ \$0.34/hr: \$$(echo "scale=2; $ELAPSED / 3600 * 0.34" | bc)"
log ""
warn "REMINDER: Stop the RunPod pod NOW — \$0.34/hr keeps billing while idle."
log ""
log "Artifacts uploaded to b2://${B2_BUCKET:-quantx-models}/"
log "model_versions rows in Supabase. Inspect via:"
log "  SELECT name, version, is_prod, primary_metric, primary_value FROM model_versions ORDER BY created_at DESC LIMIT 20;"
