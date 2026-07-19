#!/usr/bin/env bash
# Preflight check — run BEFORE kicking off training on RunPod.
#
# What it does:
#   1. Verifies pip dependencies (torch, qlib, pyqlib, neuralforecast, mlfinpy,
#      jugaad-data, lightgbm, stable_baselines3, autogluon, transformers).
#   2. Runs data audit (scripts/train/audit_trainer_data.py).
#   3. Builds Qlib NSE provider if missing (~30-45 min, one-time).
#   4. Runs data quality report (scripts/data/data_quality_report.py).
#   5. Exits non-zero if any blocking issue found.
#
# Usage:
#   bash scripts/runpod/preflight_check.sh                  # default smoke audit (10 stocks, 2y)
#   bash scripts/runpod/preflight_check.sh --universe 50    # mid-tier audit
#   SKIP_QLIB_BUILD=1 bash scripts/runpod/preflight_check.sh   # skip the slow qlib step

set -euo pipefail

UNIVERSE="${UNIVERSE:-10}"
PERIOD="${PERIOD:-2y}"
SKIP_QLIB_BUILD="${SKIP_QLIB_BUILD:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --universe) UNIVERSE="$2"; shift 2 ;;
        --period) PERIOD="$2"; shift 2 ;;
        --skip-qlib) SKIP_QLIB_BUILD=1; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

echo "============================================================"
echo " PREFLIGHT CHECK — universe=$UNIVERSE  period=$PERIOD"
echo "============================================================"

# ── 1. Python imports ─────────────────────────────────────────────────────
echo
echo "[1/4] Python deps..."
python - <<'PY'
import importlib, sys
required = [
    ("torch", "CUDA stack"),
    ("lightgbm", "LightGBM"),
    ("xgboost", "XGBoost (kept even though F9 deferred)"),
    ("stable_baselines3", "RL"),
    ("hmmlearn", "regime_hmm"),
    ("yfinance", "OHLCV last-resort fallback"),
    ("pandas", "core"),
    ("numpy", "core"),
    # No fallbacks — every model needs its real library installed.
    ("jugaad_data", "bhavcopy primary"),
    ("qlib", "qlib_alpha158"),
    ("pytorch_forecasting", "tft_swing legacy"),
    ("neuralforecast", "tft_swing primary"),
    ("mlfinpy", "AFML triple-barrier"),
    ("transformers", "TimesFM + Chronos"),
    ("timesfm", "TimesFM legacy loader"),
    ("chronos", "Chronos forecaster"),
    ("autogluon.timeseries", "Chronos-2 via AutoGluon"),
    ("kronos", "Kronos OHLCV foundation model (AAAI 2026)"),
]
fail = []
for pkg, why in required:
    try:
        importlib.import_module(pkg)
        print(f"  ✅ {pkg} ({why})")
    except ImportError as e:
        print(f"  ❌ {pkg} MISSING ({why}) — {e}")
        fail.append(pkg)
if fail:
    print(f"\nABORT: {len(fail)} required package(s) missing: {fail}")
    print("Run: bash scripts/runpod/runpod_full_pipeline.sh (Phase 4 installs everything)")
    sys.exit(1)
PY
echo "[1/4] deps OK"

# ── 2. Data audit ─────────────────────────────────────────────────────────
echo
echo "[2/4] Per-trainer data audit (universe=$UNIVERSE, period=$PERIOD)..."
python scripts/train/audit_trainer_data.py --universe "$UNIVERSE" --period "$PERIOD" \
    --report /workspace/data_audit.json 2>&1 | tail -60

# ── 3. Qlib provider build (slow — one-time) ──────────────────────────────
echo
echo "[3/4] Qlib NSE provider..."
QLIB_DIR="${HOME}/.qlib/qlib_data/nse_data"
QLIB_SYM_COUNT=$(ls -1 "$QLIB_DIR/features/" 2>/dev/null | wc -l | tr -d ' ')
if [ "$QLIB_SYM_COUNT" -ge 100 ]; then
    echo "  ✅ Qlib provider has $QLIB_SYM_COUNT symbols — skip rebuild"
elif [ "$SKIP_QLIB_BUILD" = "1" ]; then
    echo "  ⚠ SKIP_QLIB_BUILD=1 — qlib_alpha158 will FAIL until built"
else
    echo "  Building Qlib provider (one-time, 30-45 min)..."
    python scripts/data/ingest_nse_to_qlib.py 2>&1 | tee /workspace/qlib_ingest.log | tail -20
    QLIB_SYM_COUNT=$(ls -1 "$QLIB_DIR/features/" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✅ Built $QLIB_SYM_COUNT symbols in $QLIB_DIR"
fi

# ── 4. Data quality report (blocking) ─────────────────────────────────────
echo
echo "[4/4] Data quality report (blocking)..."
python scripts/data/data_quality_report.py --universe "$UNIVERSE" --period "$PERIOD" \
    2>&1 | tail -30

echo
echo "============================================================"
echo " PREFLIGHT CHECK — ALL GREEN  (universe=$UNIVERSE)"
echo "============================================================"
echo " Next: bash scripts/runpod/runpod_full_pipeline.sh"
