#!/usr/bin/env bash
# 10-stock GPU smoke for the unified training pipeline.
#
# Same shape as runpod_full_pipeline.sh, but every universe-bound + epoch-bound
# knob is shrunk via SMOKE_MODE=1. The goal: validate that every v1 trainer
# (regime_hmm / lgbm_signal_gate / qlib_alpha158 / tft_swing / intraday_lstm /
# momentum_timesfm / finrl_x_ppo / finrl_x_ddpg / finrl_x_a2c) survives an
# end-to-end RunPod run before committing to the full $6 / 8h pipeline.
# v1 scope locked 2026-05-17 — 9 trainers; dropped: momentum_chronos,
# options_rl, vix_tft, chronos2_macro.
#
# Expected runtime: ~30-60 minutes total
# Expected cost:    ~$0.40-0.70 on RTX 4090 at $0.69/hr
#
# Universe shrink (via SMOKE_MODE=1, see ml/training/smoke.py):
#   lgbm_signal_gate:    200 → 10 liquid stocks
#   intraday_lstm:       100 → 10 names
#   tft_swing:           100 → 10 names, 5y → 2y history
#   finrl_x_*:           30  → 10 names, 1M → 50k timesteps
#   qlib_alpha158:       nse_all kept but date window 2023→2025
#   earnings_xgb:        REMOVED 2026-05-11 — F9 deferred (no real labels yet)
#
# Usage:
#   1. SSH/web terminal into a fresh RunPod RTX 4090 pod
#      Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   2. Set env vars (paste once):
#        export SUPABASE_URL=...
#        export SUPABASE_ANON_KEY=...
#        export SUPABASE_SERVICE_ROLE_KEY=...
#        export B2_APPLICATION_KEY_ID=...
#        export B2_APPLICATION_KEY=...
#   3. bash scripts/runpod/runpod_smoke_pipeline.sh
#   4. After it finishes successfully, re-run the FULL pipeline:
#        bash scripts/runpod/runpod_full_pipeline.sh

set -euo pipefail

# ── MASTER SMOKE FLAG ──────────────────────────────────────────────────────
# Picked up by ml/training/smoke.py — every trainer reads this.
export SMOKE_MODE=1
export SMOKE_UNIVERSE_SIZE=10
export SMOKE_TIMESTEPS=50000
export SMOKE_EPOCHS=2
export SMOKE_YFINANCE_PERIOD=2y
# LambdaRankIC is the new objective for qlib_alpha158 — leave on for the smoke.
export QLIB_USE_LAMBDARANK_IC=1

# ── 1. Sanity checks ───────────────────────────────────────────────────────
echo "=== SMOKE Phase 1: sanity checks ==="
set +o pipefail
nvidia-smi 2>&1 | head -20 || true
set -o pipefail

python -c "
try:
    import torch
    if not torch.cuda.is_available():
        print('CUDA not available — abort'); raise SystemExit(1)
    print('  GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('torch not yet installed — install phase will handle')" || true

# Smoke uses `--dry-run` on the runner (Phase 9) which skips B2 upload
# and the Supabase model_versions write. So the smoke can run with ZERO
# external secrets — useful for the first validation. The full pipeline
# (runpod_full_pipeline.sh) still requires Supabase + B2.
missing_warn=()
for v in SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY B2_APPLICATION_KEY_ID B2_APPLICATION_KEY; do
    if [ -z "${!v:-}" ]; then
        missing_warn+=("$v")
    fi
done
if [ ${#missing_warn[@]} -gt 0 ]; then
    echo "INFO: ${#missing_warn[@]} env vars not set (${missing_warn[*]}) —"
    echo "      smoke runs with --dry-run so B2 upload + DB write are skipped."
    echo "      Set them before running runpod_full_pipeline.sh."
fi

export SUPABASE_SERVICE_KEY="${SUPABASE_SERVICE_KEY:-${SUPABASE_SERVICE_ROLE_KEY:-}}"
export B2_KEY_ID="${B2_KEY_ID:-${B2_APPLICATION_KEY_ID:-}}"
export B2_BUCKET="${B2_BUCKET:-quantx-models}"
export B2_BUCKET_MODELS="${B2_BUCKET_MODELS:-quantx-models}"
echo "env vars: smoke ready | SMOKE_MODE=$SMOKE_MODE universe=$SMOKE_UNIVERSE_SIZE"

# ── 2. Caches on /workspace volume ─────────────────────────────────────────
echo "=== SMOKE Phase 2: redirect caches to /workspace ==="
mkdir -p /workspace/.cache/{pip,huggingface,torch}
ln -sfn /workspace/.cache/pip /root/.cache/pip
ln -sfn /workspace/.cache/huggingface /root/.cache/huggingface
ln -sfn /workspace/.cache/torch /root/.cache/torch
mkdir -p /workspace/.qlib && ln -sfn /workspace/.qlib /root/.qlib
df -h /workspace
echo "caches redirected"

# ── 3. Repo ────────────────────────────────────────────────────────────────
echo "=== SMOKE Phase 3: clone/pull repo ==="
cd /workspace
if [ ! -d quantx ]; then
    git clone https://github.com/Ri2506/quantx.git
fi
cd /workspace/quantx
git pull origin main
echo "on commit: $(git log --oneline -1)"

export PYTHONPATH="/workspace/quantx:${PYTHONPATH:-}"

# Save env vars for resume. Use :- defaulting so `set -u` doesn't fail
# when B2_* or Supabase vars are intentionally unset (smoke --dry-run).
cat > /workspace/.envrc <<ENVEOF
export SMOKE_MODE=1
export SMOKE_UNIVERSE_SIZE=${SMOKE_UNIVERSE_SIZE:-10}
export SMOKE_TIMESTEPS=${SMOKE_TIMESTEPS:-50000}
export SMOKE_EPOCHS=${SMOKE_EPOCHS:-2}
export SMOKE_YFINANCE_PERIOD=${SMOKE_YFINANCE_PERIOD:-2y}
export QLIB_USE_LAMBDARANK_IC=1
export SUPABASE_URL="${SUPABASE_URL:-}"
export SUPABASE_ANON_KEY="${SUPABASE_ANON_KEY:-}"
export SUPABASE_SERVICE_ROLE_KEY="${SUPABASE_SERVICE_ROLE_KEY:-}"
export SUPABASE_SERVICE_KEY="${SUPABASE_SERVICE_KEY:-}"
export B2_APPLICATION_KEY_ID="${B2_APPLICATION_KEY_ID:-}"
export B2_APPLICATION_KEY="${B2_APPLICATION_KEY:-}"
export B2_KEY_ID="${B2_KEY_ID:-}"
export B2_BUCKET="${B2_BUCKET:-quantx-models}"
export B2_BUCKET_MODELS="${B2_BUCKET_MODELS:-quantx-models}"
ENVEOF

# ── 4. Install — clean order, dependency-safe (mirrors full pipeline) ────
echo "=== SMOKE Phase 4: install upstream libraries ==="

# Skip if already installed AND torch can import cleanly. Saves ~15 min
# on pod restarts.
if python -c "import torch, qlib, pytorch_forecasting, lightning, timesfm, ml.data; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "all libs already installed and torch.cuda works — skipping install phase"
else
    pip install --quiet --upgrade pip wheel

    # 4a. Pre-emptively reinstall blinker so subsequent installs don't try
    # to uninstall the distutils-installed system version (which always fails).
    echo "  pre-installing blinker (clears distutils block)..."
    pip install --quiet --ignore-installed blinker

    # 4b. Torch trio — the foundation. Pinned versions; force-reinstall to
    # ensure all CUDA libs land at the matching version.
    echo "  installing torch 2.4.1 + matched CUDA libs..."
    pip install --quiet --force-reinstall \
        "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
        --index-url https://download.pytorch.org/whl/cu124

    python -c "
import torch, torch.onnx
assert torch.cuda.is_available(), 'CUDA not available'
print('  torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
"

    # 4c. Lightning + pytorch-forecasting (kept as fallback for TFT)
    echo "  installing pytorch-forecasting + lightning..."
    pip install --quiet "lightning>=2.0,<2.5"
    pip install --quiet pytorch-forecasting

    # 4c-bis. Nixtla neuralforecast — REQUIRED primary TFT backend.
    echo "  installing neuralforecast (Nixtla, REQUIRED)..."
    pip install --quiet "neuralforecast>=1.7"

    # 4d. TimesFM (Google) — REQUIRED. --no-deps preserves torch 2.4.
    # jax/jaxlib pinned to 0.4.x (0.5+ needs numpy>=2 conflicting with torch).
    echo "  installing timesfm (Google, REQUIRED) — no-deps..."
    pip install --quiet --no-deps "timesfm[torch]==1.2.7"
    pip install --quiet "jax[cpu]<0.5" "jaxlib<0.5" absl-py einshape utilsforecast
    pip install --quiet --no-deps wandb typer

    # 4e. chronos-forecasting (Amazon) — REQUIRED, --no-deps.
    echo "  installing chronos-forecasting (Amazon, REQUIRED) — no-deps..."
    pip install --quiet --no-deps chronos-forecasting
    pip install --quiet "transformers>=4.30" accelerate "huggingface-hub<1.0"

    # 4f. pyqlib (Microsoft) — REQUIRED. Manual deps to preserve torch trio.
    # cvxpy <1.8 because 1.8+ requires numpy>=2.
    echo "  installing pyqlib (Microsoft, REQUIRED)..."
    pip install --quiet --no-deps pyqlib
    pip install --quiet pyyaml gym fire ruamel.yaml mlflow plotly redis-py-cluster \
        dill loguru pymongo "cvxpy<1.8" pydantic-settings python-redis-lock

    # 4g. Core ML deps — REQUIRED.
    echo "  installing core ML deps (REQUIRED)..."
    pip install --quiet jugaad-data statsmodels lightgbm xgboost \
        supabase httpx pyarrow yfinance b2sdk hmmlearn \
        stable-baselines3 gymnasium scikit-learn \
        optuna optuna-integration \
        pandas-market-calendars korean-lunar-calendar exchange-calendars

    # 4g-bis. mlfinpy — REQUIRED canonical AFML triple-barrier labeling.
    echo "  installing mlfinpy (canonical AFML labeling, REQUIRED)..."
    pip install --quiet mlfinpy

    # 4g-ter. Kronos OHLCV foundation model (AAAI 2026, NeoQuasar). REQUIRED.
    # Research repo (NOT pip-installable). Clone + requirements + PYTHONPATH.
    echo "  installing kronos (NeoQuasar, AAAI 2026, REQUIRED) via clone + PYTHONPATH..."
    if [ ! -d /workspace/Kronos ]; then
        git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /workspace/Kronos
    else
        (cd /workspace/Kronos && git pull origin main)
    fi
    pip install --quiet -r /workspace/Kronos/requirements.txt
    export KRONOS_PATH="/workspace/Kronos"
    grep -q "KRONOS_PATH=/workspace/Kronos" ~/.bashrc \
        || echo 'export KRONOS_PATH=/workspace/Kronos' >> ~/.bashrc

    # 4h. Backend repo deps (--no-deps to preserve torch trio).
    [ -f requirements-train.txt ] && pip install --quiet --no-deps -r requirements-train.txt

    # 4i. autogluon.timeseries install removed 2026-05-17 — chronos2_macro
    # dropped from v1; no other trainer needs autogluon.

    # 4j. CRITICAL: re-pin torch trio at the end
    echo "  re-pinning torch trio to 2.4.1..."
    pip install --quiet --force-reinstall --no-deps \
        "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
        --index-url https://download.pytorch.org/whl/cu124
fi

# Final verify — abort on CRITICAL imports only. timesfm is allowed to
# fail import (zero-shot via transformers integration is the primary
# path; legacy timesfm package is fallback only).
python <<'PYEOF'
import sys
critical = ["torch", "qlib", "pytorch_forecasting", "lightning"]
optional = ["timesfm", "chronos"]
failed_critical = []
for m in critical:
    try:
        __import__(m)
    except Exception as e:
        failed_critical.append((m, str(e)))
for m in optional:
    try:
        __import__(m)
        print(f"  optional OK: {m}")
    except Exception as e:
        print(f"  optional miss: {m} ({type(e).__name__}: {str(e)[:80]})")

if failed_critical:
    print("CRITICAL IMPORTS FAILED:")
    for m, e in failed_critical:
        print(f"  {m}: {e}")
    sys.exit(1)

import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"  torch: {torch.__version__} | CUDA: True | {torch.cuda.get_device_name(0)}")
import qlib
print(f"  qlib: {qlib.__version__}")
import ml.data
print("  ml.data: OK")
print("install OK (critical imports green; timesfm/chronos may fall through to transformers)")
PYEOF

# ── 5. Trainer discovery (sanity) ──────────────────────────────────────────
echo "=== SMOKE Phase 5: trainer discovery ==="
python -m ml.training.runner --list 2>&1 | tee /workspace/quantx/trainers.log

# ── 6. Smoke (regime_hmm only, ~15s) ───────────────────────────────────────
echo "=== SMOKE Phase 6: regime_hmm fast-path (~15s, dry-run) ==="
# --dry-run skips B2 upload + DB write; smoke has no B2 keys.
python -m ml.training.runner --only regime_hmm --dry-run 2>&1 | tee /workspace/quantx/smoke_regime.log

# ── 7. Backfills (lightweight in smoke; ~3 min) ───────────────────────────
echo "=== SMOKE Phase 7: minimal backfills (~3 min) ==="
# In smoke we skip the long fundamentals + sentiment backfills. The
# FII/DII backfill failure is non-fatal — only Qlib uses it and Qlib
# empty (handled inside the trainer). FII/DII backfill is best-effort:
# NSE often blocks the archive endpoint, so we accept failure here.
if [ -n "${SUPABASE_URL:-}" ]; then
    python scripts/data/backfill_fii_dii.py 2>&1 | tee /workspace/quantx/backfill_fii.log || true
else
    echo "  Skipping FII/DII backfill — SUPABASE_URL not set (--dry-run smoke)"
fi

# ── 8. Qlib provider — small or skip ──────────────────────────────────────
echo "=== SMOKE Phase 8: Qlib NSE provider (full or skip) ==="
# Explicit existence check — avoids set -e killing the script when the
# qlib_data tree doesn't exist yet (pipefail propagates ls's exit 2).
if [ -d /root/.qlib/qlib_data/nse_data/features ]; then
    qlib_sym_count=$(ls /root/.qlib/qlib_data/nse_data/features/ 2>/dev/null | wc -l)
else
    qlib_sym_count=0
fi
echo "  Qlib provider symbol count: $qlib_sym_count"
if [ "$qlib_sym_count" -ge 10 ]; then
    echo "  → sufficient for smoke; skipping rebuild"
else
    echo "  → missing/small — building (this takes ~15 min one-time)"
    rm -rf /root/.qlib/qlib_data/nse_data
    python scripts/data/ingest_nse_to_qlib.py 2>&1 | tee /workspace/quantx/qlib_ingest.log || true
fi

# ── 8b. Data quality report — gate the training phase on this ─────────────
echo "=== SMOKE Phase 8b: pre-training data quality report ==="
# Non-blocking in smoke (we test the validators, not data perfection).
# In the full pipeline this exits non-zero on blockers — see runpod_full_pipeline.sh.
python scripts/data/data_quality_report.py --top-n 10 --start 2023-01-01 \
    2>&1 | tee /workspace/quantx/data_quality.log || true

# ── 9. Full --all training in SMOKE MODE ───────────────────────────────────
# Skip when SKIP_PHASE_9=1. The detached launcher
# (run_smoke_all_detached.sh) sets this because it runs smoke_all.py
# right after, which does the per-trainer smoke at the same scale.
# Running both is a waste of GPU time.
if [ "${SKIP_PHASE_9:-0}" = "1" ]; then
    echo "=== SMOKE Phase 9: SKIPPED (SKIP_PHASE_9=1; smoke_all.py runs next) ==="
else
    echo "=== SMOKE Phase 9: --all training, SMOKE universe ($SMOKE_UNIVERSE_SIZE stocks) ==="
    echo "Expected runtime: 30-60 minutes; cost: ~\$0.40-0.70"
    # --promote OFF and --dry-run ON for smoke:
    #   • promote OFF: 10-stock universes must never produce is_prod=TRUE rows
    #   • dry-run ON:  skip B2 upload + Supabase model_versions write,
    #                  so the smoke needs zero external secrets and won't
    #                  pollute the registry with junk versions
    python -m ml.training.runner --dry-run --json 2>&1 | tee /workspace/quantx/runner.log
fi

# ── 10. Summary + smoke verdict ────────────────────────────────────────────
echo "=== SMOKE Phase 10: summary ==="
python <<'PYEOF'
import json, re, sys
try:
    with open('/workspace/quantx/runner.log') as f:
        text = f.read()
    m = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if not m:
        print("no JSON report found in runner.log"); raise SystemExit(0)
    reports = json.loads(m.group(0))
    print(f"\n{'trainer':<28} {'status':<8} {'sec':>7}  primary")
    print("-" * 80)
    n_ok = n_skipped = n_failed = 0
    failed_names = []
    for r in reports:
        m_ = r.get('metrics') or {}
        primary = f"{m_.get('primary_metric','?')}={m_.get('primary_value','?')}"
        print(f"{r['name']:<28} {r['status']:<8} {r.get('duration_sec',0):>7.0f}  {primary}")
        if r.get('error'):
            print(f"   ↳ {r['error'][:140]}")
        s = r.get('status', 'ok')
        if s == 'ok': n_ok += 1
        elif s == 'skipped': n_skipped += 1
        else:
            n_failed += 1
            failed_names.append(r['name'])
    print("-" * 80)
    print(f"SMOKE RESULT: ok={n_ok} skipped={n_skipped} failed={n_failed}")
    if n_failed:
        print(f"FAILED TRAINERS — fix before full run: {failed_names}")
        sys.exit(1)
    print("✓ Every trainer cleared the smoke. Ready for the full RunPod run.")
except SystemExit:
    raise
except Exception as exc:
    print(f"summary parse failed: {exc}")
PYEOF

echo ""
echo "============================================================================"
echo "SMOKE complete. If every trainer is OK or skipped, run the FULL pipeline:"
echo "  bash scripts/runpod/runpod_full_pipeline.sh"
echo ""
echo "STOP THE POD NOW from RunPod console if you're not running full immediately"
echo "— RTX 4090 secure cloud is \$0.69/hr."
echo "============================================================================"
