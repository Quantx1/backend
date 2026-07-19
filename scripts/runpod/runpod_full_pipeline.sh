#!/usr/bin/env bash
# PR 212 — bulletproof end-to-end RunPod training pipeline.
#
# Designed for a fresh pod on:
#   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Idempotent — re-run safely. Total runtime ~7-9 hours, ~$5-6 of $15 budget
# at $0.69/hr secure cloud RTX 4090.
#
# Usage:
#   1. SSH/web terminal into the pod
#   2. Set env vars (paste once):
#        export SUPABASE_URL=...
#        export SUPABASE_ANON_KEY=...
#        export SUPABASE_SERVICE_ROLE_KEY=...
#        export B2_APPLICATION_KEY_ID=...
#        export B2_APPLICATION_KEY=...
#   3. Run:
#        bash scripts/runpod/runpod_full_pipeline.sh

set -euo pipefail

# ── 1. Sanity checks ───────────────────────────────────────────────────────
echo "=== Phase 1: sanity checks ==="
# Disable pipefail temporarily — `nvidia-smi | head` triggers SIGPIPE when
# head closes the pipe early, which `set -o pipefail` reports as failure.
set +o pipefail
nvidia-smi 2>&1 | head -20 || true
set -o pipefail

# Real GPU check via torch (works even if nvidia-smi pipe failed)
python -c "
try:
    import torch
    if not torch.cuda.is_available():
        print('CUDA not available — abort'); raise SystemExit(1)
    print('  GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('torch not yet installed — install phase will handle')" || true

for v in SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY B2_APPLICATION_KEY_ID B2_APPLICATION_KEY; do
    if [ -z "${!v:-}" ]; then
        echo "MISSING ENV VAR: $v — abort"
        exit 1
    fi
done

# Aliases (some code reads these alternate names)
export SUPABASE_SERVICE_KEY="${SUPABASE_SERVICE_KEY:-$SUPABASE_SERVICE_ROLE_KEY}"
export B2_KEY_ID="${B2_KEY_ID:-$B2_APPLICATION_KEY_ID}"
export B2_BUCKET="${B2_BUCKET:-quantx-models}"
export B2_BUCKET_MODELS="${B2_BUCKET_MODELS:-quantx-models}"
echo "env vars OK"

# ── 2. Caches on /workspace volume (50GB) ──────────────────────────────────
echo "=== Phase 2: redirect caches to /workspace ==="
mkdir -p /workspace/.cache/{pip,huggingface,torch}
ln -sfn /workspace/.cache/pip /root/.cache/pip
ln -sfn /workspace/.cache/huggingface /root/.cache/huggingface
ln -sfn /workspace/.cache/torch /root/.cache/torch
mkdir -p /workspace/.qlib && ln -sfn /workspace/.qlib /root/.qlib
df -h /workspace
echo "caches redirected"

# 2026-05-24 v1.1 fix — download parquet caches from B2 (moved to
# Phase 6.5 below because pandas + b2sdk aren't installed until Phase 4).

# ── 3. Repo ────────────────────────────────────────────────────────────────
echo "=== Phase 3: clone/pull repo ==="
cd /workspace
if [ ! -d quantx ]; then
    git clone https://github.com/Ri2506/quantx.git
fi
cd /workspace/quantx
git pull origin main
echo "on commit: $(git log --oneline -1)"

export PYTHONPATH="/workspace/quantx:${PYTHONPATH:-}"

# Save env vars for resume
cat > /workspace/.envrc <<EOF
export SUPABASE_URL="$SUPABASE_URL"
export SUPABASE_ANON_KEY="$SUPABASE_ANON_KEY"
export SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_SERVICE_ROLE_KEY"
export SUPABASE_SERVICE_KEY="$SUPABASE_SERVICE_KEY"
export B2_APPLICATION_KEY_ID="$B2_APPLICATION_KEY_ID"
export B2_APPLICATION_KEY="$B2_APPLICATION_KEY"
export B2_KEY_ID="$B2_KEY_ID"
export B2_BUCKET="$B2_BUCKET"
export B2_BUCKET_MODELS="$B2_BUCKET_MODELS"
export PYTHONPATH="/workspace/quantx:\${PYTHONPATH:-}"
EOF
chmod 600 /workspace/.envrc

# ── 4. Install — clean order, dependency-safe ─────────────────────────────
echo "=== Phase 4: install upstream libraries ==="

# Skip if already installed AND torch can import cleanly
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

    # 4c. Lightning + pytorch-forecasting (won't touch torch; fallback for TFT).
    echo "  installing pytorch-forecasting + lightning..."
    pip install --quiet "lightning>=2.0,<2.5"
    pip install --quiet pytorch-forecasting

    # 4c-bis. Nixtla neuralforecast — REQUIRED primary TFT backend.
    echo "  installing neuralforecast (Nixtla, REQUIRED)..."
    pip install --quiet "neuralforecast>=1.7"

    # 4d. TimesFM with --no-deps to prevent torch 2.11 upgrade. REQUIRED.
    # praxis + paxml pin conflicting jaxlib versions — drop them; timesfm
    # only needs jax/jaxlib + einshape + utilsforecast at inference time.
    # jax/jaxlib pinned to 0.4.x because 0.5+ requires numpy>=2 which
    # conflicts with torch 2.4's numpy<2 requirement.
    # wandb + typer satisfy import-time guards inside timesfm package init.
    echo "  installing timesfm (Google, REQUIRED) — no-deps to preserve torch 2.4..."
    pip install --quiet --no-deps "timesfm[torch]==1.2.7"
    pip install --quiet "jax[cpu]<0.5" "jaxlib<0.5" absl-py einshape utilsforecast
    pip install --quiet --no-deps wandb typer

    # 4e. chronos-forecasting — REQUIRED, --no-deps to preserve torch.
    echo "  installing chronos-forecasting (Amazon, REQUIRED) — no-deps..."
    pip install --quiet --no-deps chronos-forecasting
    # 2026-05-23 fix: pin huggingface-hub>=0.34.0 — transformers requires it
    # but base pytorch image ships 0.33.1, so "<1.0" alone left the old
    # version in place and Phase 5 trainer discovery exploded on import.
    pip install --quiet "transformers>=4.30" accelerate "huggingface-hub>=0.34.0,<1.0"

    # 4f. pyqlib — REQUIRED for qlib_alpha158. Manual deps because pip
    # otherwise tries to uninstall torch trio. cvxpy pinned <1.8 because
    # 1.8+ requires numpy>=2 which conflicts with torch 2.4's numpy<2.
    echo "  installing pyqlib (Microsoft, REQUIRED)..."
    pip install --quiet --no-deps pyqlib
    pip install --quiet pyyaml gym fire ruamel.yaml mlflow plotly redis-py-cluster \
        dill loguru pymongo "cvxpy<1.8" pydantic-settings python-redis-lock

    # 4g. Other ML deps — REQUIRED. Well-behaved, full deps OK.
    echo "  installing core ML deps (REQUIRED)..."
    # 2026-05-20 — `ta` library added. Imported by
    # backend/services/feature_engineering.py + signal_generator.py;
    # missing it breaks lgbm_signal_gate transitive import. Discovered
    # live on pod gvn94e8wa1ktev during training run.
    pip install --quiet jugaad-data statsmodels lightgbm xgboost \
        supabase httpx pyarrow yfinance b2sdk hmmlearn \
        stable-baselines3 gymnasium scikit-learn \
        optuna optuna-integration ta \
        pandas-market-calendars korean-lunar-calendar exchange-calendars

    # 4g-bis. mlfinpy — REQUIRED canonical AFML triple-barrier labeling.
    echo "  installing mlfinpy (canonical AFML labeling, REQUIRED)..."
    pip install --quiet mlfinpy

    # 4g-ter. Kronos OHLCV foundation model (AAAI 2026, NeoQuasar/Kronos-base).
    # REQUIRED — provides 256-dim embeddings used by lgbm_signal_gate when
    # KRONOS_ENABLED=1. Research repo (NOT pip-installable). We clone it
    # to /workspace/Kronos, install its requirements.txt, and set
    # KRONOS_PATH so ml/data/kronos_features.py can import `model`.
    echo "  installing kronos (NeoQuasar, AAAI 2026, REQUIRED) via clone + PYTHONPATH..."
    if [ ! -d /workspace/Kronos ]; then
        git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /workspace/Kronos
    else
        echo "  Kronos repo exists — pulling latest (auto-detect default branch)"
        # 2026-05-20 fix — Kronos repo uses `master` (not `main`); use
        # `git pull` without a branch arg so it follows upstream tracking.
        # If that fails (e.g. no upstream set on a depth=1 clone), fall
        # back to a fresh shallow clone.
        (cd /workspace/Kronos && git pull --ff-only 2>/dev/null) || {
            echo "  Kronos pull failed — re-cloning fresh"
            rm -rf /workspace/Kronos
            git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /workspace/Kronos
        }
    fi
    pip install --quiet -r /workspace/Kronos/requirements.txt
    export KRONOS_PATH="/workspace/Kronos"
    # Persist for future shells
    grep -q "KRONOS_PATH=/workspace/Kronos" ~/.bashrc \
        || echo 'export KRONOS_PATH=/workspace/Kronos' >> ~/.bashrc

    # 4h. Backend repo deps (--no-deps so they don't disturb torch)
    [ -f requirements-train.txt ] && pip install --quiet --no-deps -r requirements-train.txt

    # 4i. autogluon.timeseries install removed 2026-05-17 — chronos2_macro
    # dropped from v1; no other trainer requires autogluon.

    # 4j. CRITICAL: re-pin torch trio at the end. Some upstream package
    # (autogluon or transformers) may have upgraded torch. Force back to
    # 2.4.1 with --no-deps so we don't disturb the ecosystem.
    echo "  re-pinning torch trio to 2.4.1 (some deps may have upgraded it)..."
    pip install --quiet --force-reinstall --no-deps \
        "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
        --index-url https://download.pytorch.org/whl/cu124

    # 4k. CRITICAL: re-pin huggingface-hub at the very end (2026-05-23 fix).
    # The earlier line 145 install put hf-hub at 0.36+, but downstream
    # installs (Kronos requirements.txt, pyqlib deps, core ML deps) keep
    # downgrading it back to 0.33.1. transformers 4.57+ hard-requires
    # >=0.34.0 — without this, Phase 5 trainer discovery explodes.
    # --no-deps prevents pip from re-resolving and downgrading again.
    echo "  re-pinning huggingface-hub>=0.34.0 (downstream installs keep downgrading)..."
    pip install --quiet --force-reinstall --no-deps 'huggingface-hub>=0.34.0,<1.0'
    python -c "import huggingface_hub; print(f'  final huggingface-hub: {huggingface_hub.__version__}')"
    python -c "from transformers import AutoTokenizer; print('  transformers import OK')"
fi

# Final verify — abort if anything is still broken
python -c "
import torch, torch.onnx
assert torch.cuda.is_available(), 'CUDA not available'
import qlib, pytorch_forecasting, lightning, timesfm
import ml.data
print('all imports OK')
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
print('qlib:', qlib.__version__)
"
echo "install OK"

# ── 5. Discovery sanity ────────────────────────────────────────────────────
echo "=== Phase 5: trainer discovery ==="
python -c "from ml.training.discovery import discover_sorted; t=discover_sorted(); print(f'{len(t)} trainers'); [print(' ', x.name) for x in t]"

# ── 6. Smoke test — verifies B2 + Supabase + GPU all working end-to-end ───
echo "=== Phase 6: smoke test (regime_hmm only, ~10s) ==="

# Pre-check B2 auth so we fail fast with a clear message
python -c "
from b2sdk.v2 import InMemoryAccountInfo, B2Api
import os
info = InMemoryAccountInfo()
api = B2Api(info)
api.authorize_account('production', os.environ['B2_APPLICATION_KEY_ID'], os.environ['B2_APPLICATION_KEY'])
print('  B2 auth OK')
"

# Pre-check Supabase connectivity
python -c "
from backend.core.config import settings
print('  Supabase URL ok:', 'abraylc' in (settings.SUPABASE_URL or ''))
print('  Supabase keys present:', bool(settings.SUPABASE_ANON_KEY) and bool(settings.SUPABASE_SERVICE_KEY))
"

python -m ml.training.runner --only regime_hmm --promote 2>&1 | tee /workspace/quantx/smoke.log
if ! grep -q "ok=1" /workspace/quantx/smoke.log; then
    echo "SMOKE FAILED — review smoke.log before proceeding"
    exit 1
fi
echo "smoke OK — regime_hmm promoted, B2 + Supabase wired"

# ── 6.5. Restore parquet caches from B2 (post-install, pre-backfill) ─────
# 2026-05-24 v1.1 fix — pulls 5 parquet caches (sentiment_history,
# fundamentals_pit, fii_dii_history, sector_history, kronos_embeddings)
# from b2://quantx-models/data_caches/ so a fresh pod inherits the
# warm caches uploaded by previous backfill runs. On first ever run
# B2 has nothing → returns 0/5 → backfills regenerate everything.
echo "=== Phase 6.5: restore parquet caches from B2 ==="
python - <<'PYEOF' || echo "  (cache restore failed — non-fatal, backfills will rebuild)"
try:
    from ml.data.cache_sync import download_all, CACHE_FILES
    print(f"  attempting download of {len(CACHE_FILES)} cache files from B2...")
    n = download_all()
    print(f"  restored {n}/{len(CACHE_FILES)} caches from B2")
except Exception as exc:
    print(f"  cache restore skipped: {exc}")
PYEOF

# ── 7. Backfills (idempotent — skip if cache already populated) ───────────
echo "=== Phase 7: backfills (~30 min total, idempotent) ==="

# Fundamentals — skip if cache already has >= 1500 rows
if python -c "
import pandas as pd
from pathlib import Path
p = Path('/workspace/quantx/ml/data/cache/fundamentals_pit.parquet')
if p.exists():
    df = pd.read_parquet(p)
    assert len(df) >= 1500, f'only {len(df)} rows'
    print(f'  fundamentals already cached: {len(df)} rows / {df.symbol.nunique()} syms — skip')
else:
    raise SystemExit(1)
" 2>/dev/null; then
    echo "  fundamentals backfill SKIP (cache populated)"
else
    python scripts/data/backfill_fundamentals.py 2>&1 | tee /workspace/quantx/backfill_fund.log
    python -c "from ml.data.cache_sync import upload_cache; upload_cache('fundamentals_pit.parquet')" || true
fi

# Sentiment — skip if cache has >= 300 unique symbols today
if python -c "
import pandas as pd
from pathlib import Path
from datetime import date
p = Path('/workspace/quantx/ml/data/cache/sentiment_history.parquet')
if p.exists():
    df = pd.read_parquet(p)
    today = pd.Timestamp(date.today())
    today_rows = df[df['date'] == today]
    assert today_rows['symbol'].nunique() >= 300, f'only {today_rows[\"symbol\"].nunique()} today'
    print(f'  sentiment already cached: {len(df)} rows total, {today_rows[\"symbol\"].nunique()} today — skip')
else:
    raise SystemExit(1)
" 2>/dev/null; then
    echo "  sentiment backfill SKIP (cache populated today)"
else
    python scripts/data/backfill_sentiment.py 2>&1 | tee /workspace/quantx/backfill_sent.log
    python -c "from ml.data.cache_sync import upload_cache; upload_cache('sentiment_history.parquet')" || true
fi

# FII/DII (best-effort — NSE archive often blocks; safe to fail)
python scripts/data/backfill_fii_dii.py 2>&1 | tee /workspace/quantx/backfill_fii.log || true
python -c "from ml.data.cache_sync import upload_cache; upload_cache('fii_dii_history.parquet')" || true

# Sector backfill (already runs inside sentiment backfill, but ensure synced)
python -c "from ml.data.cache_sync import upload_cache; upload_cache('sector_history.parquet')" || true

# ── 8. Qlib provider build (idempotent) ───────────────────────────────────
echo "=== Phase 8: Qlib NSE provider build (~15 min) ==="

# Skip if provider already has 200+ symbols. Explicit -d check avoids
# pipefail trap from `ls` exit-2 on missing directory (regression bug).
QLIB_FEATURES_DIR="/root/.qlib/qlib_data/nse_data/features"
if [ -d "$QLIB_FEATURES_DIR" ]; then
    qlib_sym_count=$(find "$QLIB_FEATURES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
else
    qlib_sym_count=0
fi
echo "  current Qlib symbol count: $qlib_sym_count"
if [ "$qlib_sym_count" -ge 200 ]; then
    echo "  Qlib provider has $qlib_sym_count symbols — skip rebuild"
else
    rm -rf /root/.qlib/qlib_data/nse_data
    python scripts/data/ingest_nse_to_qlib.py 2>&1 | tee /workspace/quantx/qlib_ingest.log
fi

# ── 8b. Data quality report — HARD GATE before training ───────────────────
echo "=== Phase 8b: pre-training data quality report (blocking) ==="
# Production gate: refuse to train if any check fails. Adds ~30s.
if ! python scripts/data/data_quality_report.py --top-n 50 --start 2020-01-01 \
        2>&1 | tee /workspace/quantx/data_quality.log; then
    echo "❌ Data quality report failed. Fix blockers before re-running."
    exit 1
fi

# ── 8c. Per-trainer EDA + preprocessing audit — HARD GATE ──────────────────
# Checks feature distributions, label balance (CRITICAL for triple-barrier
# 3-class), feature-label IC, look-ahead leakage. Aborts if any blocker.
# Locked 2026-05-12 — no fallbacks, no skips.
echo "=== Phase 8c: per-trainer EDA + preprocessing audit (blocking) ==="
if ! python scripts/train/eda_report.py --universe 30 --period 5y \
        --report /workspace/quantx/eda_report.json \
        2>&1 | tee /workspace/quantx/eda_report.log; then
    echo "❌ EDA report flagged blockers. Inspect /workspace/quantx/eda_report.json"
    echo "   before re-running. No fallback — fix data layer or preprocessing first."
    exit 1
fi

# ── 9. Full training ───────────────────────────────────────────────────────
# Set SKIP_PHASE_9=1 to stop here — useful when running per-trainer validation
# one-by-one via `python scripts/train/validate_trainer.py X` after install + qlib
# build are done.
if [ "${SKIP_PHASE_9:-0}" = "1" ]; then
    echo "=== Phase 9: SKIPPED (SKIP_PHASE_9=1) ==="
    echo
    echo "Install + Qlib provider build complete. Next:"
    echo "  python scripts/train/validate_trainer.py regime_hmm    # start trainer #1"
    echo "  python -m ml.training.runner --only regime_hmm --promote   # publish"
    exit 0
fi

# 2026-05-24 — support TRAINER_FILTER env var so Option-B targeted runs
# can train only the missing trainers (e.g. "lgbm_signal_gate,intraday_lstm")
# instead of re-doing the 3 that already promoted (regime/qlib/tft).
# Empty/unset = train all discovered trainers (default behavior).
if [ -n "${TRAINER_FILTER:-}" ]; then
    echo "=== Phase 9: targeted training run (--only $TRAINER_FILTER) ==="
    python -m ml.training.runner --promote --json --only "$TRAINER_FILTER" 2>&1 | tee /workspace/quantx/runner.log
else
    echo "=== Phase 9: full --all training run (~5-6 hours) ==="
    python -m ml.training.runner --promote --json 2>&1 | tee /workspace/quantx/runner.log
fi

# ── 10. Summary ────────────────────────────────────────────────────────────
echo "=== Phase 10: summary ==="
python <<'PYEOF'
import json, re
try:
    with open('/workspace/quantx/runner.log') as f:
        text = f.read()
    m = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if not m:
        print("no JSON report found in runner.log"); raise SystemExit(0)
    reports = json.loads(m.group(0))
    print(f"\n{'trainer':<28} {'status':<8} {'sec':>7}  {'ver':<5} {'promo':<6} primary")
    print("-" * 80)
    for r in reports:
        v = f"v{r['version']}" if r.get('version') is not None else ""
        p = "PROD" if r.get('promoted') else ""
        m_ = r.get('metrics') or {}
        primary = f"{m_.get('primary_metric','?')}={m_.get('primary_value','?')}"
        print(f"{r['name']:<28} {r['status']:<8} {r.get('duration_sec',0):>7.0f}  {v:<5} {p:<6} {primary}")
        if r.get('error'): print(f"   ↳ {r['error'][:120]}")
    counts = {"ok":0, "skipped":0, "failed":0}
    for r in reports: counts[r.get('status','ok')] = counts.get(r.get('status','ok'),0)+1
    promoted = sum(1 for r in reports if r.get('promoted'))
    print("-" * 80)
    print(f"summary: ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']} promoted={promoted}")
except Exception as exc:
    print(f"summary parse failed: {exc}")
PYEOF

# ── Forensics: upload runner.log to B2 so we never lose diagnostics
#    even if the pod is stopped immediately after. 2026-05-24 — added
#    after a pod where lgbm_signal_gate + intraday_lstm crashed silently
#    (no model_versions row written by orchestrator) and the runner.log
#    was lost when the pod was killed.
echo ""
echo "=== Phase 11: upload runner.log to B2 (forensics) ==="
python - <<'PYEOF' || echo "  (runner.log B2 upload failed — non-fatal)"
import os
from pathlib import Path
from datetime import datetime, timezone
try:
    from backend.ai.registry.b2_client import get_b2_client
    log_path = Path("/workspace/quantx/runner.log")
    if not log_path.exists() or log_path.stat().st_size == 0:
        print("  runner.log missing or empty — nothing to upload")
        raise SystemExit(0)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_sha = os.environ.get("GIT_SHA", "unknown")[:8]
    remote_path = f"runner_logs/{ts}_{git_sha}.log"
    client = get_b2_client()
    client.upload_file(log_path, remote_path)
    print(f"  runner.log → b2://{os.environ.get('B2_BUCKET','quantx-models')}/{remote_path} ({log_path.stat().st_size} bytes)")
except Exception as exc:
    print(f"  runner.log upload skipped: {exc}")
PYEOF

echo ""
echo "============================================================================"
echo "Training session complete. STOP THE POD NOW from RunPod console — \$0.69/hr"
echo "============================================================================"
