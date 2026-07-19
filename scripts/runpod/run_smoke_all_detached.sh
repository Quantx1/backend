#!/usr/bin/env bash
# Launch the full smoke batch (install + qlib build + per-trainer smokes)
# in background, surviving terminal disconnect.
#
# Usage on the RunPod web terminal:
#   cd /workspace/quantx
#   bash scripts/runpod/run_smoke_all_detached.sh
#
# Then it's safe to:
#   - Close the web terminal — work keeps running
#   - Re-open it later and run: tail -f /workspace/smoke.log
#   - Check alive: ps -p $(cat /workspace/smoke.pid)
#   - Stop it:    kill $(cat /workspace/smoke.pid)
#
# What runs:
#   1. scripts/runpod/runpod_smoke_pipeline.sh  (install + Qlib build + data audit)
#   2. scripts/ops/smoke_all.py               (10-stock smoke per trainer)
#
# Total runtime: ~3 hours on a fresh pod (~$2 of GPU). Subsequent runs
# skip the install + Qlib build, so ~1.5-2 hours.

set -euo pipefail

cd /workspace/quantx

# Refuse to start if a previous batch is still alive
if [ -f /workspace/smoke.pid ]; then
    old_pid=$(cat /workspace/smoke.pid)
    if ps -p "$old_pid" >/dev/null 2>&1; then
        echo "Smoke batch already running with PID $old_pid"
        echo "  tail -f /workspace/smoke.log    # to watch"
        echo "  kill $old_pid                    # to stop"
        exit 0
    fi
fi

# Pull latest
git pull origin main || true

# The wrapper script that runs install pipeline THEN per-trainer smokes.
# Keep this inline so we don't need to ship another helper.
cat > /workspace/_smoke_runner.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/quantx

echo "========================================================"
echo "PHASE A — install + Qlib build (runpod_smoke_pipeline.sh)"
echo "Started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================"
# Skip Phase 9 (runner --all) — smoke_all.py runs each trainer at the
# same scale in Phase B. Avoid double-burn on GPU.
SKIP_PHASE_9=1 bash scripts/runpod/runpod_smoke_pipeline.sh

echo ""
echo "========================================================"
echo "PHASE B — per-trainer 10-stock smoke (smoke_all.py)"
echo "Started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================"
python scripts/ops/smoke_all.py

echo ""
echo "========================================================"
echo "ALL PHASES COMPLETE — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================"
SCRIPT
chmod +x /workspace/_smoke_runner.sh

echo "Launching smoke batch (detached, survives terminal close)..."
nohup bash /workspace/_smoke_runner.sh > /workspace/smoke.log 2>&1 &
PID=$!
echo "$PID" > /workspace/smoke.pid
disown $PID

sleep 2
if ps -p "$PID" >/dev/null 2>&1; then
    echo ""
    echo "✅ Smoke batch launched"
    echo "   PID:    $PID"
    echo "   log:    /workspace/smoke.log"
    echo ""
    echo "Safe to close this terminal now. To monitor later:"
    echo "   tail -f /workspace/smoke.log                      # live log"
    echo "   tail -f /workspace/quantx/smoke_all_report.json   # per-trainer json"
    echo "   ps -p \$(cat /workspace/smoke.pid)                # alive check"
    echo "   kill \$(cat /workspace/smoke.pid)                 # to stop"
    echo ""
    echo "Expected runtime: ~3 hours fresh pod / ~1.5 hours cached."
else
    echo "❌ FAILED to launch — check /workspace/smoke.log"
    tail -20 /workspace/smoke.log
    exit 1
fi
