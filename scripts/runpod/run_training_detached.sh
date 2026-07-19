#!/usr/bin/env bash
# Launch the unified training run in background, surviving terminal disconnect.
#
# Usage:
#   bash scripts/runpod/run_training_detached.sh
#
# The training process runs as a detached background job. You can:
#   - Close the web terminal — training keeps running
#   - tail -f /workspace/quantx/runner.log  — watch progress live
#   - Check alive: ps -p $(cat /workspace/quantx/runner.pid)
#   - Stop it:     kill $(cat /workspace/quantx/runner.pid)

set -euo pipefail

cd /workspace/quantx

# Source env vars
[ -f /workspace/.envrc ] && source /workspace/.envrc

export PYTHONPATH="/workspace/quantx:${PYTHONPATH:-}"

# Refuse to start if a previous run is still alive
if [ -f /workspace/quantx/runner.pid ]; then
    old_pid=$(cat /workspace/quantx/runner.pid)
    if ps -p "$old_pid" >/dev/null 2>&1; then
        echo "Training already running with PID $old_pid"
        echo "  tail -f /workspace/quantx/runner.log    # to watch"
        echo "  kill $old_pid                            # to stop"
        exit 0
    fi
fi

# Launch detached
echo "Launching training (detached, survives terminal close)..."
nohup python -m ml.training.runner --promote --json \
    > /workspace/quantx/runner.log 2>&1 &
PID=$!
echo "$PID" > /workspace/quantx/runner.pid
disown $PID

sleep 2
if ps -p "$PID" >/dev/null 2>&1; then
    echo "  PID:  $PID"
    echo "  log:  /workspace/quantx/runner.log"
    echo "  tail: tail -f /workspace/quantx/runner.log"
    echo "  kill: kill $PID"
    echo ""
    echo "Training will run for ~5 hours. STOP THE POD when summary appears."
else
    echo "FAILED to launch — check runner.log"
    tail -20 /workspace/quantx/runner.log
    exit 1
fi
