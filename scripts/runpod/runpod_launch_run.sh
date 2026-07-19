#!/usr/bin/env bash
# ============================================================================
# Quant X — Phase H launch run on RunPod
# ----------------------------------------------------------------------------
# Provisions a 4090 spot pod, runs the unified training pipeline, terminates.
# Designed for one-shot use: ~₹150–250 burned on a single 4-hour training run.
#
# Prerequisites on your Mac (one-time):
#   1. brew install runpod/runpodctl/runpodctl    # the CLI
#   2. runpodctl config set apiKey <your-key>     # from runpod.io → Settings → API Keys
#   3. Add your SSH public key in runpod.io → Settings → SSH Public Keys
#
# Then:  bash scripts/runpod/runpod_launch_run.sh
# ============================================================================

set -euo pipefail

# ── Required env (export these before running) ─────────────────────────────
: "${REPO_URL:?Set REPO_URL=git@github.com:you/Swing_AI_Final.git}"
: "${B2_KEY_ID:?Set B2_KEY_ID}"
: "${B2_APP_KEY:?Set B2_APP_KEY}"
: "${B2_BUCKET:?Set B2_BUCKET}"
: "${DATABASE_URL:?Set DATABASE_URL}"

# Optional knobs
GPU_TYPE="${GPU_TYPE:-NVIDIA GeForce RTX 4090}"   # cheap + sufficient
TEMPLATE_ID="${TEMPLATE_ID:-runpod-pytorch-241}"   # Pytorch 2.4 + CUDA 12.1
DISK_GB="${DISK_GB:-40}"

echo "▶ Creating RunPod 4090 spot pod..."
POD_ID=$(runpodctl create pod \
    --name "quantx-train-$(date +%s)" \
    --gpuType "$GPU_TYPE" \
    --gpuCount 1 \
    --bid 0.20 \
    --containerDiskInGb "$DISK_GB" \
    --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04" \
    --ports "22/tcp" \
    --volumeInGb 0 \
    --communityCloud \
    --json | jq -r '.id')

echo "▶ Pod $POD_ID requested. Waiting for SSH..."
for i in {1..60}; do
    SSH_HOST=$(runpodctl get pod "$POD_ID" --json 2>/dev/null | jq -r '.publicIp // empty')
    SSH_PORT=$(runpodctl get pod "$POD_ID" --json 2>/dev/null | jq -r '.ports[] | select(.privatePort==22) | .publicPort // empty' | head -1)
    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then break; fi
    sleep 5
done
[[ -z "$SSH_HOST" ]] && { echo "Pod never came up"; exit 1; }
echo "✔ SSH ready: root@$SSH_HOST:$SSH_PORT"

# ── Bootstrap script that runs on the pod ──────────────────────────────────
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT root@$SSH_HOST"

echo "▶ Bootstrapping repo + venv on pod..."
$SSH bash -s <<EOF
set -e
git clone "$REPO_URL" Swing_AI_Final
cd Swing_AI_Final
python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install -r requirements-train.txt
EOF

echo "▶ Running unified training pipeline (~3-4h on 4090)..."
$SSH bash -s <<EOF
cd Swing_AI_Final
source .venv/bin/activate
export B2_KEY_ID='$B2_KEY_ID'
export B2_APP_KEY='$B2_APP_KEY'
export B2_BUCKET='$B2_BUCKET'
export DATABASE_URL='$DATABASE_URL'
export GIT_SHA=\$(git rev-parse --short HEAD)
export TRAINED_BY="rishi-runpod-\$(date +%F)"
python -m ml.training.runner --all --promote --json 2>&1 | tee launch.json
EOF

echo "▶ Fetching launch.json back to local..."
scp -P "$SSH_PORT" -o StrictHostKeyChecking=no "root@$SSH_HOST:/root/Swing_AI_Final/launch.json" "./launch.json"

echo "▶ Tearing down pod $POD_ID..."
runpodctl remove pod "$POD_ID"

echo "✅ Done. Report saved to ./launch.json"
echo "   Verify with: jq '.[] | {name, status, version}' launch.json"
