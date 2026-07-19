#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Guards against re-introducing the DELETED strategy_selector module.
# NOTE: regime_detector.py was intentionally dropped from this pattern — it is
# LIVE production code (ml/regime_detector.py, imported by
# backend/ai/signals/generator.py); the planned ml->src migration never
# happened, so forbidding references to it was a false positive.
PATTERN='ml\.strategies\.strategy_selector|strategy_selector\.py'
HITS_FILE="$(mktemp)"

cleanup() {
  rm -f "${HITS_FILE}"
}
trap cleanup EXIT

echo "Running drift gate for deleted selector/regime references..."

if command -v rg >/dev/null 2>&1; then
  if rg -n --glob '*.py' --glob '*.md' --glob '*.ts' --glob '*.tsx' "${PATTERN}" "${ROOT_DIR}" >"${HITS_FILE}"; then
    echo "Drift gate failed. Found forbidden references:"
    cat "${HITS_FILE}"
    exit 1
  fi
else
  if grep -RInE \
    --include='*.py' \
    --include='*.md' \
    --include='*.ts' \
    --include='*.tsx' \
    "${PATTERN}" "${ROOT_DIR}" >"${HITS_FILE}"; then
    echo "Drift gate failed. Found forbidden references:"
    cat "${HITS_FILE}"
    exit 1
  fi
fi

echo "Drift gate passed."
