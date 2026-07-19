#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SYMBOLS="${SYMBOLS:-RELIANCE,TCS,INFY}"
PERIOD="${PERIOD:-1y}"
MIN_CONFLUENCE="${MIN_CONFLUENCE:-0.6}"
HOLD_DAYS="${HOLD_DAYS:-10}"
LOG_FILE="${LOG_FILE:-/tmp/backtest_smoke.log}"

echo "Running backtest smoke with symbols=${SYMBOLS}, period=${PERIOD}..."

python scripts/backtest/backtest_harness.py \
  --symbols "${SYMBOLS}" \
  --period "${PERIOD}" \
  --min-confluence "${MIN_CONFLUENCE}" \
  --hold-days "${HOLD_DAYS}" \
  >"${LOG_FILE}" 2>&1

PATTERN="traceback|ml\\.strategies\\.strategy_selector|regime_detector\\.py"
HAS_ISSUES="false"

if command -v rg >/dev/null 2>&1; then
  if rg -n -i "${PATTERN}" "${LOG_FILE}" >/dev/null; then
    HAS_ISSUES="true"
  fi
else
  if grep -Ein "${PATTERN}" "${LOG_FILE}" >/dev/null; then
    HAS_ISSUES="true"
  fi
fi

if [[ "${HAS_ISSUES}" == "false" ]]; then
  echo "Backtest smoke passed. Log: ${LOG_FILE}"
  exit 0
fi

echo "Backtest smoke failed. Problematic output:"
if command -v rg >/dev/null 2>&1; then
  rg -n -i "${PATTERN}" "${LOG_FILE}" || true
else
  grep -Ein "${PATTERN}" "${LOG_FILE}" || true
fi
exit 1
