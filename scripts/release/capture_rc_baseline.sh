#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_FILE="${ROOT_DIR}/docs/release/RC_BASELINE.md"
TIMESTAMP_UTC="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

run_check() {
  local title="$1"
  shift
  local cmd=("$@")

  echo "## ${title}" >>"${OUTPUT_FILE}"
  echo '```bash' >>"${OUTPUT_FILE}"
  printf '%q ' "${cmd[@]}" >>"${OUTPUT_FILE}"
  echo >>"${OUTPUT_FILE}"
  echo '```' >>"${OUTPUT_FILE}"

  set +e
  local output
  output="$("${cmd[@]}" 2>&1)"
  local status=$?
  set -e

  echo '```text' >>"${OUTPUT_FILE}"
  if [[ -n "${output}" ]]; then
    echo "${output}" >>"${OUTPUT_FILE}"
  else
    echo "<no output>" >>"${OUTPUT_FILE}"
  fi
  echo "exit_code=${status}" >>"${OUTPUT_FILE}"
  echo '```' >>"${OUTPUT_FILE}"
  echo >>"${OUTPUT_FILE}"

  if [[ ${status} -ne 0 ]]; then
    echo "Baseline capture failed at: ${title}"
    exit ${status}
  fi
}

mkdir -p "${ROOT_DIR}/docs/release"

cat >"${OUTPUT_FILE}" <<EOF
# RC Baseline Verification

- Generated: ${TIMESTAMP_UTC}
- Purpose: Capture release-candidate hard-gate results before rollout.

EOF

cd "${ROOT_DIR}"

run_check "Import SignalGenerator" python -c "from backend.ai.signals import SignalGenerator; print('SignalGenerator import OK')"
run_check "Import BacktestEngine" python -c "from ml.backtest.backtest_engine import BacktestEngine; BacktestEngine(); print('BacktestEngine init OK')"
run_check "Import ComprehensiveBacktestEngine" python -c "from ml.backtest.comprehensive_backtest import ComprehensiveBacktestEngine; ComprehensiveBacktestEngine(); print('ComprehensiveBacktestEngine init OK')"
run_check "Strategy Test Suite" pytest -q backend/tests/test_long_strategies.py backend/tests/test_signal_save_contract.py
run_check "Backtest Harness Help" python scripts/backtest/backtest_harness.py --help
run_check "Backend Hard Gates Script" bash scripts/qa/backend_hard_gates.sh
run_check "Frontend Hard Gates" bash scripts/qa/frontend_hard_gates.sh frontend true
run_check "Drift Gate" bash scripts/qa/drift_gate.sh

echo "RC baseline captured at ${OUTPUT_FILE}"
