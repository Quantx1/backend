#!/usr/bin/env bash
# scripts/check_legacy_branding.sh
# Blocks commits that re-introduce deprecated branding.
# Runs against staged file content (not full repo) for speed.

set -euo pipefail

PATTERNS=(
  'SwingMax'
  'QuantAI'
  'Whales Auto'
  'Intellectia'
  'SwingLens'
  'AlphaRank'
  'RegimeIQ'
  'SignalIQ'
  'TickPulse'
  'HorizonCast'
  'ToneScan'
  'TradingView-inspired'
  'Intellectia.ai Design System'
)

ALLOWLIST_PATHS=(
  'scripts/check_legacy_branding.sh'
  'docs/cleanup-'
  'docs/superpowers/specs/'
  'docs/superpowers/plans/'
  'tests/scripts/test_check_legacy_branding.sh'
  '.claude/'
)

is_allowlisted() {
  local file="$1"
  for prefix in "${ALLOWLIST_PATHS[@]}"; do
    [[ "$file" == "$prefix"* ]] && return 0
  done
  return 1
}

violations=0
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  [[ ! -f "$file" ]] && continue
  is_allowlisted "$file" && continue
  for pattern in "${PATTERNS[@]}"; do
    if git diff --cached "$file" | grep -q "^+.*$pattern"; then
      echo "❌ $file introduces deprecated brand '$pattern'"
      violations=$((violations + 1))
    fi
  done
done < <(git diff --cached --name-only --diff-filter=ACMR)

if [[ $violations -gt 0 ]]; then
  echo ""
  echo "Commit blocked: $violations branding violation(s)."
  echo "If you are RECORDING a rename or referencing history (specs/plans), add the file path to ALLOWLIST_PATHS in scripts/check_legacy_branding.sh."
  exit 1
fi

exit 0
