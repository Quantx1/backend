#!/usr/bin/env bash
# scripts/check_frontend_hex_literals.sh
# Blocks new hex-literal Tailwind classes (``bg-[#XXXXXX]``, ``text-[#XXXXXX]``,
# ``border-[#XXXXXX]``, ``ring-[#XXXXXX]``, ``fill-[#XXXXXX]``, ``stroke-[#XXXXXX]``)
# in frontend/app/ and frontend/components/. Forces new code to use the
# Tailwind tokens defined in lib/tokens.ts + tailwind.config.ts + globals.css.
#
# Per v2 spec §9.5 / Foundation README — Foundation primitives never inline
# arbitrary hex; the same rule should apply to every consumer.
#
# Runs against staged file content only. Allowlist intentionally tiny —
# one-off brand-specific colors (e.g. ``#5DCBD8`` for the Alpha brand color
# used in 5 sites) live in the allowlisted files.

set -euo pipefail

PATTERN='(bg|text|border|ring|fill|stroke)-\[#[0-9A-Fa-f]{6}\]'

ALLOWLIST_PATHS=(
  'scripts/check_frontend_hex_literals.sh'
  # The Alpha brand color (#5DCBD8) is the only brand-specific 1-off
  # we let stay. If it ever migrates to a real token, drop these entries.
  'frontend/app/page.tsx'
  'frontend/app/(platform)/autopilot/page.tsx'
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
  # Only check frontend tsx / ts source files
  [[ "$file" != frontend/app/* && "$file" != frontend/components/* ]] && continue
  [[ "$file" != *.tsx && "$file" != *.ts ]] && continue
  if is_allowlisted "$file"; then continue; fi

  # Look for hex literals in the STAGED content only
  if hits=$(git show ":$file" 2>/dev/null | grep -E "$PATTERN" || true); [[ -n "$hits" ]]; then
    echo "❌ $file uses raw hex literal — replace with a token from lib/tokens.ts:"
    echo "$hits" | sed 's/^/    /'
    violations=$((violations + 1))
  fi
done < <(git diff --cached --name-only --diff-filter=AM)

if (( violations > 0 )); then
  echo
  echo "Commit blocked: $violations file(s) introduce raw hex Tailwind classes."
  echo "Use semantic tokens (down / up / warning / highlight / orange / primary etc.)."
  echo "If a new color is genuinely needed, add it to lib/tokens.ts +"
  echo "tailwind.config.ts + globals.css before referencing."
  exit 1
fi

exit 0
