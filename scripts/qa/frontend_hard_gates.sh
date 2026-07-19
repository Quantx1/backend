#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <frontend_dir_relative_to_repo_root> [build]"
  echo "Example: $0 frontend true"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="$1"
RUN_BUILD="${2:-false}"
ABS_DIR="${ROOT_DIR}/${FRONTEND_DIR}"

if [[ ! -d "${ABS_DIR}" ]]; then
  echo "Frontend directory not found: ${ABS_DIR}"
  exit 1
fi

echo "Running frontend hard gates in ${FRONTEND_DIR}..."
cd "${ABS_DIR}"

npm ci
npx tsc --noEmit

if [[ "${RUN_BUILD}" == "true" ]]; then
  npm run build
fi

echo "Frontend hard gates passed for ${FRONTEND_DIR}."
