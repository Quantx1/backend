#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <rc-name>"
  echo "Example: $0 2026-02-confluence-long-only"
  exit 1
fi

RC_NAME="$1"
BRANCH_NAME="rc/${RC_NAME}"
TAG_NAME="rc-${RC_NAME}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Current directory is not inside a git repository."
  exit 1
fi

echo "Creating release-candidate branch and tag..."
echo "Branch: ${BRANCH_NAME}"
echo "Tag: ${TAG_NAME}"

git checkout -b "${BRANCH_NAME}"
git tag -a "${TAG_NAME}" -m "Release candidate ${TAG_NAME}"

echo "Created ${BRANCH_NAME} and ${TAG_NAME}."
