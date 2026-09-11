#!/usr/bin/env bash
# Create the GitHub remote and push. Run AFTER `gh auth login`.
#
# PRIVATE by design: this repo will contain baselines and analysis built on
# Prof. Liu's group's unpublished directions (DFlash, adaptive draft scheduling).
# Make it public only after checking with him. Flipping private -> public later
# is trivial; un-publishing something is not.
set -euo pipefail

REPO_NAME="${1:-dsc-capstone-specdec}"
GH_USER="anirudh9280"

if ! gh auth status >/dev/null 2>&1; then
  echo "Not authenticated. Run:  gh auth login" >&2
  exit 1
fi

gh repo create "$GH_USER/$REPO_NAME" \
  --private \
  --source=. \
  --remote=origin \
  --description "Speculative decoding baselines for UCSD DSC capstone D38 (Efficient AI)" \
  --push

echo
echo "pushed -> https://github.com/$GH_USER/$REPO_NAME (private)"
