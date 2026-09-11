#!/usr/bin/env bash
# Activate the specdec env and run a command from the project root.
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/.../scripts/run.sh python bench/roofline.py
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate specdec
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
exec "$@"
