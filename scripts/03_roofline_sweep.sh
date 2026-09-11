#!/usr/bin/env bash
# M0 deliverable: measured decode throughput vs bandwidth ceiling, across the Qwen3 ladder.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate specdec
cd "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"

OUT="${SPECDEC_RESULTS:-$HOME/specdec-results}/roofline"
mkdir -p "$OUT"

for M in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; do
  SLUG="$(echo "$M" | tr '/' '_')"
  echo
  echo "################ $M ################"
  python bench/roofline.py --model "$M" --tokens 128 --warmup 16 \
    --json-out "$OUT/${SLUG}.json"
done

echo
echo "wrote JSON to $OUT"
