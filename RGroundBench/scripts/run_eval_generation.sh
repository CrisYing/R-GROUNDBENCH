#!/usr/bin/env bash
set -euo pipefail

python evaluation/eval_generation.py \
  --model "${1:-gpt-4o}" \
  --mode "${RGBENCH_MODE:-smi}" \
  --splits easy hard
