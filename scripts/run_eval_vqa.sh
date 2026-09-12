#!/usr/bin/env bash
set -euo pipefail

python evaluation/eval_vqa.py \
  --model "${1:-gpt-4o}" \
  --mode "${RGBENCH_MODE:-img_smi}" \
  --split easy_basic medium_basic hard_basic
