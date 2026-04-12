#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/alignment_drift/simulate.py"

runs=(
  "uv run python $SCRIPT --prompt_cefr_level C1 --prompt_language English --model_size medium"
)

for cmd in "${runs[@]}"; do
  echo ">>> Running: $cmd"
  eval "$cmd"
done

