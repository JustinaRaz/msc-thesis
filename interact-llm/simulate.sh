#!/usr/bin/env bash

SCRIPT="src/scripts/alignment_drift/simulate.py"

runs=(
  "uv run python $SCRIPT --prompt_cefr_level B1 --prompt_language Lithuanian --model_size medium --student_constrain"
)

for cmd in "${runs[@]}"; do
  echo ">>> Running: $cmd"
  eval "$cmd"
done

