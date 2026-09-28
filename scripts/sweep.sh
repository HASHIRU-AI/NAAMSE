#!/usr/bin/env bash
# Run each ablation arm over several seeds with src.experiments.run_ablation.
#
# Usage: scripts/sweep.sh <target-url> [seeds...]
#   e.g. scripts/sweep.sh http://localhost:5000 1 2 3 4 5
# Env overrides: ITERATIONS (default 7), MUTATIONS (default 4),
#                OUTPUT_DIR (default outputs), DRY_RUN=1 to only print configs.
set -euo pipefail

TARGET="${1:?usage: scripts/sweep.sh <target-url> [seeds...]}"
shift
if [ "$#" -gt 0 ]; then SEEDS=("$@"); else SEEDS=(1 2 3 4 5); fi

ITERATIONS="${ITERATIONS:-7}"
MUTATIONS="${MUTATIONS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXTRA=()
[ "${DRY_RUN:-0}" = "1" ] && EXTRA+=(--dry-run)

# One line per arm: the ablation flags that differ from the baseline.
ARMS=(
  "--fitness-judge meta"                                              # baseline (Meta judges)
  "--fitness-judge meta --action-selector uniform"                    # RQ1 control
  "--fitness-judge meta --action-selector jev"                        # RQ1
  "--fitness-judge jev"                                               # RQ2
  "--fitness-judge meta --mutation-selector jev"                      # RQ3
  "--fitness-judge meta --objective coverage"                         # coverage baseline
  "--fitness-judge meta --objective coverage --action-selector jev"   # RQ1 under coverage
  "--fitness-judge meta --objective coverage --mutation-selector jev" # RQ3 under coverage
)

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    echo "=== seed=${seed} arm: ${arm}"
    # shellcheck disable=SC2086  # arm is intentionally word-split into flags
    uv run python -m src.experiments.run_ablation \
      --target "${TARGET}" --iterations "${ITERATIONS}" --mutations "${MUTATIONS}" \
      --seed "${seed}" --output-dir "${OUTPUT_DIR}" --mutation-llm meta \
      ${arm} "${EXTRA[@]+"${EXTRA[@]}"}"
  done
done
