#!/usr/bin/env bash
# Resumable runner for the plain-LLM baseline arms (Act=Muse, Mut=Muse).
#
# Runs both arms over several seeds against a Muse Spark target. Each (arm, seed) run is
# cached: run_ablation skips any run that already completed (a final_state.json with matching
# experiment settings), so re-running this script resumes from the last unfinished run.
# Runtime settings (target URL, concurrency) are ignored by the cache, so you may resume with
# different values. Safe to interrupt (Ctrl-C) and restart.
#
# Requires MODEL_API_KEY in the environment (do not hardcode it). Reads:
#   TARGET       target A2A URL              (default http://localhost:5050)
#   SEEDS        space-separated seed list   (default "1 2 3 4 5")
#   CONCURRENCY  parallel workers per run    (default 2, kept low to avoid rate limits)
#   OUTPUT_DIR   run output directory        (default outputs)
#
# Usage:
#   MODEL_API_KEY=... util/a2a_agent.py --port 5050 --provider meta &   # start the target first
#   MODEL_API_KEY=... scripts/run_muse_arms.sh                          # run (or resume)
#   STATUS=1 scripts/run_muse_arms.sh                                   # show what is done/pending
set -euo pipefail

: "${MODEL_API_KEY:?set MODEL_API_KEY (Meta Model API key) before running}"
export MUTATION_ENGINE_PROVIDER=meta

TARGET="${TARGET:-http://localhost:5050}"
SEEDS="${SEEDS:-1 2 3 4 5}"
CONCURRENCY="${CONCURRENCY:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

EXTRA=()
[ "${STATUS:-0}" = "1" ] && EXTRA+=(--status)

for seed in $SEEDS; do
  for arm in "--action-selector muse" "--mutation-selector muse"; do
    # shellcheck disable=SC2086
    "$PY" -m src.experiments.run_ablation --target "$TARGET" \
      --iterations 7 --mutations 4 --seed "$seed" --mutation-llm meta \
      --fitness-judge meta --max-concurrency "$CONCURRENCY" --output-dir "$OUTPUT_DIR" \
      $arm ${EXTRA[@]+"${EXTRA[@]}"}
    [ "${STATUS:-0}" = "1" ] || sleep 5
  done
done
[ "${STATUS:-0}" = "1" ] || echo "ALL MUSE RUNS DONE (or already cached)"
