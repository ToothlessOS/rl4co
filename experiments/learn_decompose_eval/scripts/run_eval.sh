#!/usr/bin/env bash
# Convenience wrapper: set LKH binary env, dispatch to the learn_decompose_eval CLI.
#
# Usage:
#   ./scripts/run_eval.sh                       # default n=100, raw LKH-3
#   ./scripts/run_eval.sh 200                   # raw LKH-3 at n=200
#   SOLVER=bcc_lkh_cvrp ./scripts/run_eval.sh 100
#   NUM_INSTANCES=100 ./scripts/run_eval.sh 50
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Use the parent repo's .venv (which has rl4co, hydra, scikit-learn, vrplib,
# wandb, and the editable install of learn_decompose_eval).
PY="$REPO_ROOT/.venv/bin/python"

# LKH-3 binary + nrp_eval harness on sys.path.
export LDE_LKH_BINARY="$HERE/LKH-3.0.14/LKH"
export LDE_ROOT_DIR="$REPO_ROOT"
export LDE_DATA_DIR="$HERE/data"
export LDE_OUTPUT_DIR="$HERE/results"
export PYTHONPATH="$REPO_ROOT/experiments/nrp_eval:${PYTHONPATH:-}"

NUM_LOC="${1:-200}"
SOLVER="${SOLVER:-raw_lkh_cvrp}"
NUM_INSTANCES="${NUM_INSTANCES:-10}"
SEED="${SEED:-1234}"

cd "$REPO_ROOT"
exec "$PY" -m learn_decompose_eval eval \
    experiment="learn_decompose_eval/${SOLVER}" \
    env.generator_params.num_loc="${NUM_LOC}" \
    evaluate.num_instances="${NUM_INSTANCES}" \
    seed="${SEED}"
