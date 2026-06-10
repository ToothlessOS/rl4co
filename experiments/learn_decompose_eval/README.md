# learn_decompose_eval

Evaluation pipeline for **learning to decompose** methods for CVRP, comparing raw LKH-3 vs LKH-3 + Barycenter-Clustering Decomposition (BCC) on RL4CO-style uniform random CVRP instances.

## Background

The `cvrp-decomposition` repo (Santini et al.) embeds Barycenter-Clustering Decomposition *inside* the HGS-CVRP/ALNS search loop, pulling the current best solution every 500 GA iterations, splitting its routes via k-means on route barycentres, re-solving each cluster as a sub-CVRP, and re-injecting the best sub-solutions into the master population.

This experiment asks the same question for **LKH-3**: does periodically interrupting a long Lin-Kernighan search, decomposing the current best, re-solving the clusters, and warm-restarting the master search improve the final cost on uniform random CVRP?

## Components

| Component | Role |
|-----------|------|
| `LKH-3.0.14/SRC/*` (patched) | C source with new `INTERMEDIATE_TOUR_FILE` parameter; writes the current best tour to disk on every improvement within a run |
| `learn_decompose_eval/solvers/lkh_format.py` | TSPLIB CVRP writer, tour parser, action converter |
| `learn_decompose_eval/solvers/decomposition.py` | Python port of `BarycentreClusteringDecomposition` (k-means on route barycentres, capacity-respecting split) |
| `learn_decompose_eval/solvers/orchestration.py` | `IntermediateTourWatcher` — launches patched LKH-3, polls for intermediate tours, decomposes, runs subproblems in parallel, warm-restarts master |
| `learn_decompose_eval/solvers/classical_lkh.py` | `RawLKH3CVRSolver` and `BarycentreLKH3CVRSolver` (registered with `nrp.solvers.SolverRegistry`) |
| `configs/` | Hydra config tree |

## Build LKH-3

```bash
./scripts/build_lkh.sh
```

## Run

```bash
# raw LKH-3 on n=100 (1000 instances, 60s time budget per instance)
./scripts/run_eval.sh 100

# LKH-3 + BCC on n=100
LDE_EVAL_OVERRIDES='solver_name=bcc_lkh_cvrp' \
    uv run python -m learn_decompose_eval eval \
    experiment=learn_decompose_eval/bcc_lkh_cvrp \
    env.generator_params.num_loc=100


cd /home/toothlessos/Projects/nrp/rl4co/experiments/learn_decompose_eval
./scripts/build_lkh.sh                # rebuild patched LKH-3 (already done)
./scripts/run_eval.sh 100              # raw LKH-3, n=100
SOLVER=bcc_lkh_cvrp ./scripts/run_eval.sh 100  # LKH-3 + BCC, n=100

```

## Tests

```bash
uv run pytest tests/ -v
```

## Notes

- The C patch is minimal: 4 single-line additions to LKH-3 source, plus a small block in `FindTour.c` to call `WriteTour(IntermediateTourFileName, ...)` on every trial-level improvement. The un-patched reference is preserved as `LKH-3.0.14.tgz`.
- The capacity-respecting cluster split is a deliberate divergence from the C++ original: the HGS sub-GA handled capacity implicitly; LKH-3 needs a feasible sub-CVRP, so we enforce `sum(demand) <= vehicle_capacity` per cluster post-hoc.
- Wandb project: `nrp-learn-decompose`.
