# Implementation Report: LKH-3 ± Barycenter-Clustering CVRP Evaluation

## 1. Background & Motivation

The `cvrp-decomposition` repo (Santini et al., _Computers & Operations Research_ 2023) embeds **Barycenter-Clustering Decomposition (BCC)** *inside* the HGS-CVRP and ALNS search loops. The decomposition is invoked periodically (every `decompositionIters=500` GA iterations), the master routes' barycentres are clustered with k-means into `k = ⌈n / targetMaxSpCustomers⌉` groups, each group is solved as a sub-CVRP, and the best sub-solutions are re-injected into the master population. This is a "search-space diversification" move — the master search periodically *interrupts itself*, decomposes, and uses the sub-solutions as new starting points.

This experiment asks the same question for **LKH-3** as the master solver. LKH-3 is a single-tour Lin-Kernighan-Helsgott local search with no population, so the literal "pull from population" pattern doesn't fit. We implement the analog:

- A small C patch to LKH-3 that exposes the current-best tour on every trial-level improvement.
- A Python orchestrator that runs LKH-3 in a budget-bounded loop, decomposes intermediate tours, runs subproblems in parallel, and warm-restarts the master with the stitched sub-solutions.
- An evaluation harness that compares raw LKH-3 vs. LKH-3 + BCC on RL4CO-style uniform random CVRP instances.

## 2. Architecture

```
              ┌─────────────────────────────────────┐
              │  Hydra config (env, sizes, seeds)   │
              └──────────────┬──────────────────────┘
                             ▼
              ┌─────────────────────────────────────┐
              │  python -m learn_decompose_eval    │
              │   eval experiment=…/{raw,bcc}_lkh_cvrp
              └──────────────┬──────────────────────┘
                             ▼
              ┌─────────────────────────────────────┐
              │  nrp_eval harness (reused)          │
              │   - build CVRPEnv, batch=1000        │
              │   - for each instance:              │
              │       solver.solve(td_instance)    │
              │   - log per-instance to wandb      │
              └──────────────┬──────────────────────┘
                             ▼
   ┌─────────────────────┐         ┌────────────────────────┐
   │ RawLKH3CVRSolver    │         │ BarycentreLKH3CVRSolver│
   │  - subprocess: LKH  │         │  - patch LKH + IPC loop│
   │  - parse .tour      │         │  - decompose + stitch │
   │  - return actions   │         │  - return actions     │
   └──────────┬──────────┘         └──────────┬─────────────┘
              │                               │
              └────────────┬──────────────────┘
                           ▼
              ┌─────────────────────────────────────┐
              │  LKH-3 binary (patched)             │
              │  - INTERMEDIATE_TOUR_FILE on every  │
              │    trial-level improvement         │
              └─────────────────────────────────────┘
```

The orchestration loop (per instance, BCC mode):

1. Launch patched LKH-3 in background with `RUNS=1, TIME_LIMIT=t_seg, INTERMEDIATE_TOUR_FILE=int.tour, OUTPUT_TOUR_FILE=final.tour`.
2. Poll `int.tour` mtime at 100ms granularity.
3. When the file updates: parse the current-best tour, run BCC decomposition, solve each subproblem in a `ThreadPoolExecutor` (each sub-LKH has its own `.vrp` + warm-start from the corresponding sub-tour of the parent).
4. Stitch the best sub-solutions into a new initial tour; SIGTERM the running master; relaunch with `INITIAL_TOUR_FILE=…`.
5. Continue until `max_total_s` is exhausted; read `final.tour` for the result.

## 3. What Was Built

### 3.1 LKH-3 source patch (`LKH-3.0.14/SRC/`)

Four small additions — minimal, easy to review:

| File | Change |
|------|--------|
| `INCLUDE/LKH.h` | Declare `char *IntermediateTourFileName;` |
| `LKH.c` | Define the global |
| `ReadParameters.c` | Parse the new `INTERMEDIATE_TOUR_FILE = <path>` parameter |
| `FindTour.c` | Call `WriteTour(IntermediateTourFileName, BetterTour, BetterCost);` **unconditionally** on every better-tour update (the existing `OUTPUT_TOUR_FILE` write is still gated on global-best) |

Build: `cd LKH-3.0.14 && make clean && make` (or use `scripts/build_lkh.sh`).

### 3.2 Python package (`learn_decompose_eval/`)

```
learn_decompose_eval/
├── __init__.py
├── __main__.py            # python -m learn_decompose_eval
├── cli.py                 # argparse + Hydra
├── solvers/
│   ├── __init__.py
│   ├── lkh_format.py      # TSPLIB CVRP writer/parser, action converter
│   ├── decomposition.py   # BarycentreClusteringDecomposer
│   ├── orchestration.py   # IntermediateTourWatcher
│   └── classical_lkh.py   # RawLKH3CVRSolver, BarycentreLKH3CVRSolver
└── data/
    └── __init__.py
```

#### `lkh_format.py`
- `cvrp_td_to_lkh_problem(td)` — RL4CO CVRP TensorDict → TSPLIB CVRP string. Uses **Uchoa convention** (depot=1, customers=2..N+1). `EDGE_WEIGHT_TYPE: EXPLICIT, EDGE_WEIGHT_FORMAT: FULL_MATRIX` with precomputed L2 distances × `LKH_SCALING_FACTOR = 100_000`. Critical: the matrix section is emitted as values only (no leading row index), because LKH-3's `Read_EDGE_WEIGHT_SECTION` uses `fscanf("%lf")` to consume exactly `Dim*Dim` doubles and any extra leading integers throw off the count.
- `parse_lkh_tour(path, depot_id=1)` — `.tour` file → list of routes (1-indexed, depot repeated at route boundaries).
- `routes_to_action(routes, num_loc, depot_id=1)` — LKH 1-indexed → CVRPEnv 1-indexed action format. Customer id `n` (2..N+1 in Uchoa) → action value `n-1` (1..N). Phantom depots (ids > N+1) → emit `0`.
- `LKHParameters` — dataclass for `.par` file generation with `WriteTour`. Includes a `salesmen` field that emits `SALESMEN = {n}` (used for warm-started subproblems so the problem's expanded DIMENSION matches the warm-start tour's DIMENSION).
- `write_cvrp_initial_tour(path, customer_seq, num_loc, salesmen, name)` — TSPLIB TOUR writer with phantom depots, used for the master's restart initial tour.
- `write_subproblem_initial_tour(path, parent_routes, num_loc, name)` — TSPLIB TOUR writer that preserves the parent route structure (one segment per parent route, depot-separated) for warm-starting subproblems.

#### `decomposition.py`
- `BarycentreClusteringDecomposer` — Python port of `BarycentreClusteringDecomposition.cpp`:
  1. Compute barycentres of non-empty master routes.
  2. `k = ⌈n / target_max_subproblem_size⌉` (capped by `#non-empty routes` because k-means needs `n_samples ≥ k`).
  3. sklearn `KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=100, tol=1e-2, random_state=0)` — matches the C++ `KMeans.h:180-245` parameters.
  4. Empty routes distributed round-robin to clusters.
  5. **No post-hoc capacity split** (deliberate departure from the prior implementation). Each `Subproblem` carries the parent routes assigned to its cluster (`parent_routes` field, subproblem-local 1-indexed Uchoa customer ids, depot implicit) and `n_parent_routes` (= `SALESMEN` for the subproblem). Per-route capacity-feasibility is inherited from the parent, so the subproblem is provably feasible by construction.
- `Subproblem` — dataclass with `customer_ids, xy, demand, capacity, depot_xy, parent_routes, n_parent_routes`; `to_td()` converts to a CVRPEnv-compatible TensorDict.

#### `orchestration.py`
- `OrchestratorConfig` — dataclass for the watcher.
- `IntermediateTourWatcher` — the main orchestrator. Manages the master LKH-3 subprocess, polls the intermediate-tour file, decomposes, runs subproblems in parallel, restarts master with warm start.
- `IntermediateTourWatcher.solve(td)` — returns `(routes, total_seconds)`.
- `_solve_subproblems._one(sp)` — for each subproblem: write a warm-start TSPLIB TOUR from `sp.parent_routes` (via `write_subproblem_initial_tour`), pass it as `INITIAL_TOUR_FILE` and `SALESMEN = n_parent_routes` to LKH-3, and log `sub_improved = sub_cost < parent_cost` (cost comparison in scaled units).
- Edge case: subproblems with `n_parent_routes == 1` skip the LKH-3 call and return the parent route directly.
- `_mtime`, `_read_tour_safely` (with retry on partial writes), `_infer_depot_id`, `_routes_zero_indexed`, `_stitch_routes`, `_write_initial_tour`, `_compute_parent_route_cost_scaled` — internal helpers.

#### `classical_lkh.py`
- `RawLKH3CVRSolver(ClassicalSolver)` — `@SolverRegistry.register("raw_lkh_cvrp", env_names=("cvrp",))`. One LKH-3 invocation per instance, fixed time budget.
- `BarycentreLKH3CVRSolver(ClassicalSolver)` — `@SolverRegistry.register("bcc_lkh_cvrp", env_names=("cvrp",))`. Wraps `IntermediateTourWatcher`.
- Both inherit `solve(td)` from a `ClassicalSolver`-style helper that handles device round-trip + `env.get_reward` computation.
- `ThreadPoolExecutor(num_workers)` for per-instance parallelism.

### 3.3 Hydra config tree (`configs/`)

Mirrors `experiments/nrp_eval/configs/`, with these specific files:
- `main.yaml` — defaults to `experiment: learn_decompose_eval/raw_lkh_cvrp`
- `env/cvrp.yaml` — `rl4co.envs.CVRPEnv`, `num_loc: 100` default
- `logger/wandb.yaml` — project `nrp-learn-decompose`
- `experiment/learn_decompose_eval/raw_lkh_cvrp.yaml` — tags `["raw-lkh3", "cvrp"]`, `max_runtime_s: 60.0`, `num_workers: 4`
- `experiment/learn_decompose_eval/bcc_lkh_cvrp.yaml` — tags `["bcc-lkh3", "cvrp"]`, `max_total_s: 60.0`, `decompose_every_s: 5.0`, `num_workers: 4`, `target_max_subproblem_size: 200`

### 3.4 Scripts

- `scripts/build_lkh.sh` — `cd LKH-3.0.14 && make clean && make`
- `scripts/run_eval.sh <num_loc>` — sets `LDE_LKH_BINARY`, `LDE_ROOT_DIR`, etc., and dispatches to `python -m learn_decompose_eval eval`. `SOLVER=bcc_lkh_cvrp` switches to the decomposed variant.

## 4. Design Decisions

| Decision | Rationale |
|---------|-----------|
| **LKH-3 C patch over `lkh` Python package** | The `lkh` Python package is a thin wrapper that doesn't expose intermediate tours. A 4-line C patch is small, auditable, and gives us full control over the per-trial output. |
| **EXPLICIT FULL_MATRIX (not EUC_2D)** | RL4CO coordinates are floats in [0, 1]. LKH-3's `EUC_2D` uses fixed-point integer arithmetic with squared distances up to 1e10, which overflows int32. Pre-computing the L2 distance matrix in Python and embedding it as `EXPLICIT FULL_MATRIX` is the only safe path. |
| **Uchoa CVRP convention (depot=1, customers=2..N+1)** | This is the standard TSPLIB CVRP format used by the Uchoa benchmark set and is well-supported by LKH-3's CVRP solver. The internal LKH-3 convention (depot=last node) is more fragile and varies between versions. |
| **k-means++ via sklearn** | Mirrors the C++ `KMeans.h:180-245` parameters exactly (init='k-means++', n_init=1, max_iter=100, tol=1e-2, single-threaded). No need to reimplement k-means. |
| **Route-preserving subproblems (no post-hoc capacity split)** | Each `Subproblem` carries the parent routes that k-means assigned to its cluster, in subproblem-local 1-indexed Uchoa ids. The sub-LKH-3 is warm-started with these parent routes as `INITIAL_TOUR_FILE`, with `SALESMEN = n_parent_routes` so the problem's expanded DIMENSION matches the tour. Per-route capacity-feasibility is therefore inherited from the parent — no demand-descending split is needed. The LKH-3 LK search on the warm-start acts as the "best-of-two" merge from Santini et al. natively. |
| **ThreadPoolExecutor for subproblem parallelism** | LKH-3 is single-threaded; running K instances in parallel gives near-linear speedup on the subproblem phase. `subprocess.Popen` with `preexec_fn=os.setsid` for clean SIGTERM of the master. |
| **mtime polling at 100ms (not inotify)** | LKH-3 writes the intermediate tour non-atomically; 100ms polling is robust and doesn't require extra deps. |
| **Reuse `nrp_eval` harness** | Avoids re-implementing the eval pipeline; we get wandb logging, per-instance tables, and the `Solver` ABC for free. The `cli.py` adds `nrp_eval` to `sys.path` at runtime. |
| **Phantom depot handling** | LKH-3's CVRP output uses phantom depots (one per vehicle, ids > N+1) at route boundaries. `routes_to_action` filters these (ids > N+1) and emits a `0` for each. |
| **`_routes_zero_indexed` for decomposition** | The decomposer expects 0-indexed customer ids (matching `td["locs"]` indices). The orchestrator strips depots and phantoms, then subtracts 2 (Uchoa → 0-indexed). |

## 5. Test Coverage

| Test file | Tests | Status |
|-----------|-------|--------|
| `tests/test_decomposition.py` | 5 (k=1, k=3, empty routes, parent-route capacity-feasibility, k=1 single-route warm start) | ✅ All pass |
| `tests/test_lkh_format.py` | 5 (writer sections, round-trip, action conversion, hand-rolled tour) | ✅ All pass |
| `tests/test_classical_lkh.py` | 3 (raw solver, BCC k=1, BCC k=2) | 2 pass, 1 documented known-fail |

The failing `test_bcc_solver_runs_k2` (n=500) is documented as a known issue: the orchestrator's parallel sub-LKH invocations can produce invalid tours for very large problems with high vehicle counts. The decomposition and format helpers themselves are tested separately and pass.

## 6. Usage

### Setup

```bash
cd /home/toothlessos/Projects/nrp/rl4co
# Install missing deps into the existing venv (parent rl4co venv already has most):
uv pip install scikit-learn vrplib

# Source the venv and the nrp_eval harness on PYTHONPATH:
source .venv/bin/activate
export PYTHONPATH="$PWD/experiments/nrp_eval:$PWD/experiments/learn_decompose_eval:$PYTHONPATH"
export LDE_LKH_BINARY="$PWD/experiments/learn_decompose_eval/LKH-3.0.14/LKH"
```

### Build the patched LKH-3 (already done; rebuild only when the patch changes)

```bash
cd experiments/learn_decompose_eval
./scripts/build_lkh.sh
```

### Run unit tests

```bash
python -m pytest experiments/learn_decompose_eval/tests/ -v
```

### Smoke test: raw LKH-3

```bash
./scripts/run_eval.sh 20
# → raw LKH-3, n=20, 1000 instances, wandb run `raw-lkh3-cvrp20-seed1234`
```

### Smoke test: LKH-3 + BCC

```bash
SOLVER=bcc_lkh_cvrp ./scripts/run_eval.sh 100
# → LKH-3 + BCC, n=100, max_total_s=60s per instance
```

### Full comparison

```bash
# Baseline
./scripts/run_eval.sh 100

# Decomposed
SOLVER=bcc_lkh_cvrp ./scripts/run_eval.sh 100
SOLVER=bcc_lkh_cvrp ./scripts/run_eval.sh 200
```

The two runs land in the same wandb project (`nrp-learn-decompose`) with different groups (`raw-lkh3-cvrp100` vs `bcc-lkh3-cvrp100`), enabling side-by-side comparison of:
- `mean_tour_length` (the headline metric — lower is better)
- `wallclock_per_instance_s` (parallelism overhead)
- `per_instance` table (gap % = `(cost_bcc - cost_raw) / cost_raw`)

### Direct CLI invocation (for ad-hoc overrides)

```bash
python -m learn_decompose_eval eval \
    experiment=learn_decompose_eval/bcc_lkh_cvrp \
    env.generator_params.num_loc=50 \
    evaluate.num_instances=100 \
    seed=1234
```

## 7. Files Inventory

```
experiments/learn_decompose_eval/
├── CLAUDE.md                           # project intent
├── README.md                           # short overview
├── IMPLEMENTATION_REPORT.md            # this file
├── pyproject.toml
├── .gitignore
├── LKH-3.0.14.tgz                      # un-patched reference (prebuilt sources inside)
├── LKH-3.0.14/                         # patched LKH-3 source
│   ├── LKH                             # compiled binary
│   └── SRC/
│       ├── INCLUDE/LKH.h               # (+1 line: extern decl)
│       ├── LKH.c                       # (+1 line: global def)
│       ├── ReadParameters.c            # (+5 lines: parse param + init)
│       └── FindTour.c                  # (+6 lines: WriteTour call)
├── cvrp-decomposition/                 # reference: alberto-santini/cvrp-decomposition
├── learn_decompose_eval/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── lkh_format.py              # 257 lines
│   │   ├── decomposition.py           # 220 lines
│   │   ├── orchestration.py           # 290 lines
│   │   └── classical_lkh.py           # 245 lines
│   └── data/__init__.py
├── configs/
│   ├── main.yaml
│   ├── hydra/default.yaml
│   ├── paths/default.yaml
│   ├── callbacks/default.yaml
│   ├── trainer/default.yaml
│   ├── extras/default.yaml
│   ├── logger/{wandb,csv}.yaml
│   ├── env/cvrp.yaml
│   └── experiment/learn_decompose_eval/
│       ├── raw_lkh_cvrp.yaml
│       └── bcc_lkh_cvrp.yaml
├── scripts/
│   ├── build_lkh.sh
│   └── run_eval.sh
└── tests/
    ├── conftest.py
    ├── test_decomposition.py
    ├── test_lkh_format.py
    └── test_classical_lkh.py
```

## 8. Known Limitations

1. **n=500 BCC orchestrator race**: With many vehicles, parallel sub-LKH invocations occasionally produce invalid stitched tours. Decomposition and format helpers are unaffected. Workaround: reduce `num_workers` or skip decomposition for very large problems until the race is debugged.
2. **CVRP capacity in EXPLICIT format**: LKH-3's CVRP solver internally expands dimension to `n + Salesmen`; phantom depots are handled correctly, but very large capacity subproblems may need additional tuning of `target_max_subproblem_size`.
3. **WandB integration not yet tested end-to-end**: Config tree is wired but the live `wandb run` flow requires a logged-in wandb account. For offline use, set `WANDB_DISABLED=1` or `logger=csv` in the experiment override.
4. **No built-in optima dataset**: The Uchoa CVRP benchmark instances in `cvrp-decomposition/data/test-set/` could be wired in for known-optimum comparison, but the current implementation uses RL4CO's on-the-fly uniform sampling.

## 9. Future Work

- **Debug n=500 BCC orchestrator race** (add a barrier after each parallel subproblem; or use `multiprocessing` instead of `ThreadPoolExecutor` for true isolation).
- **Larger problem sizes** (n=500, 1000): require careful tuning of `decompose_every_s` and `target_max_subproblem_size`; consider multi-level decomposition.
- **Uchoa optima integration**: load `cvrp-decomposition/data/validation-set/*.vrp` instances and report `gap_to_opt_pct` in wandb.
- **Larger sweep**: run a Hydra sweep over `(num_loc, target_max_subproblem_size, decompose_every_s)` to map the parameter landscape.
- **Compare against HGS-CVRP**: build the `hgs` binary from `cvrp-decomposition/` and add it as a third solver in the comparison.
