# nrp_eval

A modular, reusable experiment pipeline for evaluating **RL**, **classical**,
and **hybrid** solvers on **Neural Routing Problems**, built on top of
[RL4CO](https://github.com/ai4co/rl4co). All routing envs in RL4CO's
`ENV_REGISTRY` (TSP, CVRP, SDVRP, PCTSP, OP, MTSP, ATSP, PDP, …) are
supported from day one through a single unified interface.

The design is driven by the [`phase1.md`](./phase1.md) design doc.

---

## What you get

- **`Solver` ABC + `SolverRegistry`** — plugin-style registration. Any
  solver turns a `TensorDict` of problem instances into a `TensorDict` with
  `actions` + `reward` (compatible with `env.get_reward`).
- **`RLSolver`** — wraps any RL4CO zoo policy (POMO, AM, SymNCO, MatNet,
  PointerNetwork) via `from_checkpoint` or `from_policy`. Supports
  `greedy`, `sampling`, `multistart_greedy`, and dihedral-8 augmentation
  via `rl4co.data.transforms.StateAugmentation`.
- **`ClassicalSolver`** ABC with:
  - `ORToolsTSPSolver` — nearest-neighbor + iterative 2-opt (Pythonic,
    good enough to validate the pipeline).
  - `ORToolsVRPSolver` — Clarke-Wright savings algorithm.
  - `BuiltinEnvSolver` — adapter for `env.solve(instances, …)`.
  - `LKHSolver`, `ConcordeTSPSolver`, `GurobiTSPSolver` — **stubs** that
    raise a clear `FileNotFoundError` pointing to `NRP_LKH_BINARY`,
    `NRP_CONCORDE_BINARY`, `NRP_GUROBI_BINARY` (or
    `solver.binary_path` config). Real subprocess wrappers land in
    stage 2.
- **Unified `evaluate(solver, env, dataset, method, ...)` harness** —
  wraps any `Solver` as a policy via `_SolverAsPolicy(nn.Module)`, builds
  the right `rl4co.tasks.eval.*` class, does warmup-aware timing, writes
  per-instance results to a JSONL / pickle, and logs to W&B.
- **`train_and_evaluate(cfg)`** — thin glue that delegates training to
  `rl4co.tasks.train.train` (Hydra entry), then runs the unified eval.
- **Hydra config tree** — full `main.yaml` + `env/` + `model/` +
  `experiment/` + `logger/` + `trainer/` configs, mirroring RL4CO's
  convention. Every config key is overridable from the CLI.
- **`python -m nrp {train, eval, sweep}`** CLI.
- **`wandb` integration** for both training and eval-only paths.
- **35 unit tests + 1 smoke test** — all CPU-only, all passing.

### Deferred to stage 2

- `HybridSolver` ABC + `L2DSolver` (RL4CO's L2D is for FJSP/JSSP
  scheduling, not routing).
- Real LKH-3 / Concorde / Gurobi subprocess wrappers.
- Real TSPLIB / CVRPLIB benchmark loaders.
- HPO (Optuna / Ray Tune).

---

## Install

```bash
cd /home/toothlessos/Projects/nrp/rl4co
uv sync --all-extras                   # the main repo
uv pip install -e experiments/nrp_eval/
```

---

## Quick start

### 1. Evaluate a solver on TSP-20 (no training needed)

```bash
cd /home/toothlessos/Projects/nrp/rl4co/experiments/nrp_eval

# OR-Tools (no checkpoint needed)
python -m nrp eval experiment=nrp_eval/ortools_tsp \
    env.generator_params.num_loc=20 \
    evaluate.num_instances=20 \
    trainer.max_epochs=0 \
    logger=csv
```

Writes `results/ortools_tsp-tsp-20.pkl` and prints a summary like:

```
num_instances: 20
mean tour length: 4.0
min: 3.1, max: 4.5
feasible_ratio: 1.0
method: greedy
solver_name: ortools_tsp
```

### 2. Train POMO-TSP-20, then evaluate the best checkpoint

```bash
python -m nrp train_and_eval experiment=nrp_eval/pomo_tsp \
    trainer.max_epochs=5 \
    model.batch_size=16 \
    model.train_data_size=200 \
    model.val_data_size=64 \
    model.test_data_size=64 \
    env.generator_params.num_loc=20 \
    evaluate.num_instances=50 \
    evaluate.method=augment_dihedral_8 \
    logger=csv
```

The `train_and_eval` mode runs `rl4co.tasks.train.train` first, then
auto-finds the best checkpoint under `paths.output_dir` and runs
`evaluate(...)` on it.

### 3. Cartesion-product sweep

```bash
python -m nrp sweep \
    --solvers pomo,ortools_tsp,am \
    --envs tsp \
    --num_locs 50,100 \
    --seeds 42,1337
```

This runs every (solver, num_loc, seed) combo in sequence and writes
one pickle per combo under `results/`.

### 4. Programmatic use

```python
import torch
from rl4co.envs import TSPEnv
from rl4co.models import POMO
from nrp.solvers import RLSolver
from nrp.harness.evaluate import evaluate

env = TSPEnv(generator_params={"num_loc": 50})
model = POMO(env, num_augment=0)
solver = RLSolver.from_policy(env, model.policy, decode_type="greedy")
td = env.reset(batch_size=[100])
result = evaluate(solver, env, td, method="greedy")
print(result.summary["mean"], result.summary["feasible_ratio"])
```

---

## CLI reference

```
python -m nrp {train,eval,sweep} [overrides...]
```

| Command  | What it does                                                            |
|----------|-------------------------------------------------------------------------|
| `train`  | Train a model. Delegates to `rl4co.tasks.train.train` with Hydra.       |
| `eval`   | Build a solver from `cfg.solver_name` and evaluate on the test set.     |
| `sweep`  | Cartesian product over (solver, env, num_loc, seed) and run each combo. |

Common overrides:

```bash
experiment=nrp_eval/<name>     # which experiment YAML to compose
env.generator_params.num_loc=N # problem size
evaluate.method=greedy|...     # greedy | sampling | multistart_greedy |
                               # augment | augment_dihedral_8 |
                               # multistart_greedy_augment[_dihedral_8]
evaluate.num_instances=N       # how many instances to evaluate on
solver_name=pomo|am|...        # which solver to use
logger=wandb|csv|null          # which logger to use
trainer.max_epochs=N           # for the training path
```

---

## Eval methods

| `evaluate.method`              | Decoder                             | When to use |
|--------------------------------|-------------------------------------|-------------|
| `greedy`                       | Greedy, batch-serial                | fast, deterministic |
| `sampling`                     | Sampling, `num_starts` rollouts      | RL policies only |
| `multistart_greedy`            | Greedy, `num_starts` rollouts        | RL policies only |
| `augment`                      | Symmetric augmentation              | RL policies only |
| `augment_dihedral_8`           | 8-way dihedral augmentation         | RL policies only |
| `multistart_greedy_augment`    | multistart + symmetric aug          | RL policies only |
| `multistart_greedy_augment_dihedral_8` | multistart + dihedral-8    | POMO-style eval (default) |

The harness uses `rl4co.tasks.eval.{Greedy, Sampling, Augmentation,
GreedyMultiStart, GreedyMultiStartAugment}Eval` under the hood, dispatched
on the method string. Classical solvers always run greedy (the other
methods are RL-only).

---

## Output

For each evaluation run, the harness produces:

1. **`results/<solver>-<env>-<num_loc>.pkl`** — a `versioned` pickle (schema
   `evaluation_result`) with:
   - `per_instance`: list of dicts with `instance_idx`, `tour_length`,
     `reward`, `feasible`, optional `gap_to_opt_pct`.
   - `summary`: `mean / std / min / max / p50 / p95 / feasible_ratio`,
     plus `wallclock_total_s`, `wallclock_per_instance_s`, `method`,
     `solver_name`, `env_name`, `num_loc`.
   - `metadata`: solver type, batch size, num instances, elapsed time.
2. **W&B** (if `wandb_run` is given): per-instance `wandb.Table` and
   summary scalars, with config = full Hydra config.
3. **JSONL** (optional, via `nrp.utils.metrics.PerInstanceWriter`) —
   streaming per-instance rows to a `.jsonl` file.

Load a saved result back with:

```python
from nrp.utils.pkl import load_versioned
result = load_versioned("results/ortools_tsp-tsp-20.pkl",
                       expected_schema="evaluation_result")
print(result.summary)
```

---

## Architecture

```
experiments/nrp_eval/
├── nrp/                          # the package
│   ├── __main__.py               # `python -m nrp ...`
│   ├── cli.py                    # argparse: train | eval | sweep
│   ├── solvers/
│   │   ├── base.py               # Solver ABC + SolverRegistry  (the keystone)
│   │   ├── rl.py                 # RLSolver (wraps any RL4CO zoo policy)
│   │   └── classical.py          # ORTools / Clarke-Wright / Builtin + LKH/Concorde/Gurobi stubs
│   ├── envs/
│   │   ├── env_registry.py       # ENV_INFO for all routing envs
│   │   ├── factory.py            # build_env(name, generator_params, ...)
│   │   └── dataset_factory.py    # build_dataset_from_spec
│   ├── data/
│   │   ├── dataset.py            # DatasetSpec + build_eval_dataset
│   │   ├── generation.py         # re-exports of rl4co.data.generate_data
│   │   ├── benchmarks.py         # TSPLIB/CVRPLIB loader stubs
│   │   └── splits.py             # SplitSpec, split_filename
│   ├── harness/
│   │   ├── evaluate.py           # unified evaluate() with _SolverAsPolicy adapter
│   │   ├── train.py              # train_and_evaluate(cfg)
│   │   ├── runner.py             # cartesian-product sweep
│   │   └── timing.py             # warmup-aware Timer
│   └── utils/
│       ├── metrics.py            # gap_to_optimal, tour_length_summary, PerInstanceWriter
│       ├── reproducibility.py    # seed_everything
│       ├── device.py             # resolve_device
│       ├── pkl.py                # versioned pickle save/load
│       └── logging.py            # init_wandb helper
├── configs/                      # Hydra config tree
│   ├── main.yaml
│   ├── env/{tsp,cvrp,pctsp,op,mtsp,sdvrp,spctsp}.yaml
│   ├── model/{pomo,am,symnco,matnet}.yaml
│   ├── experiment/nrp_eval/<pomo_tsp, pomo_cvrp, am_tsp, symnco_tsp,
│   │                       ortools_tsp, ortools_cvrp, builtin_tsp>.yaml
│   ├── logger/{wandb,csv}.yaml
│   ├── trainer/default.yaml
│   ├── callbacks/default.yaml
│   ├── hydra/default.yaml
│   ├── paths/default.yaml
│   └── extras/default.yaml
├── scripts/
│   ├── eval.py / train.py / sweep.py
│   └── install_classical_solvers.sh   # docs only; user installs binaries
├── tests/                        # 35 unit tests + 1 smoke test
├── results/                      # gitignored
├── checkpoints/                  # gitignored
├── data/                         # gitignored
├── pyproject.toml
└── README.md                     # this file
```

### The `Solver` keystone

The whole framework hinges on one tiny contract:

```python
class Solver(ABC):
    name: str = "abstract"
    is_trainable: bool = False

    def __init__(self, env, device="cpu", **kwargs): ...
    @abstractmethod
    def solve(self, td: TensorDict) -> TensorDict:
        """Return TensorDict with `actions` (int64 [B, L]) and `reward` ([B])."""
        ...

    def warmup(self, td): self.solve(td[:1].clone())  # best-effort
    def to(self, device): self.device = torch.device(device); return self
```

The `_SolverAsPolicy(nn.Module)` adapter in `nrp/harness/evaluate.py` wraps
any `Solver` so it can be passed to RL4CO's existing
`GreedyEval`/`AugmentationEval`/etc. classes — that's how classical and
hybrid solvers reuse RL4CO's eval machinery unchanged.

---

## Tests

```bash
# All tests (CPU only)
cd /home/toothlessos/Projects/nrp/rl4co/experiments/nrp_eval
python -m pytest tests/ -v

# End-to-end smoke test (tiny train + eval, ~1-2 min)
NRP_RUN_SMOKE=1 python -m pytest tests/smoke_test.py -v -s
```

Coverage:

- `test_registry.py` — register + lookup, unknown name raises.
- `test_reproducibility.py` — `seed_everything(42)` is idempotent.
- `test_dataset_factory.py` — synthetic + TSPLIB-stub fixture.
- `test_hydra_config.py` — `hydra.compose("main")` resolves all defaults.
- `test_rl_solver.py` — POMO/AM via `from_policy` and `from_checkpoint`.
- `test_classical_solver.py` — OR-Tools returns valid tours; classical
  is within 2× of untrained POMO greedy.
- `test_metrics.py` — `gap_to_optimal`, summary, JSONL round-trip.
- `test_evaluate.py` — `evaluate(...)` returns the right shape,
  honours optima, saves pickles, dispatches on method.
- `smoke_test.py` — train POMO-TSP-20 for 1 epoch + eval 50 instances.

```bash
# Lint
ruff check nrp/ tests/ configs/
```

---

## Classical solver installation

Stage 1 does **not** auto-install LKH-3, Concorde, or Gurobi. The
`scripts/install_classical_solvers.sh` script contains documentation for
each. The general pattern is:

```bash
# OR-Tools (Python wheels — recommended)
pip install ortools

# LKH-3: build from source, then point the pipeline at the binary
export NRP_LKH_BINARY=/usr/local/bin/lkh

# Concorde (academic use)
export NRP_CONCORDE_BINARY=/usr/local/bin/concorde

# Gurobi (free academic license)
export NRP_GUROBI_BINARY=/opt/gurobi*/bin/gurobi_cl
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

Or per-experiment: `solver.lkh_tsp.binary_path: /path/to/lkh`.

Until stage 2, the LKH/Concorde/Gurobi solver classes raise a clear
`FileNotFoundError` with the env-var hint.

---

## RL4CO utilities reused (not reinvented)

| Need                | Reuse |
|---------------------|-------|
| Env construction    | `rl4co.envs.get_env(name)` |
| Model loading       | `ZOO[name].load_from_checkpoint(path, load_baseline=False)` |
| Policy call         | `policy(td, env=env, phase="test", decode_type=..., num_starts=...)` |
| Eval dispatch       | `rl4co.tasks.eval.{Greedy, Sampling, Augmentation, ...}Eval` |
| Reward              | `env.get_reward(td, actions)` |
| Auto batch size     | `rl4co.tasks.eval.get_automatic_batch_size(eval_fn)` |
| Augmentation        | `rl4co.data.transforms.StateAugmentation` |
| Dataset             | `rl4co.data.dataset.TensorDictDataset` |
| Data generation     | `rl4co.data.generate_data.{generate_env_data, generate_dataset}` |
| Training entry      | `rl4co.tasks.train.train(cfg)` |
| Reproducibility     | `lightning.fabric.utilities.seed.seed_everything` |

---

## References

- Design doc: [`phase1.md`](./phase1.md)
- RL4CO upstream: <https://github.com/ai4co/rl4co>
- Reference: `experiments/POMO_TSP_baseline/` is the **old** hand-written
  pattern; this package is the replacement.
