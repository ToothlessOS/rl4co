# NRP Evaluation Pipeline — Stage 1 Plan

## Context

This repo is a research fork of RL4CO at `/home/toothlessos/Projects/nrp/rl4co/`. The user wants a **modular, reusable experiment pipeline** for evaluating different solvers (RL, classical, hybrid) for **Neural Routing Problems** — with all of RL4CO's routing problems (TSP, CVRP, VRP variants) supported from day one. The existing `experiments/POMO_TSP_baseline/` is a hand-written one-off and **not** the pattern to follow.

Goals for this stage (the "first stage" — design + skeleton):
- A single, unified `Solver` abstraction that works for RL4CO zoo policies, classical solvers (OR-Tools, LKH, Concorde, Gurobi), and hybrid methods (L2D, NeuroLKH).
- A single evaluation harness that produces comparable metrics across all solver types on identical, reproducible datasets.
- Hydra-driven configuration, consistent with RL4CO's library convention.
- W&B logging (per the modified `CLAUDE.md`).
- Training path delegates to RL4CO (`rl4co.tasks.train.train`); no reinvented training loop.

**The pipeline is at `experiments/nrp_eval/`** as a self-contained framework folder.

---

## High-level design

### The `Solver` abstraction (the keystone)

A `Solver` is anything that turns a `TensorDict` of problem instances into a `TensorDict` with an `actions` key accepted by `env.get_reward(td, actions)`. Implemented as an **ABC + registry** (`@SolverRegistry.register("name", env_names=(...))`).

Three concrete subclasses:
- `RLSolver` — wraps an `rl4co.models.zoo.<X>.policy.<X>Policy` (loaded via `<X>.load_from_checkpoint(path, load_baseline=False, strict=False, weights_only=False)`). `solve(td)` calls `self.policy(td, env, phase="test", decode_type=...)` then `env.get_reward`. Supports `decode_type ∈ {greedy, sampling, multistart_greedy, multistart_sampling}` and dihedral-8 augmentation via `rl4co.data.transforms.StateAugmentation`.
- `ClassicalSolver` (ABC) — concrete `ORToolsTSPSolver`, `ORToolsVRPSolver`, `LKHSolver`, `ConcordeTSPSolver`, `GurobiTSPSolver`, `BuiltinEnvSolver` (adapts `env.solve(instances, max_runtime, num_procs)`). Round-trips `td` through CPU; calls external solver per instance (or via `multiprocessing.Pool`); returns a `[B, max_len]` int64 tensor of actions.
- `HybridSolver` (ABC) — `L2DSolver` reference impl loads `rl4co.models.zoo.l2d.L2DModel`. Uses `ImprovementEnvBase.step_to_solution` to seed an initial solution, then loops: `td = self.policy(td); td = self.env.step(td)["next"]` until done.

### The Evaluation Harness

`nrp/harness/evaluate.evaluate(solver, env, dataset, method, ...)` does:
1. Wrap `ClassicalSolver` / `HybridSolver` in a `_SolverAsPolicy(torch.nn.Module)` adapter so they plug into the *unmodified* `rl4co.tasks.eval.{GreedyEval, AugmentationEval, GreedyMultiStartAugmentEval}` machinery.
2. Auto-batch via `rl4co.tasks.eval.get_automatic_batch_size`.
3. Warmup one batch (excluded from timing), then time the real loop with `time.perf_counter()` + `cuda.synchronize()`.
4. Per-instance: `reward`, `tour_length = -reward`, `wallclock_s`, `gap_to_opt_pct` (only if dataset provides optima, e.g. TSPLIB).
5. Summary: `mean ± std`, `min`, `max`, `p50`, `p95`, `feasible_ratio`.
6. W&B: `wandb.Table` for per-instance, scalars for summary, `wandb.config = OmegaConf.to_container(cfg)`. Group by `(solver, env, num_loc)`.
7. Pickle the full `EvaluationResult` to `results/<run_id>.pkl`.

### The Training Harness (thin glue)

`nrp/harness/train.train_and_evaluate(cfg)` is a 80-line wrapper that:
- If `cfg.mode == "train"`: calls `rl4co.tasks.train.train(cfg)` (the existing Hydra entry) and captures the best checkpoint path.
- If `cfg.mode == "eval"`: instantiates the `Solver` from `cfg.ckpt_path` + `cfg.solver_name`, calls `nrp.harness.evaluate.evaluate(...)`.

The training path **is** RL4CO's training code; we just compose a Hydra config and run it.

---

## Directory layout

```
experiments/nrp_eval/
├── README.md
├── pyproject.toml                # local package; depends on rl4co
├── nrp/
│   ├── __init__.py
│   ├── __main__.py               # `python -m nrp ...`
│   ├── cli.py                    # argparse: train | eval | sweep
│   ├── solvers/
│   │   ├── base.py               # Solver ABC, SolverRegistry
│   │   ├── rl.py                 # RLSolver (wraps RL4CO policies)
│   │   ├── classical.py          # ClassicalSolver + ORTools/LKH/Concorde/Gurobi
│   │   ├── hybrid.py             # HybridSolver + L2DSolver
│   │   └── builtin_solve.py      # adapter for env.solve hook
│   ├── envs/
│   │   ├── factory.py            # build_env(name, generator_params, dataset_params)
│   │   ├── env_registry.py       # env_name -> (env_cls, supports_improvement)
│   │   └── dataset_factory.py    # build_dataset(env, spec)
│   ├── data/
│   │   ├── dataset.py            # DatasetSpec, build_eval_dataset
│   │   ├── benchmarks.py         # TSPLIB/CVRPLIB loaders (vrplib)
│   │   ├── generation.py         # thin wrapper over rl4co.data.generate_data
│   │   └── splits.py             # train/val/test split, optional opt_tour
│   ├── harness/
│   │   ├── evaluate.py           # the unified evaluate() function
│   │   ├── train.py              # train_and_evaluate() — thin glue
│   │   ├── runner.py             # loop over (solver, env, size) tuples
│   │   └── timing.py             # Timer, warmup-aware context manager
│   └── utils/
│       ├── metrics.py            # gap_to_optimal, tour_length_summary, JSONL writer
│       ├── logging.py            # W&B init helper, structured logger
│       ├── reproducibility.py    # seed_everything (python/numpy/torch/cuda/cudnn)
│       ├── device.py             # resolve_device, to_device
│       └── pkl.py                # versioned pickle save/load
├── configs/                      # self-contained Hydra tree
│   ├── main.yaml
│   ├── env/                      # tsp.yaml, cvrp.yaml, ... (one per env)
│   ├── model/                    # pomo.yaml, am.yaml, symnco.yaml, matnet.yaml
│   ├── solver/                   # rl.yaml, classical.yaml, hybrid.yaml
│   ├── experiment/nrp_eval/      # pomo_tsp.yaml, ortools_tsp.yaml, lkh_tsp.yaml, l2d_tsp.yaml, ...
│   ├── logger/wandb.yaml
│   └── trainer/default.yaml
├── scripts/
│   ├── eval.py                   # thin wrapper
│   ├── train.py
│   ├── sweep.py                  # cartesian product runner
│   └── install_classical_solvers.sh  # docs only (no auto-install)
├── tests/                        # 8 unit tests + 1 smoke test
├── results/                      # gitignored
├── checkpoints/                  # gitignored
└── data/                         # gitignored
```

---

## Key design decisions

1. **ABC + registry for `Solver`**, not a Protocol. Need name-based lookup from a config string.
2. **Warmup-aware wall clock** for timing (`time.perf_counter()` + `cuda.synchronize()`). Per-instance CPU user time captured via `resource.getrusage` for classical solvers.
3. **Classical solvers round-trip to CPU** at the boundary (`td.to("cpu")`, `locs.numpy()`, actions back to original device).
4. **Three primary metrics** (always reported together): `tour_length` (mean/std/percentiles), `gap_to_opt_pct` (when optima exist; otherwise note "optimal: unknown"), `wallclock_per_instance_s`.
5. **Classical binaries are user-provided.** Config keys `solver.binary_path`, `solver.max_runtime_s`; env-var fallback `NRP_LKH_BINARY`. `install_classical_solvers.sh` is *documentation*, not auto-install.
6. **Multi-file eval** uses RL4CO's `env.dataset(filename=[...])` dict-of-datasets convention; the harness iterates.
7. **W&B** — one run per `(solver, env, num_loc, method, seed)`. `wandb.config` is the full Hydra config; per-instance data goes to a `wandb.Table`; summary scalars to `wandb.summary`.
8. **Reproducibility** — `seed_everything(seed, deterministic=False)` is the single entry point. Eval datasets regenerate from `(seed, num_loc, num_instances, loc_distribution)` and are cached as `.npz`.
9. **Training is delegated to `rl4co.tasks.train.train(cfg)`** — no new training code. The training harness just composes a Hydra config and runs it.
10. **Augment-aware methods for non-RL solvers** — methods like `augment_dihedral_8` and `multistart_greedy` only apply to `RLSolver`; for `ClassicalSolver`/`HybridSolver` we fall back to `greedy` with a warning.

---

## RL4CO utilities to reuse (do not reinvent)

- **Env/Generator**: `rl4co.envs.{TSPEnv, CVRPEnv, ...}`, `rl4co.envs.common.utils.Generator`, `rl4co.envs.ENV_REGISTRY` (`rl4co/envs/__init__.py`), `rl4co.envs.get_env(name)`.
- **Model loaders**: `rl4co.models.zoo.{POMO, AttentionModel, SymNCO, PointerNetwork, MATNet}.load_from_checkpoint(path, load_baseline=False, strict=False, weights_only=False)`.
- **Decoding strategies**: `rl4co.utils.decoding.get_decoding_strategy` (registry: `greedy`, `sampling`, `multistart_greedy`, `multistart_sampling`, `beam_search`).
- **Augmentation**: `rl4co.data.transforms.{StateAugmentation, dihedral_8_augmentation, symmetric_augmentation}`.
- **TensorDict ops**: `rl4co.utils.ops.{batchify, unbatchify, gather_by_index}`.
- **Eval base classes**: `rl4co.tasks.eval.{EvalBase, GreedyEval, SamplingEval, AugmentationEval, GreedyMultiStartEval, GreedyMultiStartAugmentEval, get_automatic_batch_size, evaluate_policy}`.
- **Training entry**: `rl4co.tasks.train.train(cfg)` (Hydra `@hydra.main` wrapper).
- **Data generation**: `rl4co.data.generate_data.generate_dataset`, `generate_env_data`.
- **Classical hook**: `RL4COEnvBase.solve(instances, max_runtime, num_procs)` (built into every env).
- **Improvement envs**: `rl4co.envs.common.base.ImprovementEnvBase` (`step_to_solution`).
- **L2D**: `rl4co.models.zoo.l2d.{L2DModel, L2DPolicy, L2DAttnPolicy, L2DPPOModel, L2DPolicy4PPO}` (already exist in RL4CO).
- **W&B logger**: `lightning.pytorch.loggers.WandbLogger` for the training path.
- **Reproducibility**: `lightning.fabric.utilities.seed.seed_everything` (or hand-rolled equivalent).
- **VRPLIB**: `vrplib` package for TSPLIB/CVRPLIB parsing.
- **Hydra root**: `configs/main.yaml` patterns from `/home/toothlessos/Projects/nrp/rl4co/configs/` (env/, model/, experiment/<domain>/<method>.yaml, logger/wandb.yaml).

---

## Critical files to create

- `experiments/nrp_eval/nrp/solvers/base.py` — `Solver` ABC + `SolverRegistry` (decorator-based registration, name lookup, `available(env_name=...)` filter).
- `experiments/nrp_eval/nrp/solvers/rl.py` — `RLSolver` with `from_checkpoint(env, ckpt_path, model_name)` and `from_policy(env, policy)` classmethods. Handles `decode_type`, `num_starts`, `num_augment` (via `StateAugmentation`).
- `experiments/nrp_eval/nrp/solvers/classical.py` — `ClassicalSolver` ABC + `ORToolsTSPSolver`, `ORToolsVRPSolver`, `LKHSolver`, `ConcordeTSPSolver`, `GurobiTSPSolver`, `BuiltinEnvSolver` (last one calls `env.solve(...)`).
- `experiments/nrp_eval/nrp/solvers/hybrid.py` — `HybridSolver` ABC + `L2DSolver` (loads `L2DModel`, uses `ImprovementEnvBase.step_to_solution`).
- `experiments/nrp_eval/nrp/envs/factory.py` — `build_env(name, generator_params, dataset_params)`. Uses `rl4co.envs.get_env(name)`.
- `experiments/nrp_eval/nrp/data/dataset.py` — `DatasetSpec` dataclass + `build_eval_dataset(env, spec)` returning `TensorDictDataset | dict[name, TensorDictDataset]`. Backed by synthetic (`.npz`), TSPLIB, CVRPLIB.
- `experiments/nrp_eval/nrp/harness/evaluate.py` — `evaluate(solver, env, dataset, method, ...)`. Contains `_SolverAsPolicy(torch.nn.Module)` adapter that lets `ClassicalSolver`/`HybridSolver` plug into `rl4co.tasks.eval.*` EvalBase subclasses.
- `experiments/nrp_eval/nrp/harness/train.py` — `train_and_evaluate(cfg)`. Dispatches on `cfg.mode` to `rl4co.tasks.train.train(cfg)` or `nrp.harness.evaluate.evaluate(...)`.
- `experiments/nrp_eval/nrp/utils/metrics.py` — `gap_to_optimal`, `tour_length_summary(per_instance)`, `PerInstanceWriter` (JSONL).
- `experiments/nrp_eval/nrp/utils/reproducibility.py` — `seed_everything(seed, deterministic=False)`.
- `experiments/nrp_eval/nrp/utils/logging.py` — W&B init helper, structured logger.
- `experiments/nrp_eval/nrp/utils/pkl.py` — versioned pickle save/load.
- `experiments/nrp_eval/configs/main.yaml` — top-level Hydra entry, defaulting to `experiment: nrp_eval/pomo_tsp` and `logger: wandb`.
- `experiments/nrp_eval/configs/env/{tsp,cvrp,...}.yaml` — one per routing env.
- `experiments/nrp_eval/configs/model/{pomo,am,symnco,matnet}.yaml` — one per RL4CO zoo model.
- `experiments/nrp_eval/configs/experiment/nrp_eval/{pomo_tsp, pomo_cvrp, am_tsp, ortools_tsp, lkh_tsp, l2d_tsp, symnco_tsp}.yaml` — one per (solver, problem) pair.
- `experiments/nrp_eval/tests/test_*.py` — 8 unit tests + 1 smoke test.

---

## Skeleton of the most critical pieces

### `nrp/solvers/base.py` (the keystone)

```python
class Solver(ABC):
    name: str = "abstract"
    is_trainable: bool = False
    is_differentiable: bool = False
    def __init__(self, env, device="cpu", **kwargs): ...
    @abstractmethod
    def solve(self, td: TensorDict) -> TensorDict: ...
    def warmup(self, td): self.solve(td[:1].clone())  # best-effort
    def to(self, device): self.device = torch.device(device); return self

class SolverRegistry:
    registry: dict[str, type[Solver]] = {}
    @classmethod
    def register(cls, name, env_names=()):
        def deco(klass):
            cls.registry[name] = klass
            klass._registered_name = name
            klass._supported_envs = env_names
            return klass
        return deco
    @classmethod
    def build(cls, name, env, **kwargs): ...
    @classmethod
    def available(cls, env_name=None) -> list[str]: ...
```

### `nrp/solvers/rl.py` (the RL adapter)

```python
@SolverRegistry.register("pomo", env_names=("tsp","cvrp","sdvrp","pctsp","spctsp","mtsp","op"))
@SolverRegistry.register("am", env_names=("tsp","cvrp",...))
class RLSolver(Solver):
    is_trainable = True

    @classmethod
    def from_checkpoint(cls, env, ckpt_path, model_name="POMO", **kwargs):
        model = ZOO[model_name.lower()].load_from_checkpoint(
            ckpt_path, load_baseline=False, strict=False, weights_only=False)
        return cls(env, model=model, model_name=model_name, ckpt_path=ckpt_path, **kwargs)

    def solve(self, td):
        td = td.to(self.device).clone()
        if self.augmentation is not None:
            td = self.augmentation(td)
        out = self.policy(td, env=self.env, phase="test", decode_type=self.decode_type,
                          num_starts=self.num_starts, ...)
        actions = out["actions"]
        reward = self.env.get_reward(td, actions)
        # select_best aggregation for multistart/augment
        return TensorDict(actions=actions, reward=reward, batch_size=actions.shape[:1])
```

### `nrp/solvers/classical.py` (CPU round-trip)

```python
class ClassicalSolver(Solver, ABC):
    is_trainable = False
    @abstractmethod
    def solve_batch(self, td_cpu) -> np.ndarray: ...
    def solve(self, td):
        device = td.device
        actions_np = self.solve_batch(td.to("cpu").clone())  # [B, L]
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
        reward = self.env.get_reward(td.to(device), actions)
        return TensorDict(actions=actions, reward=reward, batch_size=actions.shape[:1])

@SolverRegistry.register("ortools_tsp", env_names=("tsp",))
class ORToolsTSPSolver(ClassicalSolver): ...
@SolverRegistry.register("lkh_tsp", env_names=("tsp",))
class LKHSolver(ClassicalSolver): ...  # uses subprocess + vrplib I/O
@SolverRegistry.register("concorde_tsp", env_names=("tsp",))
class ConcordeTSPSolver(ClassicalSolver): ...
@SolverRegistry.register("gurobi_tsp", env_names=("tsp",))
class GurobiTSPSolver(ClassicalSolver): ...
@SolverRegistry.register("builtin_solve", env_names=tuple())
class BuiltinEnvSolver(ClassicalSolver):
    """Adapts RL4COEnvBase.solve(instances, max_runtime, num_procs)."""
    def solve_batch(self, td_cpu):
        # call env.solve and reshape (actions, costs) -> [B, L] int64
        ...
```

### `nrp/harness/evaluate.py` (the unified harness — uses _SolverAsPolicy adapter)

```python
class _SolverAsPolicy(torch.nn.Module):
    """Lets ClassicalSolver / HybridSolver plug into rl4co.tasks.eval EvalBase."""
    def __init__(self, solver, env, method):
        super().__init__()
        self.solver, self.env, self.method = solver, env, method
        self.dummy = torch.nn.Parameter(torch.zeros(1))
    def forward(self, td, decode_type=None, num_starts=0, **kwargs):
        out = self.solver.solve(td)
        return {"actions": out["actions"], "reward": out.get("reward")}

def evaluate(solver, env, dataset, method="augment_dihedral_8", batch_size=None,
             samples=1280, num_augment=8, optima=None, wandb_run=None,
             save_dir=None, run_id=None):
    # 1. resolve method -> rl4co EvalBase subclass
    # 2. wrap solver in _SolverAsPolicy
    # 3. warmup one batch (excluded from timing)
    # 4. loop DataLoader, time each batch, build per_instance rows
    # 5. summarize, log to W&B, pickle to results/<run_id>.pkl
    ...
```

### `configs/main.yaml` + `experiment/nrp_eval/pomo_tsp.yaml`

```yaml
# configs/main.yaml
defaults:
  - _self_
  - callbacks: default
  - hydra: default
  - logger: wandb
  - trainer: default
  - paths: default
  - extras: default
  - model: pomo
  - env: tsp
  - experiment: nrp_eval/pomo_tsp
mode: ${oc.env:NRP_MODE,train}
solver_name: pomo
seed: 42
paths:
  data_dir: ${oc.env:NRP_DATA_DIR,data}
  output_dir: ${oc.env:NRP_OUTPUT_DIR,results}
```

```yaml
# configs/experiment/nrp_eval/pomo_tsp.yaml
env:
  generator_params: { num_loc: 100, loc_distribution: uniform }
  val_file:  tsp${env.generator_params.num_loc}_val_seed4321.npz
  test_file: tsp${env.generator_params.num_loc}_test_seed1234.npz
model: { batch_size: 64, train_data_size: 100_000, val_data_size: 1_000,
         test_data_size: 1_000, optimizer_kwargs: { lr: 1e-4 } }
trainer: { max_epochs: 100, accelerator: gpu, devices: 1 }
logger:
  wandb: { project: rl4co, tags: ["pomo","tsp","nrp_eval"],
           group: "pomo-tsp100" }
solver: { rl: { decode_type: augment_dihedral_8, num_augment: 8 } }
evaluate: { method: augment_dihedral_8, num_instances: 10000 }
```

---

## Verification

Stage 1 is "done" when:

1. **`pytest experiments/nrp_eval/tests/`** passes (CPU-only):
   - `test_registry.py` — register + lookup, unknown name raises.
   - `test_rl_solver.py` — fresh `POMO(env).policy` returns valid actions on a random TSP-20 batch.
   - `test_classical_solver.py` — `ORToolsTSPSolver` returns valid tours; mean length within 2× of an untrained POMO greedy (sanity).
   - `test_hybrid_solver.py` — `L2DSolver` (untrained) runs ≥5 iters, cost non-increasing.
   - `test_evaluate.py` — `evaluate(solver, env, dataset, method="greedy")` returns `{per_instance, summary, metadata}` with finite values.
   - `test_metrics.py` — `gap_to_optimal([10,20,30], [10,10,10]) == [0,100,200]`; JSONL writer round-trips.
   - `test_reproducibility.py` — `seed_everything(42)` is idempotent.
   - `test_dataset_factory.py` — synthetic and TSPLIB (5-city fixture) both produce correct-shape TensorDicts.
   - `test_hydra_config.py` — `hydra.compose("main")` resolves all defaults; `cfg.env._target_ == "rl4co.envs.TSPEnv"`.

2. **End-to-end smoke test** (`tests/smoke_test.py`): trains POMO-TSP-20 for 1 epoch, then evaluates 50 instances. Must complete in < 5 min on GPU. Asserts `summary["num_instances"] == 50`, `summary["mean"]` finite, `summary["feasible_ratio"] ≥ 0.95`.

3. **Manual checklist**:
   - `python -m nrp train experiment=nrp_eval/pomo_tsp` (or with `trainer.max_epochs=3` for a quick check) — W&B run created, checkpoint saved.
   - `python -m nrp eval experiment=nrp_eval/pomo_tsp ckpt_path=...` — produces `results/<run_id>.pkl` + W&B table.
   - `python -m nrp eval experiment=nrp_eval/ortools_tsp env.generator_params.num_loc=100` — OR-Tools on the same 10k instances; mean tour length within 5–10% of POMO.
   - `python -m nrp sweep --config experiments/nrp_eval/configs/sweeps/rl_vs_classical.yaml` — cartesian product over (POMO, AM, OR-Tools) × (TSP-50, TSP-100) × {greedy, augment_dihedral_8}.

---

## Out of scope (stage 1)

- Production-scale training (100+ epochs, 10k val) — config-driven, runs through `rl4co.tasks.train.train`.
- Auto-installation of LKH-3 / Gurobi / Concorde — user provides binaries; `install_classical_solvers.sh` is docs only.
- Hybrid solvers beyond `L2DSolver` (NeuroLKH, BQ-NCO, GLOP) — the `HybridSolver` base + harness already supports them; concrete subclasses are added by the user.
- Full TSPLIB / CVRPLIB sweep configs — user adds their own sweep YAMLs.
- Differentiable solvers (`is_differentiable=True`) — registry flag reserved, no impl yet.
- Multi-GPU / DDP eval — single-process only in stage 1.
- HPO (Optuna, Ray Tune) — stage 1.5.
- Visualization — `env.render(...)` exists, no wrapper yet.
- Loggers other than W&B / TensorBoard — RL4CO already supports them via its config tree; we re-export, smoke-test only W&B.
- New env implementations — use `rl4co.envs.ENV_REGISTRY` only; custom envs are added to RL4CO first.

The architecture supports all of the above without redesign — they're additive subclasses / config files, not core changes.
