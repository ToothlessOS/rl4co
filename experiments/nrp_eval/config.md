# Configuring experiments in `nrp_eval`

The pipeline is fully **Hydra-driven**: every config key can be overridden
from the CLI, layered into named experiment YAMLs, or changed in the shared
defaults. This document walks through the three approaches, in order of how
often you should use them.

## 1. CLI overrides (recommended — don't edit files)

Hydra composes the config tree at runtime, so any key can be overridden
from the CLI without touching YAML:

```bash
# Single key
python -m nrp eval experiment=nrp_eval/pomo_tsp env.generator_params.num_loc=200

# Nested keys with dot notation
python -m nrp eval experiment=nrp_eval/pomo_tsp \
    model.optimizer_kwargs.lr=3e-4 \
    trainer.max_epochs=50 \
    evaluate.method=augment_dihedral_8

# Override lists
python -m nrp eval experiment=nrp_eval/pomo_tsp model.metrics.train=[loss,reward]

# Append a new key (use +)
python -m nrp eval experiment=nrp_eval/pomo_tsp +my_custom_flag=true

# Remove a key (use ~)
python -m nrp eval experiment=nrp_eval/pomo_tsp ~trainer.callbacks
```

**Pro:** zero file changes, fully reproducible. Hydra saves the full
composed config to `outputs/<date>/.hydra/config.yaml` for every run.

## 2. Add a new experiment YAML (when you want to commit a recipe)

If you find yourself always passing the same 5 overrides, save them as a
new file under `configs/experiment/nrp_eval/`:

```bash
# New file: configs/experiment/nrp_eval/pomo_tsp_200_lr3e4.yaml
```

```yaml
# @package _global_
defaults:
  - override /model: pomo
  - override /env: tsp
  - override /trainer: default
  - override /callbacks: default
  - override /logger: wandb

env:
  generator_params:
    num_loc: 200

model:
  batch_size: 32
  optimizer_kwargs:
    lr: 3e-4

trainer:
  max_epochs: 50

seed: 1234
solver_name: pomo
solver:
  pomo:
    decode_type: augment_dihedral_8
    num_augment: 8

evaluate:
  method: augment_dihedral_8
  num_instances: 1000

logger:
  wandb:
    project: "rl4co"
    tags: ["pomo", "tsp200", "nrp_eval"]
    group: "pomo-tsp200-lr3e4"
    name: "pomo-tsp200-lr3e4"
```

Then run with:

```bash
python -m nrp eval experiment=nrp_eval/pomo_tsp_200_lr3e4
```

**Pro:** version-controlled, sharable, named, appears in `--help`.

The first line `# @package _global_` is important — it tells Hydra that
this YAML is in the global package, so `override /` (e.g. `override /model`)
escapes to the root of the config tree.

## 3. Edit the default YAMLs (rare — only for true defaults)

Only edit `configs/env/*.yaml`, `configs/model/*.yaml`,
`configs/trainer/default.yaml`, etc. when you want to **change the default
for every experiment that uses them**. For example:

| File | When to edit |
|------|--------------|
| `configs/env/tsp.yaml` | the default `num_loc: 20` (rarely; prefer CLI override per-experiment) |
| `configs/model/pomo.yaml` | `num_augment: 8` (RL4CO default; only change with a reason) |
| `configs/trainer/default.yaml` | `precision: "16-mixed"` (set per project) |
| `configs/logger/wandb.yaml` | `project: "rl4co"` (your team's W&B project) |
| `configs/paths/default.yaml` | `data_dir` / `output_dir` defaults |

If you change these, every experiment that doesn't override the same key
will pick up the new default.

## Practical tips

### Check the composed config before running

```bash
python -c "
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(config_dir='configs', version_base='1.3'):
    cfg = compose(config_name='main', overrides=[
        'experiment=nrp_eval/pomo_tsp',
        'env.generator_params.num_loc=200',
    ])
    print(OmegaConf.to_yaml(cfg))
"
```

### Tab-complete available keys

Hydra supports a `--help` flag at every level:

```bash
python -m nrp eval --help env.generator_params
```

### Reproducibility

Every Hydra run writes the exact composed config + all overrides to:

- `outputs/<date>/<time>/.hydra/config.yaml` — fully resolved config
- `outputs/<date>/<time>/.hydra/overrides.yaml` — the CLI overrides

Commit those (or the W&B run config) to recreate the run later.

### Local-only config (machine-specific overrides)

For things that should not be committed (different `data_dir`, your W&B
entity, etc.), create `configs/optional/local.yaml` and reference it from
`main.yaml` defaults:

```yaml
# configs/main.yaml
defaults:
  ...
  - optional local: default   # <- add this
  ...
```

Then any key you put in `configs/optional/local.yaml` overrides the
default for that machine. This is the RL4CO pattern.

### Common keys to know

| Key | What it controls |
|-----|------------------|
| `experiment` | which `experiment/nrp_eval/<name>.yaml` to compose |
| `env.name` | routing env (tsp, cvrp, …) |
| `env.generator_params.num_loc` | problem size |
| `solver_name` | which solver to use (pomo, am, ortools_tsp, …) |
| `solver.<name>.decode_type` | decoder (greedy, sampling, …) |
| `solver.<name>.num_augment` | dihedral-8 augmentation factor (RL only) |
| `evaluate.method` | eval decoder (greedy, augment_dihedral_8, …) |
| `evaluate.num_instances` | eval batch size |
| `trainer.max_epochs` | training epochs |
| `model.batch_size` | training batch size |
| `model.optimizer_kwargs.lr` | learning rate |
| `seed` | global RNG seed |
| `logger` | wandb, csv, or null |

## Quick workflow recommendation

1. **First run:** start with an existing experiment and add CLI overrides
   to feel out the config space.
2. **Repeatable recipe:** once you settle on a setting, copy the CLI
   overrides into a new `experiment/nrp_eval/<my_recipe>.yaml` so you
   (and collaborators) can re-run with one flag.
3. **Project-wide defaults:** if you find yourself always changing the
   same thing (e.g., a particular W&B project name or a different
   `precision`), edit the matching default YAML once.
