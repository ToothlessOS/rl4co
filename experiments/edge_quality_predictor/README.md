# Edge Quality Predictor

Train a model that takes a TSP tour (or partial tour) and predicts the
probability of each edge being in the optimal tour. Per-edge scores are a
downstream signal for tour decomposition: high-prob edges are tunneled
(preserved as a subproblem), low-prob edges are cut.

> **Status (v1).** Initial implementation. Trained from scratch with a
> forked SGN backbone; sparse graph input is the **input tour's edges
> (2 per node)**, not the 20-NN. Pretrained NeuroLKH weights are not used
> (the 20-slot shape contract doesn't transfer to 2 slots). All training
> is fully synthetic; the per-edge score is a learned scalar.

---

## What the model does

For each TSP instance the dataset builds the **2-slot edge graph** of an
input tour (each node has exactly 2 neighbours in a closed cycle) and
asks the model to label each slot as either **in-OPT** or **not-in-OPT**.
Labels come from the ground-truth OPT tour stored in the training file.

The 4 strategies from the spec are mixed on the fly per-item:

| Strategy | What it does | Default weight |
|---|---|---|
| `nn` | Greedy nearest-neighbour from a random start | 0.20 |
| `fi` | Farthest-insertion (Rosenkrantz et al. 1977) | 0.15 |
| `kopt` | Random 2-opt/3-opt reversals on OPT | 0.40 |
| `random` | Uniform random permutation | 0.15 |
| `opt` | Pass-through (identity) | 0.10 |

The kopt + opt strategies produce labels that are mostly 1 (since the
edges coincide with OPT or are unchanged). The nn/fi/random strategies
produce a mix of 0/1 depending on which tour edges actually appear in OPT.

---

## File layout

```
experiments/edge_quality_predictor/
├── README.md                  this file
├── implementation.md          stage-1 spec
├── pyproject.toml             project metadata
├── train_eqp.py               Hydra entrypoint
├── test_eqp.py                smoke tests (10 cases, run as a plain script)
├── configs/                   Hydra config tree (mirrors POMO_eval)
│   ├── eqp_main.yaml
│   ├── callbacks/
│   ├── trainer/
│   ├── hydra/
│   ├── paths/
│   ├── logger/
│   └── extras/
├── eqp/                       the package
│   ├── data/
│   │   ├── tsp_data.py        train/test/TSPlib parsers
│   │   ├── intermediate_tours.py  5 on-the-fly strategies + registry
│   │   └── dataset.py         EdgeQualityDataset + collate
│   ├── model/
│   │   ├── sgn.py             forked SparseGCNModel with n_edges_per_node=2
│   │   ├── classifier.py      SGNEdgeClassifier + edge-score matrix builder
│   │   └── losses.py          masked NLL with pos_weight
│   ├── lightning/
│   │   ├── module.py          EdgeQualityModule
│   │   └── data.py            EdgeQualityDataModule
│   └── eval/
│       └── edge_metrics.py    per-instance metrics + aggregator
├── ref/env/                   reference wrappers (untouched)
└── data/
    ├── README.md              dataset provenance + citation
    ├── TSP/                   1M TSP-100 train + 5 test sets
    └── CVRP/                  unused (TSP-only v1)
```

---

## How to run

From `experiments/edge_quality_predictor/` with the parent `.venv`
activated. Wandb logging is enabled by default; override with
`logger.wandb=null` for local-only runs.

```bash
# Smoke tests (10 cases, ~10s on CPU)
python test_eqp.py

# End-to-end dry run (16 train / 8 val, batch_size=4, 1 epoch, ~5s)
WANDB_MODE=disabled PROJECT_ROOT=. python train_eqp.py \
    data.train_limit=16 data.val_limit=8 data.test_limit=8 \
    trainer.max_epochs=1 trainer.devices=1 train=true test=false \
    model.hidden_dim=32 model.n_gcn_layers=5 model.n_mlp_layers=2 \
    data.batch_size=4 data.num_workers=0 \
    trainer.accelerator=cpu trainer.precision=32-true \
    logger.wandb=null

# Real run (defaults: n_gcn_layers=30, hidden_dim=128, batch_size=32, 20 epochs)
WANDB_MODE=disabled PROJECT_ROOT=. python train_eqp.py \
    data.train_limit=10000 data.val_limit=1000 data.test_limit=128 \
    trainer.max_epochs=20 trainer.devices=1

# Override any Hydra knob from the CLI, e.g.:
#   - smaller model: model.hidden_dim=64 model.n_gcn_layers=10
#   - more kopt weight: data.strategy_weights='[0.10,0.05,0.65,0.10,0.10]'
#   - longer training: trainer.max_epochs=50
#   - run on GPU: trainer.accelerator=gpu trainer.devices=1
```

### Config keys of interest

`configs/eqp_main.yaml` exposes:

- `data.train_path`, `data.val_path`, `data.test_paths`
- `data.train_limit`, `data.val_limit`, `data.test_limit`
- `data.pad_to_n` — pad all items to this n (default 100, matches the
  largest available training file)
- `data.strategy_weights` — 5 floats summing to 1
- `data.kopt_n_moves`, `data.kopt_p_3opt` — kopt aggressiveness
- `data.batch_size`, `data.num_workers`
- `model.hidden_dim`, `model.n_gcn_layers`, `model.n_mlp_layers`
- `model.pos_weight` — class weight for OPT-positive slots (default 9.0)
- `optim.lr`, `optim.weight_decay`
- `trainer.max_epochs`, `trainer.devices`, `trainer.accelerator`

---

## Verification

### Smoke tests (`test_eqp.py`)

Run from the experiment directory. Each test prints `OK` or `FAIL`.

| Test | What it checks |
|---|---|
| `parse_train_line` | `train_TSP100_n100w-002.txt` first line → (100, 2) coords + length-100 permutation |
| `parse_tsplib` | `TSPlib_70instances.txt` → name, cost, coords |
| `intermediate_tours` | All 5 strategies return valid permutations; kopt drops ~4n edges; opt==opt; random≠opt |
| `edge_graph_build` | 4-node cycle → edge_index = [3, 1], distances = 1.0, inverse correctly tracks reverse slots |
| `label_derivation` | OPT strategy produces 2n directed positives (one per endpoint of each undirected OPT edge) |
| `sgn_v_forward` | `SparseGCNModelV` forward returns `(B, n*2, 2)` log-probs; loss is finite |
| `classifier_forward` | `SGNEdgeClassifier` forward + backward + `predict_edge_scores` shapes |
| `pad_mask_correctness` | Mixed-size batch (60/80/100) → `pad_mask` correctly encodes validity; loss only counts valid slots |
| `metrics_basic` | `edge_quality_metrics` + `aggregate_metrics` on synthetic data |
| `end_to_end_short` | Full Lightning fit on 16 train / 8 val, 1 epoch, CPU |

### Headline metrics

The Lightning module logs:

- `train/loss`, `val/loss`, `test/loss`
- `val/edge_accuracy`, `val/balanced_accuracy`, `val/precision_at_1`, `val/precision_at_2`
- `val/top1_hit_rate`, `val/mean_p_in_opt`
- `val/pos_pred_rate`, `val/neg_pred_rate`, `val/mean_p`
- Per-strategy: `val/strategy_{nn,fi,kopt,random,opt}/<metric>`

Sanity invariants on an untrained model (uniform ~50/50 predictions):

- `edge_accuracy ≈ 0.50` (random)
- `precision_at_2 ≈ fraction of OPT edges among the 2 input edges`:
  ~0.50–0.71 across the 4 mixed strategies on a typical instance.
- For the `kopt` strategy, most labels are 1, so `precision_at_2` is high
  (>0.85) even before training.

After training the model should move `edge_accuracy` substantially above
0.50 and push `top1_hit_rate` toward 1 on the easier strategies.

---

## Reused utilities

| Purpose | Path |
|---|---|
| Farthest-insertion heuristic | `experiments/decompose-on-edges/utils/farthest_insertion.py::farthest_insertion_tsp` |
| Train-file line format | `experiments/edge_quality_predictor/ref/env/TSPEnv.py::load_raw_data` |
| TSPlib format | `experiments/edge_quality_predictor/ref/env/TSPEnv_inTSPlib.py::make_tsplib_data` |
| SGN backbone (to fork) | `experiments/decompose-on-edges/NeuroLKH/net/sgcn_model.py` |
| SGN layers (to fork) | `experiments/decompose-on-edges/NeuroLKH/net/sgcn_layers.py` |
| Lightning training pattern | `experiments/POMO_eval/train_pomo_tsp.py` |
| Hydra config scaffold | `experiments/POMO_eval/configs/` |
| Trainer wrapper | `rl4co/utils/trainer.py::RL4COTrainer` |

No LKH-3 dependency. The SGN fork is a pure-Python module under `eqp/model/`.

---

## Notes & caveats

- **No pretrained weights.** The forked SGN is trained from scratch
  (random init). With 30 GCN layers and hidden_dim=128 the model has
  ~990K parameters; even a small subset of the 1M training instances
  trains it adequately.
- **Varying n.** Padding to a fixed `pad_to_n` handles any n at training
  and eval time; the loss ignores padded slots via `pad_mask` and
  `y_edges = -100`.
- **TSPlib eval.** TSPlib instances have no stored OPT tour (only the
  cost), so OPT-dependent metrics are NaN. The forward pass and
  edge_accuracy still work.
- **kopt semantics.** A 2-opt move on a tour drops 4 edges (the two
  endpoints of each removed edge) and adds 4 new ones. After
  `kopt_n_moves` perturbations, most labels are still 1 (only the edges
  involved in the moves flip).
- **Per-strategy metrics.** Use `fixed_strategy=N` at dataset construction
  time to force every item to come from strategy `N`. Useful for
  per-strategy eval sweeps.

---

## Future work

- Re-introduce pretrained NeuroLKH weights by padding the 2-slot graph
  to 20 slots with NN edges (preserves the pretrained shape contract).
- Add the 0/1 edge feature as an extra input dimension (currently only
  used as the target).
- Wire LKH-3 augmentation (LKH rerun from a starting tour) as a 5th
  strategy for harder training samples.
- Eval the per-edge predictor as a downstream signal for tunnel/keep
  decomposition (compare to α-nearness from the sibling experiment).