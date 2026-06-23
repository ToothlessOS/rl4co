# NeuroLKH configuration notes

Reference for the Stage-2 inference pipeline (`utils/neurolkh_runner.py`).
Pairs with `LKH3-configs.md` (Stage-1 α-nearness derivation).

## Upstream

* Repo: <https://github.com/liangxinedu/NeuroLKH>
* Paper: Xin et al., NeurIPS 2021. `SparseGCNModel` (30 GCN layers, 128 hidden) predicts edge probabilities; LKH-3 is then rerun with a candidate set restricted to the model's top-5 picks per node.
* We use only the *first* half of that pipeline (feature generation + SGN forward), discarding the LKH rerun.

## Two-stage inference

1. **FeatGenerate** — the LKH-3 binary is invoked with a `.par` file
   containing the bare minimum flags plus `GerenatingFeature` and
   `Feat_FILE = <path>`. This dumps the 20-NN graph for every node to
   a plain-text file (`NeuroLKH/test.py:38-42`). ~0.1 s for TSP-100.
2. **SGN forward** — `SparseGCNModel.forward(node_feat, edge_feat,
   edge_index, inverse_edge_index, y_edges=None, edge_cw=None,
   n_edges=20)` returns `(y_pred_edges, loss, y_pred_nodes)` with
   `y_pred_edges` shape `(batch, n*20, 2)` log-probs. ~0.1 s on CUDA
   for TSP-100.

## Feat-file format

Produced by LKH FeatGenerate, parsed by `utils/neurolkh_runner._read_feat`.
One line per node (n lines), each with 60 space-separated integers:

```
<edge_idx_1> <dist_1> <inverse_idx_1> ... <edge_idx_20> <dist_20> <inverse_idx_20>
```

* `edge_idx`, `inverse_idx` are 1-indexed; we subtract 1 before returning.
* `dist` is the integer-scaled Euclidean distance (LKH multiplies by 1e6
  internally for EUC_2D); we divide back to raw distance in `[0, 1]`.
* Line `n+1` is a single float = LKH-reported runtime in seconds.

## SGN input shapes (batch_size = 1)

| Tensor | Shape (np) | Shape (torch, view) | Notes |
|---|---|---|---|
| `node_feat` | `(n, 2)` | `(1, n, 2)` | raw `(x, y)` coords, no scaling |
| `edge_feat` | `(n, 20)` | `(1, n*20, 1)` | raw distances from feat file |
| `edge_index` | `(n, 20)` | `(1, n*20)` | 0-indexed node ids |
| `inverse_edge_index` | `(n, 20)` | `(1, n*20)` | 0-indexed node ids; `inverse_edge_index[i*20+k]` is the row in `edge_index` that points back to node `i` |

## Score interpretation

`y_pred_edges[b, i*20+k, 1] = log(p[i, k])` where

```
p[i, k] = exp(mlp_out[i, k]) / sum_{k'} exp(mlp_out[i, k'])
```

i.e. a **softmax over the 20 candidates at node `i`**, NOT the absolute
probability of edge `(i, edge_index[i, k])` being in the optimal tour.
High score = "among i's 20 closest neighbours, the model strongly
prefers this one". For nodes whose top-20 are all good candidates,
most scores hover near 1/20 ≈ 0.05; only standout edges get higher
mass.

Concretely: the mean NeuroLKH score on an LKH-2 tour is *not* reliably
higher than on a farthest-insertion tour. LKH's picks are still
within i's top-20, so they get similar per-node mass. What the score
*does* discriminate is "which edge among i's closest neighbours the
model considers most promising" — i.e. it picks a few top picks per
node (the model's top-5) and ranks them by importance.

Direction (compare with α): higher NeuroLKH score = better edge. The
opposite of α, where lower = better.

## Coverage caveat

The score matrix is **sparse**. Only ~20n directed edges (~10n unique
undirected) are scored; the remaining `n^2 − 10n` cells are `NaN`. For
TSP-100, coverage is ~65 %; for TSP-20, ~50 %. Tour edges missing
from every node's top-20 get NaN and fall back to the global non-NaN
mean in the visualization (otherwise the colormap receives NaN and
crashes).

## Pretrained model

`pretrained/neurolkh.pt` — uniform TSP, n=100, 20-NN candidate graph.
For other sizes:

* n < 20: cannot use 20-NN; the SGN forward assumes 20 candidates/node.
* n < 100: works mechanically but is out-of-distribution; expect
  qualitatively similar but noisier scores.
* n > 100: same — fine in principle, may need `finetune_node.py` to
  update the Pi predictor (the `mlp_nodes` head).
* `pretrained/neurolkh_m.pt` — uniform + clustered (n=100). Same
  caveats apply; not yet wired into the wrapper.

## Build quirks

The upstream `SRC/Makefile` was patched to add `-fcommon` to `CFLAGS`
— required on GCC 10+ which defaults to `-fno-common` and trips on
the legacy LKH-3 header definitions. `build_neurolkh_binary_if_needed`
applies the patch in-place if the file is unmodified (idempotent).
The `SRC/OBJ/` directory is created if missing.

## Gotchas

* `GerenatingFeature` (typo preserved) — LKH-3's keyword is misspelled
  but is what the binary accepts.
* `_N_EDGES = 20` is hard-coded in the wrapper and the model. Changing
  it would require retraining the SGN.
* The SGN forward uses `eval()` mode but `BatchNormNode` is configured
  with `track_running_stats=False`, so BN uses batch statistics — fine
  for inference at batch_size=1 but slightly out-of-spec.
* The feat file's runtime line is `0.0` for trivial runs; that's an
  LKH artefact, not a wrapper bug.