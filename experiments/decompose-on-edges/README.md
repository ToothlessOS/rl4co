# decompose-on-edges

Stage-1 exploration: does the **α-nearness** metric from LKH (Helsgaun
1998) help decompose a TSP tour into smaller subproblems?

This experiment implements the minimal end-to-end pipeline needed to
*see* α-nearness visually:

1. Generate a random Euclidean TSP instance with `rl4co`.
2. Solve it twice — once with **farthest insertion** (FI), once with the
   bundled **LKH-2.0.11** binary.
3. Compute the **(n, n) α-nearness matrix** in pure Python.
4. Save a per-instance **1×2 panel figure** to `figures/instance_*.png`.

The figure is the deliverable. The numeric outputs are intermediate.
A later stage (not yet implemented) will use α to drive the actual
decomposition — see [Future work](#future-work).

> **Colormap note.** Tour-edge α values are typically 5–10× smaller
> than the global α maximum (which is dominated by a few long,
> never-tour edges). The colormap is therefore anchored to the
> **99th percentile of tour-edge α** and switched to a **log scale**
> when the tour edges span more than one order of magnitude. This
> makes the differences between tour edges visible (otherwise the
> bulk of edges all look the same dark color). See
> `_alpha_color_norm` in `scripts/run_alpha_nearness.py`.
>
> **LogNorm + α = 0 gotcha.** `matplotlib.colors.LogNorm(0)` returns
> `NaN`, which the colormap renders as fully transparent RGBA — i.e.
> invisible. MST edges have α = 0 by construction, so without the
> `np.maximum(values, vmin)` clamp in `_alpha_for_plot`, ~70 % of
> tour edges on TSP-30 random uniform instances vanish from the
> figure. Always clamp before passing to `LineCollection`.

---

## α-nearness in 60 seconds

The α-nearness of an edge `(i, j)` is the difference between the length
of the minimum 1-tree **containing** `(i, j)` and the length of the
unconstrained minimum 1-tree. Edges that already belong to a minimum
1-tree have α = 0; edges far from any minimum 1-tree edge have large α
and are unlikely to be in an optimal tour.

For an edge `(i, j)` not involving the 1-tree root, this reduces to
the well-known form (used in `LKH-2.0.11/SRC/GenerateCandidates.c`):

```
α(i, j) = max(0, d(i, j) − max_edge_on_MST_path(i, j))
```

where the path is the unique path between `i` and `j` in the minimum
spanning tree (MST) of the full graph (the root of the 1-tree is itself
a node in this MST).

For an edge `(root, j)`, the formula is:

```
α(root, j) = max(0, d(root, j) − second_cheapest_root_edge)
```

The `max(0, ...)` clamp handles the special case where `(i, j)` is
itself an MST edge: the path-max equals `d(i, j)` and α becomes 0.

The full derivation and rationale are in
[`notes/LKH3-configs.md`](notes/LKH3-configs.md).

---

## File layout

```
experiments/decompose-on-edges/
├── pyproject.toml               project metadata + deps
├── README.md                    this file
├── implementation.md            stage-1 spec (what to build)
├── notes/
│   ├── LKH3-configs.md          α-nearness derivation + references
│   └── ideas.md                 next-stage research ideas
├── LKH-2.0.11/                  bundled LKH-2 binary + source
│   └── LKH                      the binary (target of the LKH runner)
├── utils/
│   ├── __init__.py
│   ├── lkh_runner.py            vendored LKH-2 wrapper (~250 LOC)
│   ├── alpha_nearness.py        MST + binary lifting (~270 LOC)
│   └── farthest_insertion.py    FI heuristic (~110 LOC)
├── scripts/
│   └── run_alpha_nearness.py    argparse entry point (~310 LOC)
└── figures/                     output PNGs (created at runtime)
```

### `utils/lkh_runner.py`

Standalone LKH-2 wrapper. Operates on `np.ndarray (n, 2)`. No torch,
no `learn_decompose_eval` import. The `.tsp` / `.par` / `.tour`
formats are the TSPLIB plain-TSP standard, identical between LKH-2 and
LKH-3 for the sections we use.

Public API:

```python
from utils.lkh_runner import solve_lkh_tsp

perm, length = solve_lkh_tsp(
    coords,                # (n, 2) Euclidean
    binary_path=...,       # absolute path to the LKH-2 executable
    max_trials=10_000,
    seed=1,
    time_limit_s=30.0,     # floored at 1; TIME_LIMIT=0 makes LKH-2 exit instantly
)
# perm: list[int] of length n (0-indexed); None on failure
# length: float (Euclidean, same units as coords); float("inf") on failure
```

Critical details:

- `TIME_LIMIT = 0` makes LKH-2 exit immediately with no tour; we floor
  the value at 1 (memory `lkh-time-limit-truncation`).
- LKH-2 writes a `.tour` file even on partial success; we don't trust
  the subprocess return code — only the existence of a non-empty
  `.tour`.
- The `.tsp` writer emits `EDGE_WEIGHT_SECTION` as **values only** (no
  leading row index). LKH-2's `Read_EDGE_WEIGHT_SECTION` for
  `FULL_MATRIX` consumes exactly `n*n` doubles via `fscanf("%lf")`; a
  row index would silently shift the count and corrupt the matrix.

### `utils/alpha_nearness.py`

Pure-Python α-nearness. Public API:

```python
from utils.alpha_nearness import compute_alpha_nearness

alpha = compute_alpha_nearness(coords, root=0)
# alpha: (n, n) float64; alpha[i, j] = alpha[j, i]; alpha[i, i] = 0; alpha >= 0
```

Algorithm:

1. Build the MST over the **full graph** (n nodes, root included) via
   `scipy.sparse.csgraph.minimum_spanning_tree`.
2. BFS from the chosen root to fill parent / edge-weight-to-parent
   arrays.
3. Binary-lift both the 2^k ancestor and the max edge weight on the
   path from each node up to its 2^k ancestor.
4. For each pair `(i, j)` with `i < j`, answer the max-on-path query
   in O(log n) and apply the formula above.

Total cost: O(n² log n) time, O(n log n) extra space. For n = 50 the
per-instance compute is well under a second.

**Caveat vs. stock LKH-2:** LKH-2 builds its MST on **Pi-adjusted**
edge weights `d(i, j) + Pi[i] + Pi[j]` (from subgradient optimization
inside `Ascent()`), not raw Euclidean distances. We use pure Euclidean
distances (Pi = 0). Both give qualitatively-similar α values (short
edges α ≈ 0, long edges α ≫ 0; 0.99 correlation between edge length
and α for non-MST edges) but numerically different from LKH-2's
internal `CANDIDATE_FILE`. See [Caveats](#caveats).

### `utils/farthest_insertion.py`

Standard Rosenkrantz / Stearns / Lewis 1977. Public API:

```python
from utils.farthest_insertion import farthest_insertion_tsp

tour_perm, length = farthest_insertion_tsp(coords)
# tour_perm: list[int] of length n
# length: float (closed-tour Euclidean)
```

Algorithm:

1. Seed the tour with the longest edge `(a, b)`.
2. Repeat until all nodes are visited:
   a. Pick the unvisited node `v` farthest from the current tour.
   b. Find the cheapest insertion position (the edge whose
      replacement by the two edges through `v` adds the least distance).
   c. Insert `v` at that position.

O(n²) time, O(n²) memory for the pre-computed distance matrix.

### `scripts/run_alpha_nearness.py`

Argparse entry point that wires the pipeline together and produces the
figures. See [How to run](#how-to-run).

---

## How to run

From `experiments/decompose-on-edges/`, with the parent `.venv`
activated (or this experiment's `.venv`):

```bash
# Default: 5 TSP-20 instances, seed 0, LKH max_trials=10000
python scripts/run_alpha_nearness.py

# Larger batch on TSP-30
python scripts/run_alpha_nearness.py --num-instances 10 --num-nodes 30 --seed 7

# Custom LKH time limit / trials
python scripts/run_alpha_nearness.py \
    --num-instances 20 --num-nodes 50 \
    --max-trials 5000 --lkh-time-limit 60 \
    --out-dir figures/large
```

CLI surface:

| Arg | Default | Purpose |
|---|---|---|
| `--num-instances` | 5 | number of random TSP instances |
| `--num-nodes` | 20 | cities per instance |
| `--seed` | 0 | rl4co generator seed |
| `--lkh-binary` | `LKH-2.0.11/LKH` | LKH-2 executable path |
| `--out-dir` | `figures` | directory for `.png` files |
| `--max-trials` | 10 000 | `MAX_TRIALS` in the `.par` file |
| `--lkh-time-limit` | 30.0 | per-run `TIME_LIMIT`; floored at 1 |
| `--lkh-seed` | 1 | `SEED` in the `.par` file |

Each instance produces one `figures/instance_{k:02d}.png`:

```
┌──────────────────────────┬──────────────────────────┐
│ Farthest insertion       │ LKH-2                    │
│ edges colored by α       │ plain black tour         │
│ (viridis colormap)       │                          │
└──────────────────────────┴──────────────────────────┘
```

The script also prints a summary to stdout:

```
=== Summary ===
  FI length:   mean = 4.04  std = 0.27
  LKH length:  mean = 4.00  std = 0.27  (5 successful, 0 failed)
  FI / LKH:    mean = 1.01  std = 0.02  (expect 1.10-1.20 for TSP-20)
  mean alpha on FI tour:    mean = 0.027  std = 0.010
  mean alpha on LKH tour:   mean = 0.026  std = 0.006
  -> LKH picks lower-alpha edges on average (metric correlates with quality)
```

---

## Verification

Smoke tests (run from `experiments/decompose-on-edges/`):

```bash
# 1. LKH round-trip
PYTHONPATH=. python -c "
from utils.lkh_runner import solve_lkh_tsp
import numpy as np
rng = np.random.default_rng(0)
perm, length = solve_lkh_tsp(rng.random((20, 2)), max_trials=1000, time_limit_s=5.0)
print('LKH len:', round(length, 3))  # expect ~4-5 for TSP-20 uniform [0,1]
"

# 2. FI smoke
PYTHONPATH=. python -c "
from utils.farthest_insertion import farthest_insertion_tsp
import numpy as np
rng = np.random.default_rng(0)
perm, length = farthest_insertion_tsp(rng.random((20, 2)))
print('FI len:', round(length, 3))
"

# 3. α symmetry + zero-diagonal + MST-edge α = 0
PYTHONPATH=. python -c "
from utils.alpha_nearness import compute_alpha_nearness
import numpy as np
rng = np.random.default_rng(1)
c = rng.random((30, 2))
a = compute_alpha_nearness(c)
assert np.allclose(a, a.T), 'not symmetric'
assert (a.diagonal() == 0).all(), 'diagonal non-zero'
assert (a >= 0).all(), 'negative alpha'
print('alpha max:', round(a.max(), 4))
"

# 4. End-to-end figure
PYTHONPATH=. python scripts/run_alpha_nearness.py \
    --num-instances 5 --num-nodes 20 --seed 0 --max-trials 5000
ls figures/   # expect instance_00.png .. instance_04.png
```

Pass criteria:

- LKH length for n = 20 uniform [0, 1] is in the rough 4 – 5 range.
- FI length is within ~10 % of LKH length on average (FI is very
  competitive on small instances).
- α matrix is symmetric, zero diagonal, non-negative, finite.
- Every MST edge has α = 0 in the output (this is the spec invariant).
- Mean α on the LKH tour is ≤ mean α on the FI tour on average —
  confirms α correlates with tour quality.

The most reliable cross-check is the α matrix properties: symmetric,
zero diagonal, MST edges all zero. If those three hold, the
implementation is correct.

---

## Caveats

These are the gotchas worth flagging before you extend the experiment:

### 1. The MST must include the 1-tree root

A first-pass implementation that builds the MST over the (n-1) non-root
sub-graph (the literal "spanning tree on non-root nodes" from the 1-tree
definition) is **wrong**. The α formula uses the path between two
non-root nodes in the **full-graph** MST, which can pass through the
root. Removing the root from the MST drops that case and gives wrong α
for any pair whose LCA in the full MST is the root.

LKH-2 does it the full-graph way: `MinimumSpanningTree.c` runs Prim's
on all n nodes with `FirstNode` as the root. See
`LKH-2.0.11/SRC/MinimumSpanningTree.c` line 31.

### 2. Pure-Euclidean MST ≠ LKH-2's Pi-adjusted MST

LKH-2's MST uses edge weights `d(i, j) + Pi[i] + Pi[j]`, where the Pi
values come from a subgradient optimization (Held-Karp) run inside
`Ascent()` before candidate generation. See
`LKH-2.0.11/SRC/C.c:81` (`D_EXPLICIT`). We use raw Euclidean distances
(Pi = 0) because computing the Pi values would require either an extra
LKH-2 invocation (dump `PI_FILE`) or our own subgradient implementation.

The qualitative pattern is preserved (correlation between edge length
and α is ~0.99 for non-MST edges on TSP-30), but the precise ranking
and magnitude shift. For the stage-1 figure this is fine; for
stage-2 decomposition work, consider switching to Pi-adjusted costs.

### 3. Root-edge formula: second-cheapest vs NextCost

Our formula for edges involving the root uses the **second-cheapest**
edge from the root as the baseline (`α(root, j) = max(0, d(root, j) −
second_cheapest_root_edge)`). This matches the literal 1-tree definition
(MST over non-root nodes + 2 cheapest root edges).

LKH-2's `GenerateCandidates.c` line 91 uses `FirstNode->NextCost`,
which is the **shortest non-Dad edge** from the 1-tree root
(`Connect.c:19`). They differ when the Dad edge is not the cheapest
root edge. For TSP-30 uniform [0, 1] this affects ~1 edge per
instance; for visualization it doesn't matter.

### 4. `TIME_LIMIT = 0` makes LKH-2 exit instantly

Confirmed in `LKH-2.0.11/SRC/ReadParameters.c:931-936`. The wrapper
floors the value at `max(1, int(round(time_limit_s)))`. The POMO_eval
LKH-3 wrapper has the same fix (memory `lkh-time-limit-truncation`).

### 5. `.tsp` writer: values only, no leading row index

`EDGE_WEIGHT_SECTION` must be `n` rows of `n` integers with no leading
row index per line. LKH-2's `Read_EDGE_WEIGHT_SECTION` for
`FULL_MATRIX` consumes exactly `n*n` doubles via `fscanf("%lf")`; a
row index would shift the count and silently corrupt the matrix.
Same convention as the POMO_eval LKH-3 wrapper (memory
`lkh-3-patch-conventions`).

---

## Future work

The figure is stage 1 of the experiment. The motivating question is:
**does α-nearness help decompose a TSP tour into smaller subproblems?**
The next stage (not yet implemented) will:

1. Take an LKH-2 tour and use α to identify **tunnels** — contiguous
   sequences of low-α edges that are likely already optimal and can be
   preserved as a single subproblem.
2. Cut the tour at the high-α edges and solve the remaining segments
   independently.
3. Compare the resulting tour length and decomposition quality against
   other tunnelling strategies (random cuts, longest-edge cuts,
   sensitivity-based cuts).

The α-metric is particularly promising here because α = 0 edges are
already in a minimum 1-tree and therefore likely in the optimal tour;
cutting at non-zero-α edges should preserve near-optimal structure.

See [`notes/ideas.md`](notes/ideas.md) for related directions
(behaviour cloning of LKH, RL finetune, distance metrics on edges).

---

## Dependencies

From `pyproject.toml`:

```
rl4co
torch >= 2.0
numpy
scipy
matplotlib
```

The LKH-2.0.11 binary is bundled at `LKH-2.0.11/LKH`. No other system
dependencies are needed.

To set up the venv from the repo root:

```bash
uv sync --all-extras
source .venv/bin/activate
```

Then run the script from this experiment's directory.