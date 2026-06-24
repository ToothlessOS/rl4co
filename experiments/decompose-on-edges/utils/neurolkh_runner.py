"""NeuroLKH edge-score inference.

A thin wrapper around the upstream ``NeuroLKH/test.py`` pipeline. For each
TSP instance we:

1. Run the LKH-3 binary in ``FeatGenerate`` mode to dump the 20-NN graph
   (edge_index, edge_feat, inverse_edge_index) for every node.
2. Forward the graph through ``SparseGCNModel`` (the pretrained
   NeuroLKH sparse GCN) to get per-edge log-probabilities of being in
   the optimal tour.
3. Stitch the per-node, per-candidate scores into a symmetric ``(n, n)``
   matrix where ``S[i, j]`` is the NeuroLKH score for edge ``(i, j)``
   (``NaN`` if neither ``i`` nor ``j`` named the other in its 20-NN).

The wrapper avoids the upstream ``multiprocessing.Pool`` / ``result/...``
directory tree and instead uses ``tempfile.TemporaryDirectory`` so each
instance is fully self-contained.

Notes on conventions (see ``notes/neurolkh-configs.md``):

* LKH FeatGenerate writes 1-indexed ``edge_index`` and
  ``inverse_edge_index``. We subtract 1 before returning so callers
  work in 0-indexed Python.
* ``edge_feat`` is stored scaled by ``1e6`` in the TSPLIB file; we
  divide back to raw distances (matching the upstream ``read_feat``).
* SGN output ``y_pred_edges[:, :, 1]`` is the *log*-probability of being
  an optimal-edge among the 20 candidates at each node (softmax over
  the 20, then ``cat([1-p, p], dim=2)`` + ``log``). We ``exp`` it back
  to a probability in (0, 1] before filling the sparse matrix.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_NEUROLKH_DIR = _ROOT / "NeuroLKH"

_DEFAULT_LKH_BINARY = _NEUROLKH_DIR / "LKH"
_DEFAULT_CHECKPOINT = _NEUROLKH_DIR / "pretrained" / "neurolkh.pt"
_DEFAULT_DATASET_NAME = "decompose_on_edges"

# Edge fan-out assumed by the pretrained SGN model. Anything other than
# 20 needs re-training / finetuning; we refuse to run rather than silently
# producing garbage.
_N_EDGES = 20


# ---------------------------------------------------------------------------
# Binary build / path
# ---------------------------------------------------------------------------


def build_neurolkh_binary_if_needed(
    force: bool = False,
    neurolkh_dir: Path | None = None,
) -> Path:
    """Ensure the NeuroLKH LKH-3 binary exists at ``NeuroLKH/LKH``.

    Runs ``make -C NeuroLKH/SRC`` if the binary is missing or ``force`` is
    True. Returns the absolute path to the binary.

    On modern GCC (10+), the legacy LKH-3 C headers trigger multiple-
    definition link errors; the upstream Makefile was patched to add
    ``-fcommon`` (see commit history). If the binary is missing entirely
    *and* the Makefile is unmodified, we patch it here as a fallback so
    fresh clones work without a manual step.
    """
    root = neurolkh_dir or _NEUROLKH_DIR
    binary = root / "LKH"
    if binary.exists() and not force:
        return binary

    src_makefile = root / "SRC" / "Makefile"
    if src_makefile.exists():
        text = src_makefile.read_text()
        if "-fcommon" not in text and "CFLAGS = -O3" in text:
            # Patch in-place (idempotent — only if not already present).
            src_makefile.write_text(
                text.replace(
                    "CFLAGS = -O3 -Wall -I$(IDIR) -D$(TREE_TYPE) -g",
                    "CFLAGS = -O3 -Wall -I$(IDIR) -D$(TREE_TYPE) -g -fcommon",
                )
            )

    obj_dir = root / "SRC" / "OBJ"
    obj_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale object files only when a fresh build was explicitly
    # requested (otherwise an incremental rebuild picks them up).
    if force:
        for p in obj_dir.glob("*.o"):
            p.unlink()

    subprocess.check_call(["make"], cwd=str(root))
    if not binary.exists():
        raise RuntimeError(
            f"make finished but {binary} is missing — check NeuroLKH/SRC build output"
        )
    return binary


# ---------------------------------------------------------------------------
# Feature generation (LKH-3 FeatGenerate mode)
# ---------------------------------------------------------------------------


def _write_instance_tsp(coords: np.ndarray, instance_name: str, path: Path) -> None:
    """Write a TSPLIB EUC_2D file with the ``1e6`` scaling the upstream code uses.

    Mirrors ``NeuroLKH/test.py:write_instance`` (lines 15-26). The first 10
    characters of each scaled coordinate are kept to avoid floating-point
    noise; LKH-3's EUC_2D reader rounds to nearest integer.
    """
    s = 1_000_000
    with path.open("w") as f:
        f.write(f"NAME : {instance_name}\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {len(coords)}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords):
            xs = str(x * s)[:10]
            ys = str(y * s)[:10]
            f.write(f" {i + 1} {xs} {ys}\n")
        f.write("EOF\n")


def _write_para_feats(
    instance_filename: Path,
    feat_filename: Path,
    para_filename: Path,
    *,
    seed: int = 1234,
) -> None:
    """Write a ``FeatGenerate``-mode .par file (no run, just dump features).

    Mirrors ``NeuroLKH/test.py:write_para`` (lines 28-42) for the
    ``FeatGenerate`` branch. Note the upstream typo ``GerenatingFeature``
    — LKH-3's keyword is misspelled but is what the binary expects, so we
    preserve it verbatim.
    """
    with para_filename.open("w") as f:
        f.write(f"PROBLEM_FILE = {instance_filename}\n")
        f.write("MAX_TRIALS = 1\n")
        f.write("MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 1\n")
        f.write(f"SEED = {seed}\n")
        f.write("GerenatingFeature\n")
        f.write(f"Feat_FILE = {feat_filename}\n")


def _read_feat(feat_filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Parse the feat file produced by LKH FeatGenerate.

    Mirrors ``NeuroLKH/test.py:read_feat`` (lines 44-60). Each of ``n``
    lines has 60 floats (= 20 × ``{edge_idx, dist, inverse_idx}``);
    1-indexed for edge_idx / inverse_idx, dist already scaled by 1e6.
    The final line is the LKH runtime in seconds.
    """
    edge_index: list[int] = []
    edge_feat: list[float] = []
    inverse_edge_index: list[int] = []
    with feat_filename.open("r") as f:
        lines = f.readlines()
    for line in lines[:-1]:
        parts = line.strip().split()
        for i in range(_N_EDGES):
            edge_index.append(int(parts[i * 3]) - 1)  # 1-indexed → 0-indexed
            edge_feat.append(int(parts[i * 3 + 1]) / 1_000_000.0)
            inverse_edge_index.append(int(parts[i * 3 + 2]) - 1)
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(1, -1, _N_EDGES)
    edge_feat = np.asarray(edge_feat, dtype=np.float32).reshape(1, -1, _N_EDGES)
    inverse_edge_index = np.asarray(inverse_edge_index, dtype=np.int64).reshape(
        1, -1, _N_EDGES
    )
    runtime = float(lines[-1].strip())
    return edge_index, edge_feat, inverse_edge_index, runtime


def generate_20nn_features(
    coords: np.ndarray,
    *,
    instance_name: str,
    dataset_name: str = _DEFAULT_DATASET_NAME,
    lkh_binary: Path | None = None,
    tmpdir: Path | None = None,
    timeout_s: float = 120.0,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run LKH FeatGenerate to extract the 20-NN graph for one instance.

    Returns ``(edge_index, edge_feat, inverse_edge_index, runtime)`` with
    shapes ``(1, n*20, 20)`` each and ``runtime`` the wall-clock seconds
    LKH reported. Edge indices are 0-indexed.
    """
    binary = lkh_binary or _DEFAULT_LKH_BINARY
    if not Path(binary).exists():
        raise FileNotFoundError(
            f"NeuroLKH LKH binary not found at {binary}; call "
            f"build_neurolkh_binary_if_needed() first."
        )

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2); got {coords.shape}")

    workdir = Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp(prefix="neurolkh_"))
    workdir.mkdir(parents=True, exist_ok=True)

    tsp_path = workdir / f"{instance_name}.tsp"
    para_path = workdir / f"{instance_name}.par"
    feat_path = workdir / f"{dataset_name}_{instance_name}.txt"
    log_path = workdir / f"{instance_name}.log"

    _write_instance_tsp(coords, instance_name, tsp_path)
    _write_para_feats(tsp_path, feat_path, para_path, seed=seed)

    with log_path.open("w") as logf:
        subprocess.check_call(
            [str(binary), str(para_path)],
            cwd=str(workdir),
            stdout=logf,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )

    if not feat_path.exists():
        raise RuntimeError(
            f"LKH FeatGenerate did not produce {feat_path}; check {log_path}."
        )

    return _read_feat(feat_path)


# ---------------------------------------------------------------------------
# SGN inference
# ---------------------------------------------------------------------------


def _ensure_neurolkh_on_path() -> None:
    """Inject the NeuroLKH submodule dir onto ``sys.path`` so we can
    ``import net.sgcn_model``. Idempotent — safe to call repeatedly.
    """
    p = str(_NEUROLKH_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_neurolkh_model(
    checkpoint: Path | None = None,
    device: str = "cuda",
) -> "object":
    """Load the pretrained ``SparseGCNModel``.

    The checkpoint is the ``saved`` dict produced by the upstream trainer
    (a dict with a ``"model"`` key holding the state dict). Returns the
    model in ``eval()`` mode on the requested device.
    """
    _ensure_neurolkh_on_path()
    # Import after sys.path injection.
    from net.sgcn_model import SparseGCNModel  # type: ignore[import-not-found]

    import torch

    ckpt = Path(checkpoint) if checkpoint is not None else _DEFAULT_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"NeuroLKH checkpoint not found: {ckpt}")

    model = SparseGCNModel()
    saved = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def predict_edge_scores(
    model,
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_feat: np.ndarray,
    inverse_edge_index: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """Run SGN forward and build a symmetric ``(n, n)`` score matrix.

    Returns an ``(n, n)`` float64 array where ``S[i, j]`` is the
    NeuroLKH probability (in (0, 1]) that edge ``(i, j)`` belongs to the
    optimal tour. Cells where neither endpoint named the other in its
    20-NN graph are left as ``NaN``. The diagonal is 0.

    The forward pass matches the upstream ``infer_SGN`` (test.py:108-131)
    with ``batch_size=1`` and the same channel-1 / softmax interpretation.
    """
    import torch

    coords = np.asarray(coords, dtype=np.float32)
    n = coords.shape[0]

    expected_n_edges = n * _N_EDGES
    if edge_index.size != expected_n_edges:
        raise ValueError(
            f"edge_index has {edge_index.size} entries but n={n} "
            f"requires n*20 = {expected_n_edges}"
        )
    if edge_index.shape[-1] != _N_EDGES:
        raise ValueError(
            f"edge_index last dim must be {_N_EDGES}; got {edge_index.shape}"
        )

    node_feat = torch.from_numpy(coords).view(1, n, 2).to(device)
    e_idx = torch.from_numpy(edge_index.astype(np.int64)).view(1, -1).to(device)
    inv_e_idx = (
        torch.from_numpy(inverse_edge_index.astype(np.int64)).view(1, -1).to(device)
    )
    e_feat = (
        torch.from_numpy(edge_feat.astype(np.float32))
        .view(1, -1, 1)
        .to(device)
    )

    with torch.no_grad():
        y_edges, _, _ = model.forward(
            node_feat, e_feat, e_idx, inv_e_idx, None, None, _N_EDGES
        )
    # y_edges shape: (1, n*20, 2) log-probs. Channel 1 = optimal-edge log-prob.
    # After exp + reshape, p[i, k] is the probability that edge
    # (i, edge_index[0, i, k]) is in the optimal tour.
    p = torch.exp(y_edges[0, :, 1]).detach().cpu().numpy()  # (n*20,)
    p_per_node = p.reshape(n, _N_EDGES)

    # Build the directed (n, n) score matrix. Initialize to NaN so missing
    # entries are visible. Then fill from the 20-NN graph, then symmetrize
    # with nanmax (NaN propagates only when *both* directions are missing).
    S = np.full((n, n), np.nan, dtype=np.float64)
    edge_index_np = edge_index.reshape(n, _N_EDGES)  # 0-indexed
    rows = np.repeat(np.arange(n), _N_EDGES)
    cols = edge_index_np.reshape(-1)
    S[rows, cols] = p_per_node.reshape(-1)

    # Symmetrize: max of (S, S.T) per cell, with NaN treated as -inf.
    with np.errstate(invalid="ignore"):
        S_sym = np.fmax(S, S.T)
    # NaN → NaN (preserved); zeros on the diagonal
    np.fill_diagonal(S_sym, 0.0)

    return S_sym


def compute_topk_mask(score_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """Symmetric boolean mask of edges that are in some node's top-K candidates.

    For each node ``i``, take the ``k`` highest-scoring neighbours of ``i``
    (treating NaN/scored-but-unscored as ineligible — they never enter a
    top-K). Build a directed ``(n, n)`` mask, then symmetrize by logical OR
    so an edge is flagged whenever *either* endpoint ranks it in its top-K.

    Parameters
    ----------
    score_matrix : (n, n) float64
        Symmetric score matrix from :func:`predict_edge_scores`. NaN marks
        edges outside the union of all 20-NN graphs.
    k : int
        Number of top candidates per node.

    Returns
    -------
    (n, n) bool
        ``mask[i, j] = True`` iff ``j`` is in ``i``'s top-K or ``i`` is in
        ``j``'s top-K (with NaN entries from the input propagated as False).
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1; got {k}")
    score_matrix = np.asarray(score_matrix)
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError(f"score_matrix must be square; got {score_matrix.shape}")
    n = score_matrix.shape[0]

    directed = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = score_matrix[i]
        finite = np.isfinite(row)
        finite_idx = np.where(finite)[0]
        if finite_idx.size == 0:
            continue
        # Sort by row value descending, take top-k (or fewer if row is sparse)
        top_local = np.argsort(-row[finite_idx])[: min(k, finite_idx.size)]
        directed[i, finite_idx[top_local]] = True

    return directed | directed.T


# ---------------------------------------------------------------------------
# Tour-augmented sparse graph (stage 3)
# ---------------------------------------------------------------------------
#
# The pretrained ``SparseGCNModel`` is shape-locked to ``n_edges = 20`` per
# node — the per-node softmax in ``sgcn_model.py`` and the embedding
# reshape both treat 20 as a hard dimension, and the pretrained weights
# have no way to distinguish slot 5 from slot 25 if we fed 25 slots.
# However, the model imposes no constraint on the *content* of those 20
# slots: any 20-per-node sparse graph of the right shape runs forward and
# yields a score matrix.
#
# Stage 3 exploits this. For each node ``i`` we rebuild a 20-set that
# **guarantees** the two edges incident to the heuristic tour are in the
# candidate list, displacing the two farthest of the 20 nearest neighbors
# when needed. The inverse edge index is recomputed in Python (the LKH
# ``FeatGenerate`` shortcut does not apply because we are mutating the
# per-node candidate sets). All conventions match
# ``NeuroLKH/SRC/FeatureGenerate.c:35-50``.


def _build_tour_edge_set(tour_perm: list[int], n: int) -> list[set[int]]:
    """Return per-node sets of tour neighbors.

    For each node ``i``, returns the set of nodes ``j`` such that the
    closed tour visits ``i`` and ``j`` consecutively (either order). In
    a valid undirected TSP tour each node has exactly 2 tour neighbors,
    so the returned set has size 2 (or 1 when ``n == 2``). ``n == 1``
    yields one empty set.

    Args:
        tour_perm: Closed-cycle permutation (the cycle closes implicitly
            via ``tour_perm[-1] -> tour_perm[0]``).
        n: Number of nodes.

    Returns:
        ``list[set[int]]`` of length ``n``.
    """
    if len(tour_perm) != n:
        raise ValueError(
            f"tour_perm length {len(tour_perm)} != n {n}"
        )
    if n == 1:
        return [set()]
    if n == 2:
        # Only edge is (0, 1) and its reverse; both endpoints see the other.
        return [{1}, {0}]

    sets: list[set[int]] = [set() for _ in range(n)]
    cycle = np.asarray(tour_perm, dtype=np.int64)
    next_idx = np.roll(cycle, -1)
    for a, b in zip(cycle.tolist(), next_idx.tolist()):
        sets[int(a)].add(int(b))
        sets[int(b)].add(int(a))
    return sets


def _recompute_inverse_edge_index(ei_2d: np.ndarray, n: int) -> np.ndarray:
    """Recompute the flat inverse edge index for an arbitrary 20-per-node graph.

    Python equivalent of ``NeuroLKH/SRC/FeatureGenerate.c:35-50``: for
    each directed edge ``(i, j)`` at flat position ``i * 20 + slot``,
    find the slot of the reverse edge ``(j, i)`` in ``j``'s candidate
    set. If ``j`` does not name ``i`` in its own set (asymmetric kNN),
    the inverse is ``-1``.

    Args:
        ei_2d: ``(n, 20)`` int array, 0-indexed. May contain ``-1``
            sentinels (treated as "no edge here"); such entries get
            ``-1`` inverse.
        n: Number of nodes.

    Returns:
        ``(n * 20,)`` flat int64 array. ``out[i * 20 + slot]`` is the
        flat position ``j * 20 + rev_slot`` where ``j = ei_2d[i, slot]``
        and ``rev_slot`` is the position of ``i`` in ``ei_2d[j]``;
        ``-1`` if either endpoint has no valid edge to the other.
    """
    # Pre-build per-node {neighbor: slot} dict for O(1) reverse lookup.
    slots_of: list[dict[int, int]] = [
        {int(neigh): slot for slot, neigh in enumerate(ei_2d[i])}
        for i in range(n)
    ]
    inv = np.full(n * _N_EDGES, -1, dtype=np.int64)
    for i in range(n):
        for slot in range(_N_EDGES):
            j = int(ei_2d[i, slot])
            if j < 0:
                continue
            rev = slots_of[j].get(i, -1)
            if rev >= 0:
                inv[i * _N_EDGES + slot] = j * _N_EDGES + rev
    return inv


def _pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    """``(n, n)`` Euclidean distance matrix; ``D[i, i] = 0``."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def build_tour_augmented_features(
    coords: np.ndarray,
    *,
    base_edge_index: np.ndarray,
    base_edge_feat: np.ndarray,
    base_inverse_edge_index: np.ndarray,
    tour_perm: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild a 20-per-node candidate set with the tour edges forced in.

    For each node ``i``, the new 20-set is::

        tour_set[i] ∪ (knn_set[i] \\ tour_set[i])

    where ``tour_set[i]`` has the two tour neighbors of ``i`` and
    ``knn_set[i]`` is the 20 nearest neighbors from the base 20-NN
    graph. If the union exceeds 20 (rare on TSP-100 but possible on
    small dense instances), the farthest kNN members are dropped first.
    If the union is below 20 (also rare: happens when many kNN slots
    coincide with tour slots), the leftover kNN slots are appended.

    Slot ordering within each node is deterministic:

    1. Tour edges not already in the kNN (sorted ascending by neighbor
       id), then
    2. Tour edges that already matched a kNN slot (sorted ascending by
       neighbor id), then
    3. The closest remaining kNN slots (ascending by kNN distance),
       padding to 20 with the longest dropped kNN when needed.

    The inverse edge index is recomputed in Python (see
    :func:`_recompute_inverse_edge_index`); ``base_inverse_edge_index``
    is accepted for API symmetry but ignored.

    Args:
        coords: ``(n, 2)`` Euclidean coordinates.
        base_edge_index: ``(1, n, 20)`` int array, 0-indexed, from
            :func:`generate_20nn_features`.
        base_edge_feat: ``(1, n, 20)`` float32 array (raw Euclidean
            distances, NOT the 1e6-scaled LKH values).
        base_inverse_edge_index: ``(1, n, 20)`` int array; unused
            (kept for symmetry with :func:`predict_edge_scores`).
        tour_perm: Closed-cycle permutation (length ``n``).

    Returns:
        ``(edge_index, edge_feat, inverse_edge_index)`` with shapes
        ``(1, n, 20)``, 0-indexed. ``edge_feat`` is recomputed from
        raw Euclidean distances (NOT reused from ``base_edge_feat``)
        so the units are consistent with the model's expectation.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2); got {coords.shape}")
    n = coords.shape[0]

    expected_flat = n * _N_EDGES
    if base_edge_index.size != expected_flat:
        raise ValueError(
            f"base_edge_index has {base_edge_index.size} entries but "
            f"n={n} requires n*20 = {expected_flat}"
        )
    if base_edge_index.shape[-1] != _N_EDGES:
        raise ValueError(
            f"base_edge_index last dim must be {_N_EDGES}; "
            f"got {base_edge_index.shape}"
        )

    # Unwrap to (n, 20) for the per-node rebuild.
    ei_2d = base_edge_index.reshape(n, _N_EDGES)
    ef_2d = base_edge_feat.reshape(n, _N_EDGES)

    D = _pairwise_euclidean(coords)
    tour_sets = _build_tour_edge_set(tour_perm, n)

    new_ei = np.full((n, _N_EDGES), -1, dtype=np.int64)
    new_ef = np.full((n, _N_EDGES), np.inf, dtype=np.float32)

    for i in range(n):
        tour_set = tour_sets[i]

        # Partition the 20 kNN slots into "already in tour" and "not in tour".
        # Self-loops are dropped first: LKH FeatGenerate occasionally includes
        # the node itself in its 20-NN at a small non-zero distance (the
        # upstream TSPLIB writer truncates scaled coordinates to 10 chars,
        # so the "self distance" can round to e.g. 0.001 instead of 0.0).
        in_tour_knn: list[tuple[int, float]] = []
        out_tour_knn: list[tuple[int, float]] = []
        for slot in range(_N_EDGES):
            j = int(ei_2d[i, slot])
            if j < 0 or j == i:
                continue
            d = float(ef_2d[i, slot])
            if j in tour_set:
                in_tour_knn.append((j, d))
            else:
                out_tour_knn.append((j, d))

        # Step 1+2: tour slots that did NOT come from kNN (insert first),
        # then tour slots that did come from kNN. Sort each by neighbor
        # id for determinism.
        in_tour_set = {j for j, _ in in_tour_knn}
        missing_tour = sorted(tour_set - in_tour_set)
        in_tour_existing = sorted(j for j, _ in in_tour_knn)
        new_neighbors: list[int] = list(missing_tour) + list(in_tour_existing)
        used: set[int] = set(new_neighbors)

        # Step 3: fill remaining slots with closest kNN not in tour.
        out_tour_knn.sort(key=lambda t: t[1])
        remaining = _N_EDGES - len(new_neighbors)
        for j, _ in out_tour_knn:
            if remaining <= 0:
                break
            if j not in used:
                new_neighbors.append(j)
                used.add(j)
                remaining -= 1

        # Step 4 (rare): pad with the closest unused nodes from the full
        # distance matrix. This happens when the kNN slots were heavily
        # dominated by self-loops + tour edges (so we ran out of fresh
        # candidates). Self-loops are skipped.
        if remaining > 0:
            row = D[i].copy()
            row[i] = np.inf  # skip self
            for j in np.argsort(row):
                j = int(j)
                if j in used:
                    continue
                new_neighbors.append(j)
                used.add(j)
                remaining -= 1
                if remaining <= 0:
                    break

        # Defensive: drop the farthest kNN if the union exceeded 20.
        if len(new_neighbors) > _N_EDGES:
            # Re-sort kNN contributions by distance descending; drop
            # the longest first. Tour slots are protected.
            tour_positions = len(missing_tour) + len(in_tour_existing)
            tail_with_dist = [
                (j, float(D[i, j])) for j in new_neighbors[tour_positions:]
            ]
            tail_with_dist.sort(key=lambda t: -t[1])
            kept_tail = tail_with_dist[: _N_EDGES - tour_positions]
            new_neighbors = (
                new_neighbors[:tour_positions]
                + [j for j, _ in kept_tail]
            )

        # Final assertion: 20 slots, no self-loops, no -1.
        if len(new_neighbors) != _N_EDGES:
            raise RuntimeError(
                f"node {i}: augmented candidate set has {len(new_neighbors)} "
                f"slots (expected {_N_EDGES})"
            )
        if any(j == i for j in new_neighbors):
            raise RuntimeError(f"node {i}: self-loop in augmented set")
        if any(j < 0 for j in new_neighbors):
            raise RuntimeError(f"node {i}: -1 sentinel in augmented set")

        new_ei[i] = np.asarray(new_neighbors, dtype=np.int64)
        new_ef[i] = np.asarray(
            [float(D[i, j]) for j in new_neighbors], dtype=np.float32
        )

    # Recompute inverse edge index from scratch (the base inverse is
    # now stale — we changed the per-node candidate sets).
    inverse_flat = _recompute_inverse_edge_index(new_ei, n)
    return (
        new_ei.reshape(1, n, _N_EDGES),
        new_ef.reshape(1, n, _N_EDGES),
        inverse_flat.reshape(1, n, _N_EDGES),
    )


def predict_edge_scores_augmented(
    model,
    coords: np.ndarray,
    tour_perm: list[int],
    base_edge_index: np.ndarray,
    base_edge_feat: np.ndarray,
    base_inverse_edge_index: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """One-shot helper: build the augmented graph for ``tour_perm`` and score it.

    Wraps :func:`build_tour_augmented_features` + :func:`predict_edge_scores`.
    Returns the symmetric ``(n, n)`` score matrix where ``S[i, j]`` is the
    NeuroLKH score (in (0, 1]) of edge ``(i, j)`` *given the augmented
    candidate set at node i* — i.e. the per-node softmax is over the
    augmented 20 slots, not the original 20-NN. Edges outside the union
    of the augmented graph are ``NaN``.

    Args:
        model: A loaded :class:`SparseGCNModel` (eval mode).
        coords: ``(n, 2)`` Euclidean coordinates.
        tour_perm: Closed-cycle permutation of length ``n``.
        base_edge_index: ``(1, n, 20)`` int array from
            :func:`generate_20nn_features`.
        base_edge_feat: ``(1, n, 20)`` float32 array from
            :func:`generate_20nn_features`.
        base_inverse_edge_index: Unused; accepted for API symmetry.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(n, n)`` float64 symmetric score matrix.
    """
    aug_ei, aug_ef, aug_iei = build_tour_augmented_features(
        coords,
        base_edge_index=base_edge_index,
        base_edge_feat=base_edge_feat,
        base_inverse_edge_index=base_inverse_edge_index,
        tour_perm=tour_perm,
    )
    return predict_edge_scores(
        model, coords, aug_ei, aug_ef, aug_iei, device=device
    )