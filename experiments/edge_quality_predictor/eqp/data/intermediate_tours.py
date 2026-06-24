"""On-the-fly intermediate-tour generation.

Four strategies from the spec — each returns a permutation of ``[0..n-1]``
representing a closed TSP tour:

1. **nearest_neighbor_tsp** — fast O(n^2) heuristic. Greedy: from the
   current node, pick the closest unvisited node. Cycle closes at the end.
2. **farthest_insertion_tsp** — Rosenkrantz / Stearns / Lewis 1977. Imported
   from ``experiments/decompose-on-edges/utils/farthest_insertion.py``.
3. **kopt_perturb_tsp** — apply ``n_moves`` random 2-opt (and 3-opt with
   probability ``p_3opt``) reversals to a starting tour (typically the OPT).
4. **random_edge_tour** — random permutation (uniform shuffle).
5. **opt_passthrough** — identity, returns the OPT tour unchanged.

The registry exposes all strategies under stable string keys; the dataset
samples a key from a configurable weight vector per-item to mix the
strategies. Per the spec, strategies (1) and (2) are weighted more heavily
than (3)/(4); (4) "opt" is kept for an upper-bound baseline.

Sampling happens inside ``__getitem__`` so no tours are precomputed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse the farthest-insertion implementation from the sibling experiment.
_DECOMPOSE_DIR = (
    Path(__file__).resolve().parents[3] / "decompose-on-edges"
)
if str(_DECOMPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(_DECOMPOSE_DIR))

from utils.farthest_insertion import farthest_insertion_tsp  # noqa: E402


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def nearest_neighbor_tsp(
    coords: np.ndarray,
    opt_tour: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedy nearest-neighbour heuristic.

    Args:
        coords: ``(n, 2)`` Euclidean coordinates.
        opt_tour: unused; accepted for uniform strategy signature.
        rng: optional numpy random generator for the start choice.

    Returns:
        ``(n,)`` int64 permutation. The cycle closes implicitly back to
        ``perm[0]`` when the consumer treats it as a closed loop.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))

    start = int(rng.integers(0, n)) if rng is not None else 0
    visited = np.zeros(n, dtype=bool)
    perm = np.empty(n, dtype=np.int64)
    perm[0] = start
    visited[start] = True
    for k in range(1, n):
        last = int(perm[k - 1])
        # Mask visited nodes with +inf so they're never picked.
        row = dist[last].copy()
        row[visited] = np.inf
        nxt = int(row.argmin())
        perm[k] = nxt
        visited[nxt] = True
    return perm


def kopt_perturb_tsp(
    coords: np.ndarray,
    opt_tour: np.ndarray,
    n_moves: int = 5,
    p_3opt: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random 2-opt / 3-opt reversals to ``opt_tour``.

    A 2-opt move reverses the segment between two random non-adjacent
    edges. A 3-opt move reverses a contiguous block. Both moves preserve
    the edge set for 2-opt (only ordering changes); the block reversal
    surrogate for 3-opt may drop edges.

    Args:
        coords: ``(n, 2)`` — accepted for API symmetry (k-opt is geometric
            only in the edge lengths, not in the move selection).
        opt_tour: ``(n,)`` int64 0-indexed starting permutation.
        n_moves: number of random moves to apply.
        p_3opt: probability of a 3-opt move per iteration; otherwise 2-opt.
        rng: numpy random generator.

    Returns:
        ``(n,)`` int64 permutation.
    """
    del coords  # unused
    rng = rng or np.random.default_rng()
    perm = np.asarray(opt_tour, dtype=np.int64).copy()
    n = perm.shape[0]
    if n < 4:
        return perm

    for _ in range(n_moves):
        if rng.random() < p_3opt and n >= 6:
            i, j = sorted(rng.choice(n, size=2, replace=False))
            perm[i : j + 1] = perm[i : j + 1][::-1]
        else:
            i, j = sorted(rng.choice(n, size=2, replace=False))
            if j - i > 1:
                perm[i : j + 1] = perm[i : j + 1][::-1]
    return perm


def random_edge_tour(
    coords: np.ndarray,
    opt_tour: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random Hamiltonian tour (uniform shuffle of nodes).

    Used as the "random edge connections" baseline from the spec.
    """
    del opt_tour  # unused
    rng = rng or np.random.default_rng()
    n = coords.shape[0]
    perm = np.arange(n, dtype=np.int64)
    rng.shuffle(perm)
    return perm


def opt_passthrough(
    coords: np.ndarray,
    opt_tour: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Identity: returns ``opt_tour`` unchanged.

    Provided so callers can dispatch on the strategy registry uniformly.
    """
    del coords, rng  # unused
    return np.asarray(opt_tour, dtype=np.int64).copy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _farthest_insertion_wrapper(coords, opt_tour=None, rng=None):
    del opt_tour, rng  # unused
    perm, _ = farthest_insertion_tsp(coords)
    return np.asarray(perm, dtype=np.int64)


STRATEGY_REGISTRY = {
    "nn": nearest_neighbor_tsp,
    "fi": _farthest_insertion_wrapper,
    "kopt": kopt_perturb_tsp,
    "random": random_edge_tour,
    "opt": opt_passthrough,
}

STRATEGY_WEIGHTS_DEFAULT: list[float] = [0.20, 0.15, 0.40, 0.15, 0.10]
"""Default sampling weights for ``[nn, fi, kopt, random, opt]``.

Per spec, heuristic + kopt dominate (0.20 + 0.15 + 0.40 = 0.75); random
and OPT are minority (0.15 + 0.10 = 0.25).
"""


def sample_strategy_id(
    rng: np.random.Generator, weights: list[float] | None = None
) -> int:
    """Return an index into ``STRATEGY_REGISTRY`` sampled from ``weights``."""
    w = weights if weights is not None else STRATEGY_WEIGHTS_DEFAULT
    if len(w) != len(STRATEGY_REGISTRY):
        raise ValueError(
            f"weights length {len(w)} != {len(STRATEGY_REGISTRY)} "
            "(must match STRATEGY_REGISTRY order)"
        )
    return int(rng.choice(len(w), p=np.asarray(w) / np.sum(w)))