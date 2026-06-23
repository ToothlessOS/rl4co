"""Farthest-insertion TSP heuristic (Rosenkrantz / Stearns / Lewis 1977).

Algorithm:

1. Seed the tour with the longest edge ``(a, b)``.
2. Repeat until all nodes are visited:
   a. Pick the unvisited node ``v`` farthest from the current tour
      (max of ``nearest_to_tour[v]``).
   b. Find the cheapest insertion position in the tour (the edge
      ``(tour[k], tour[k+1])`` whose replacement by the two edges
      ``(tour[k], v)`` and ``(v, tour[k+1])`` adds the least distance).
   c. Insert ``v`` at that position and update ``nearest_to_tour``.

Total cost: O(n^2) time and O(n^2) memory for the pre-computed distance
matrix. The longest-edge seed (vs. an arbitrary single-node start)
produces a tighter initial tour but does not change the asymptotic cost.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["farthest_insertion_tsp"]


def _pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    """``(n, n)`` Euclidean distance matrix; ``D[i, i] = 0``."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def farthest_insertion_tsp(coords: np.ndarray) -> Tuple[List[int], float]:
    """Solve a single Euclidean TSP instance via farthest insertion.

    Args:
        coords: ``(n, 2)`` Euclidean coordinates.

    Returns:
        ``(tour_perm, length)`` where ``tour_perm`` is a length-``n``
        permutation of ``0..n-1`` (the order of node visits; the
        cycle closes implicitly back to ``tour_perm[0]``) and ``length``
        is the closed-tour Euclidean length.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    n = coords.shape[0]
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(2.0 * _pairwise_euclidean(coords)[0, 1])

    D = _pairwise_euclidean(coords)

    # Seed: start from the longest edge so the initial tour is non-trivial.
    # ``argmax`` on the strict upper triangle (D[i, j] for i < j) avoids
    # double-counting the diagonal zeros.
    iu = np.triu_indices(n, k=1)
    flat = D[iu]
    seed_flat = int(flat.argmax())
    a = int(iu[0][seed_flat])
    b = int(iu[1][seed_flat])

    tour: List[int] = [a, b]
    visited = np.zeros(n, dtype=bool)
    visited[a] = True
    visited[b] = True

    # ``nearest_to_tour[v]`` = min distance from v to any visited node.
    # Initialize after the seed tour.
    nearest = np.full(n, np.inf, dtype=np.float64)
    for v in range(n):
        nearest[v] = min(D[v, a], D[v, b])

    for _ in range(n - 2):
        # Pick the unvisited node farthest from the current tour.
        # ``np.argmax`` on a masked array ignores visited nodes
        # (their ``nearest`` value is irrelevant; we mask them with
        # -inf to be safe).
        masked = np.where(visited, -np.inf, nearest)
        v = int(masked.argmax())

        # Find the cheapest insertion position. ``m = len(tour)``.
        m = len(tour)
        best_delta = np.inf
        best_k = 0
        for k in range(m):
            u = tour[k]
            w = tour[(k + 1) % m]
            delta = D[v, u] + D[v, w] - D[u, w]
            if delta < best_delta:
                best_delta = delta
                best_k = k
                # Tie-break: lower k wins (deterministic).

        # Insert v between tour[best_k] and tour[(best_k + 1) % m].
        tour.insert(best_k + 1, v)
        visited[v] = True

        # Update nearest-to-tour for the remaining unvisited nodes.
        diff = D[:, v] - nearest
        nearer = (~visited) & (diff < 0)
        nearest[nearer] = D[nearer, v]

    # Compute the closed-tour length: n edges cycling through ``tour``.
    length = 0.0
    for k in range(n):
        length += float(D[tour[k], tour[(k + 1) % n]])
    return tour, length