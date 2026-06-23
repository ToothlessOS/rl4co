"""alpha-nearness (Helsgaun 1998) for symmetric Euclidean TSP.

The ``alpha``-nearness of an edge ``(i, j)`` is defined as the difference
between the length of the minimum 1-tree *containing* ``(i, j)`` and the
length of the unconstrained minimum 1-tree. Edges that already belong
to the minimum 1-tree have ``alpha = 0``; edges far from any minimum
1-tree edge have large ``alpha`` and are unlikely to be in an optimal
tour.

For an edge ``(i, j)`` not involving the 1-tree root, the definition
reduces to the well-known form used by LKH-2 in
``SRC/GenerateCandidates.c`` (lines 4-30, 90-101)::

    alpha(i, j) = max(0, d(i, j) - max_edge_on_MST_path(i, j))

where the path is the unique path in the MST over the full graph (the
root of the 1-tree is itself a node in the MST). For an edge
``(root, j)`` the formula is::

    alpha(root, j) = max(0, d(root, j) - NextCost(root))

where ``NextCost(root)`` is the second-cheapest edge from ``root`` to
the rest of the 1-tree (the 1-tree has exactly two edges incident to
the root). The "max(0, ...)" clamp covers the special case where
``(i, j)`` is itself an MST edge: the path-max equals ``d(i, j)`` and
``alpha`` becomes exactly 0.

We use ``root = 0`` (rl4co's depot convention). The choice of root is
arbitrary for alpha-nearness purposes — only the 1-tree cost changes,
and the ``alpha`` values are invariant up to a permutation of which
edges get the "root edge" special case.

Caveat vs. stock LKH-2:

LKH-2 builds its MST on **Pi-adjusted** edge weights
``d(i, j) + Pi[i] + Pi[j]`` where the ``Pi`` values come from a
subgradient optimization (Held-Karp) run inside ``Ascent()`` before
candidate generation. We use pure Euclidean distances (Pi = 0) — both
because the Pi values would require an extra LKH-2 invocation (or our
own subgradient implementation) and because the experiment's purpose
is the qualitative pattern of alpha-nearness on a tour, which is
preserved. Concretely: short edges still get ``alpha ≈ 0`` and long
edges still get ``alpha >> 0``; only the precise ranking and magnitude
shift slightly compared to LKH-2's internal values.

Algorithm:

1. Build the MST over the full graph (n nodes, root included) via
   :func:`scipy.sparse.csgraph.minimum_spanning_tree`.
2. Run a BFS from the chosen root to fill parent / edge-weight-to-parent
   arrays.
3. Binary-lift both the 2^k ancestor and the max edge weight on the
   path from each node up to its 2^k ancestor.
4. For each pair ``(i, j)`` with ``i < j``, answer the max-on-path query
   in O(log n) and apply the formula above.

Total cost: O(n^2 log n) time, O(n log n) extra space. For n = 50 the
per-instance compute is well under a second.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

__all__ = ["compute_alpha_nearness"]


# ---------------------------------------------------------------------------
# Pairwise Euclidean
# ---------------------------------------------------------------------------


def _pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    """``(n, n)`` Euclidean distance matrix; ``D[i, i] = 0``."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


# ---------------------------------------------------------------------------
# MST extraction + binary lifting
# ---------------------------------------------------------------------------


def _mst_with_edges(
    coords: np.ndarray,
    root: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the MST over the **full** graph and lift its structure.

    LKH's alpha-nearness uses the MST of the full graph (n nodes,
    including the 1-tree root) — the path between two non-root nodes
    may pass through the root. Building the MST over the (n-1)-node
    sub-graph would silently drop that case and produce wrong values
    for any pair whose LCA in the full-MST is the root.

    Args:
        coords: ``(n, 2)`` Euclidean coordinates (all n nodes).
        root: Index of the 1-tree root.

    Returns:
        mst_cost: scalar total MST cost.
        parent: ``(n,)`` int array; ``parent[root] = -1``.
        edge_weight_to_parent: ``(n,)`` float array; weight of the MST
            edge from ``v`` to ``parent[v]``; ``-1.0`` for ``root``.
        up: ``(LOG, n)`` int array for binary lifting.
        mx: ``(LOG, n)`` float array; ``mx[k, v]`` is the max edge
            weight on the path from ``v`` up to ``up[k, v]``.
        depth: ``(n,)`` int array; depth from ``root`` in the rooted MST.
    """
    n = coords.shape[0]
    D = _pairwise_euclidean(coords)
    np.fill_diagonal(D, 0.0)
    mst = minimum_spanning_tree(csr_matrix(D))

    coo = mst.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    weights = coo.data.astype(np.float64)
    mst_cost = float(weights.sum())

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in zip(rows, cols, weights):
        adj[int(u)].append((int(v), float(w)))
        adj[int(v)].append((int(u), float(w)))

    parent = np.full(n, -1, dtype=np.int64)
    edge_weight_to_parent = np.full(n, -1.0, dtype=np.float64)
    depth = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    visited[root] = True
    queue: deque[int] = deque([root])
    while queue:
        u = queue.popleft()
        for v, w in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                edge_weight_to_parent[v] = w
                depth[v] = depth[u] + 1
                queue.append(v)

    if not visited.all():
        raise RuntimeError(
            "MST BFS did not visit all nodes — the MST is disconnected, "
            "which should be impossible for a connected input."
        )

    up, mx = _binary_lift(n, parent, edge_weight_to_parent)
    return mst_cost, parent, edge_weight_to_parent, up, mx, depth


def _binary_lift(
    n: int, parent: np.ndarray, edge_weight_to_parent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build binary-lifting tables over a rooted tree.

    Args:
        n: Number of nodes.
        parent: ``(n,)`` int array; ``parent[root] = -1``.
        edge_weight_to_parent: ``(n,)`` float array; weight of the edge
            from ``v`` to ``parent[v]``; ``-1.0`` for the root.

    Returns:
        up: ``(LOG, n)`` int array; ``up[k, v]`` is the 2^k-th ancestor
            of ``v`` (clamped to ``-1`` past the root).
        mx: ``(LOG, n)`` float array; max edge weight on the path from
            ``v`` up to ``up[k, v]``. The root's ``mx[0, root] = -1``
            contributes a sentinel low value (we never query a single
            element by accident — paths have at least one edge).
    """
    if n <= 1:
        # Empty lift tables; only the trivial root exists.
        return (
            np.full((1, n), -1, dtype=np.int64),
            np.full((1, n), -1.0, dtype=np.float64),
        )
    LOG = (n - 1).bit_length() + 1
    up = np.full((LOG, n), -1, dtype=np.int64)
    mx = np.full((LOG, n), -1.0, dtype=np.float64)
    up[0] = parent
    mx[0] = edge_weight_to_parent
    for k in range(1, LOG):
        prev_up = up[k - 1]
        prev_mx = mx[k - 1]
        cur_up = up[k]
        cur_mx = mx[k]
        for v in range(n):
            mid = prev_up[v]
            if mid == -1:
                cur_up[v] = -1
                cur_mx[v] = prev_mx[v]
            else:
                cur_up[v] = prev_up[mid]
                # max(prev_mx[v], edge from v up to mid,
                #     edge from mid up to up[k, v])
                # == max(prev_mx[v], prev_mx[mid])
                a = prev_mx[v]
                b = prev_mx[mid]
                cur_mx[v] = a if a >= b else b
    return up, mx


def _max_on_path(
    u: int,
    v: int,
    up: np.ndarray,
    mx: np.ndarray,
    depth: np.ndarray,
) -> float:
    """Return the max edge weight on the path between two MST nodes.

    Both endpoints are in the same connected MST (guaranteed by the
    caller). The path is empty when ``u == v``; we return ``-1.0`` to
    represent "no edge", which the caller clamps via ``max(0, ...)``.
    """
    if u == v:
        return -1.0

    LOG = up.shape[0]
    running_max = -1.0

    # Lift the deeper node until both are at the same depth.
    du = int(depth[u])
    dv = int(depth[v])
    if du > dv:
        diff = du - dv
        for k in range(LOG):
            if diff & (1 << k):
                w = mx[k, u]
                if w > running_max:
                    running_max = w
                u = int(up[k, u])
                if u == -1:
                    break
    elif dv > du:
        diff = dv - du
        for k in range(LOG):
            if diff & (1 << k):
                w = mx[k, v]
                if w > running_max:
                    running_max = w
                v = int(up[k, v])
                if v == -1:
                    break

    if u == v:
        return running_max

    # Lift both together until they meet.
    for k in range(LOG - 1, -1, -1):
        if up[k, u] != up[k, v]:
            wu = mx[k, u]
            wv = mx[k, v]
            if wu > running_max:
                running_max = wu
            if wv > running_max:
                running_max = wv
            u = int(up[k, u])
            v = int(up[k, v])

    # Final step: u and v are now distinct children of the LCA.
    wu = mx[0, u]
    wv = mx[0, v]
    if wu > running_max:
        running_max = wu
    if wv > running_max:
        running_max = wv
    return running_max


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_alpha_nearness(coords: np.ndarray, root: int = 0) -> np.ndarray:
    """Compute the full ``(n, n)`` alpha-nearness matrix.

    Args:
        coords: ``(n, 2)`` Euclidean coordinates.
        root: Index of the 1-tree root node. Defaults to ``0`` to match
            rl4co's depot convention. The choice of root is arbitrary
            for alpha-nearness — only the rows / columns corresponding
            to the root change relative to a different rooting.

    Returns:
        Float64 ``(n, n)`` array. ``alpha[i, j] = alpha[j, i]``;
        ``alpha[i, i] = 0``; ``alpha[i, j] >= 0`` for all ``i, j``.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    n = coords.shape[0]

    if n <= 2:
        # Degenerate: the MST over the single non-root node is empty, so
        # any pair "contains" the only edge and alpha is 0.
        return np.zeros((n, n), dtype=np.float64)

    if not (0 <= root < n):
        raise ValueError(f"root must satisfy 0 <= root < n={n}, got {root}")

    D = _pairwise_euclidean(coords)

    # Build the MST over the FULL graph (n nodes, root included) and
    # pre-compute the binary-lifting tables.
    _, _, _, up, mx, depth = _mst_with_edges(coords, root)

    # NextCost(root): the second-cheapest edge from root to any other
    # node. LKH's ``Minimum1TreeCost`` picks the leaf whose second-
    # nearest-neighbor edge is longest; this is the edge that becomes
    # the second root edge of the 1-tree. ``alpha(root, j)`` is then
    # ``max(0, d(root, j) - NextCost(root))`` for every other j.
    root_row = D[root].copy()
    root_row[root] = np.inf
    sorted_dists = np.sort(root_row)
    next_cost_root = float(sorted_dists[1])  # valid because n >= 3

    # Build the alpha matrix. For non-root pairs, query the max-on-path
    # in the full-graph MST (which may pass through root).
    alpha = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if i == root or j == root:
                a = max(0.0, D[i, j] - next_cost_root)
            else:
                path_max = _max_on_path(i, j, up, mx, depth)
                a = max(0.0, float(D[i, j] - path_max))
            alpha[i, j] = a
            alpha[j, i] = a
    return alpha