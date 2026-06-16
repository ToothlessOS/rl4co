"""Held-Karp 1-tree utility for Euclidean TSP.

This module provides a dense vectorized Prim's algorithm for building a
minimum-spanning-tree (MST) on a batch of complete distance matrices, the
Held-Karp 1-tree lower bound (MST over non-root nodes plus the two cheapest
root edges), and helpers for comparing TSP tours against the 1-tree via
edge overlap.

All routines operate on PyTorch tensors end-to-end and run on whichever
device the inputs live on (CPU or CUDA). Edge canonicalization
(``(min, max)`` per pair) is used everywhere so direction is irrelevant.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "held_karp_one_tree_lower_bound",
    "tour_edges",
    "edge_overlap",
]


def _pairwise_euclidean(locs: Tensor) -> Tensor:
    """Euclidean pairwise distance matrix.

    Args:
        locs: Tensor of shape ``[B, n, 2]``.

    Returns:
        Tensor of shape ``[B, n, n]`` with zeros on the diagonal.
    """
    return torch.cdist(locs, locs, p=2.0)


def _dense_mst(D: Tensor) -> Tuple[Tensor, Tensor]:
    """Batched dense Prim's algorithm.

    Args:
        D: Tensor of shape ``[B, m, m]`` — symmetric non-negative distance
            matrices with zero diagonals.

    Returns:
        mst_cost: Tensor of shape ``[B]`` — total cost of the MST.
        mst_edges: Tensor of shape ``[B, m - 1, 2]`` — the ``m - 1`` MST
            edges as sorted ``(u, v)`` pairs (orig-node ids).
    """
    B, m, _ = D.shape
    device = D.device
    dtype = D.dtype

    # Seed: sub-node 0 is in the tree.
    in_tree = torch.zeros(B, m, dtype=torch.bool, device=device)
    in_tree[:, 0] = True
    # Cheapest edge from each sub-node into the current tree.
    # `min_cost[:, 0]` is the diagonal (0), which is masked out via `in_tree`.
    min_cost = D[:, 0, :].clone()                  # [B, m]
    min_from = torch.zeros(B, m, dtype=torch.long, device=device)
    mst_cost = torch.zeros(B, device=device, dtype=dtype)

    # Tie-break: subtract a tiny per-index offset so `argmin` is reproducible
    # when two outside nodes have equal cheapest edge. 1e-7 is below float32
    # ULP for typical cost magnitudes.
    tie_off = (torch.arange(m, device=device, dtype=dtype) * 1e-7).view(1, m)

    edge_buf = []
    for _ in range(m - 1):
        masked = torch.where(in_tree, torch.full_like(min_cost, float("inf")), min_cost - tie_off)
        best_flat = masked.view(B, -1).argmin(dim=1)           # [B]
        best = best_flat % m                                    # [B]

        # Accumulate the true cost (pre-mask value).
        mst_cost = mst_cost + min_cost.gather(1, best.unsqueeze(1)).squeeze(1)

        # Record this MST edge as the sorted pair (best, parent).
        parent = min_from.gather(1, best.unsqueeze(1)).squeeze(1)  # [B]
        e_min = torch.minimum(best, parent)
        e_max = torch.maximum(best, parent)
        edge_buf.append(torch.stack([e_min, e_max], dim=-1))        # [B, 2]

        in_tree.scatter_(1, best.unsqueeze(1), True)

        # Relax: update min_cost/min_from using the row of D for the new node.
        # Gather row `best[b]` from each batch -> shape [B, m].
        arange = torch.arange(B, device=device)
        new_d = D[arange, best, :]
        better = new_d < min_cost
        min_cost = torch.where(better, new_d, min_cost)
        min_from = torch.where(better, best.unsqueeze(1).expand_as(min_from), min_from)

    mst_edges = torch.stack(edge_buf, dim=1) if edge_buf else D.new_zeros(B, 0, 2, dtype=torch.long)
    return mst_cost, mst_edges


def held_karp_one_tree_lower_bound(
    locs: Tensor, root: int = 0
) -> Tuple[Tensor, Tensor]:
    """Held-Karp 1-tree lower bound for batched Euclidean TSP.

    The 1-tree for an instance with ``n`` nodes rooted at ``root`` is:

      1. An MST over the induced sub-graph on nodes ``{0, ..., n-1} \\ {root}``.
      2. Plus the two cheapest edges from ``root`` into that MST.

    For Euclidean TSP the 1-tree cost is a lower bound on the optimal tour
    length; the gap between a heuristic and this bound measures how far the
    heuristic sits above the optimum.

    Args:
        locs: Tensor of shape ``[B, n, 2]`` — Euclidean coordinates on any
            device (will run on that device).
        root: Root node index. Default ``0`` matches rl4co's depot convention.
            Must satisfy ``0 <= root < n``.

    Returns:
        bound: Tensor of shape ``[B]`` — HK 1-tree lower bound per instance.
        edges: Tensor of shape ``[B, n, 2]`` — the ``n`` edges of the 1-tree,
            each as a sorted ``(min_node, max_node)`` pair. Useful for
            edge-overlap comparisons with a tour.
    """
    if locs.dim() != 3 or locs.shape[-1] != 2:
        raise ValueError(f"locs must have shape [B, n, 2], got {tuple(locs.shape)}")
    B, n, _ = locs.shape
    if n < 2:
        raise ValueError(f"HK 1-tree requires n >= 2, got n={n}")
    if not (0 <= root < n):
        raise ValueError(f"root must satisfy 0 <= root < n={n}, got {root}")

    D = _pairwise_euclidean(locs)                           # [B, n, n]

    # Induced sub-distance matrix on the non-root nodes.
    if root == 0:
        D_sub = D[:, 1:, 1:].contiguous()                  # [B, n-1, n-1]
        sub_to_orig = torch.arange(1, n, device=locs.device)
    else:
        idx = [i for i in range(n) if i != root]
        sub_to_orig = torch.tensor(idx, device=locs.device, dtype=torch.long)
        D_sub = D[:, idx, :][:, :, idx].contiguous()
    m = n - 1

    # MST over the non-root nodes (sub-graph ids 0..m-1, orig ids sub_to_orig).
    mst_cost, mst_edges_sub = _dense_mst(D_sub)             # [B], [B, m-1, 2]

    # Translate MST edges back to original-node ids (already sorted per-edge).
    mst_edges = sub_to_orig[mst_edges_sub]                  # [B, m-1, 2]

    # Two cheapest edges from root to the rest (excluding self-loop).
    root_row = D[:, root, :].clone()                        # [B, n]
    root_row[:, root] = float("inf")
    k = min(2, n - 1)
    top2 = torch.topk(root_row, k=k, largest=False, dim=1)
    hk_bound = mst_cost + top2.values.sum(dim=1)             # [B]

    # Root edges: (root, target1), (root, target2), canonicalized.
    root_targets = top2.indices                             # [B, k]
    root_edges = torch.stack(
        [
            torch.full_like(root_targets, root),
            root_targets,
        ],
        dim=-1,
    )                                                      # [B, k, 2]
    root_edges, _ = torch.sort(root_edges, dim=-1)

    # Stack the n total edges: (m-1) from MST + k=2 from root.
    edges = torch.cat([mst_edges, root_edges], dim=1)       # [B, n, 2]
    edges, _ = torch.sort(edges, dim=-1)                    # canonicalize
    return hk_bound, edges


def tour_edges(actions: Tensor) -> Tensor:
    """Convert a batch of TSP tours to sorted-edge form.

    A POMO ``actions[b]`` is a permutation of ``{0, 1, ..., n-1}`` that visits
    every node and (via the closed cycle implied by ``get_reward``) returns
    to node 0. The closed tour therefore has ``n`` edges:
    ``(a[i], a[(i+1) % n])`` for ``i in [0, n)``.

    Args:
        actions: Tensor of shape ``[B, n]`` with int64 node indices.

    Returns:
        Tensor of shape ``[B, n, 2]`` — each row is a sorted
        ``(min_node, max_node)`` pair.
    """
    a = actions.to(torch.long)
    a_next = torch.roll(a, shifts=-1, dims=-1)
    e_min = torch.minimum(a, a_next)
    e_max = torch.maximum(a, a_next)
    return torch.stack([e_min, e_max], dim=-1)


def edge_overlap(edges_a: Tensor, edges_b: Tensor) -> Tensor:
    """Fraction of edges shared between two edge sets per instance.

    Both inputs are expected to be in sorted ``(min, max)`` form, and both
    contain exactly ``n`` edges per instance (no self-loops). The overlap is
    ``|A ∩ B| / n`` per batch element, returned as a float tensor in
    ``[0, 1]``.

    Args:
        edges_a: Tensor of shape ``[B, n, 2]`` (sorted-edge format).
        edges_b: Tensor of shape ``[B, n, 2]`` (sorted-edge format).

    Returns:
        Tensor of shape ``[B]`` — overlap ratios in ``[0, 1]``.
    """
    if edges_a.shape != edges_b.shape:
        raise ValueError(
            f"edge sets must share shape, got {tuple(edges_a.shape)} vs "
            f"{tuple(edges_b.shape)}"
        )
    if edges_a.dim() != 3 or edges_a.shape[-1] != 2:
        raise ValueError(
            f"edges must have shape [B, n, 2], got {tuple(edges_a.shape)}"
        )
    B, n, _ = edges_a.shape
    device = edges_a.device

    # Canonicalize per-edge (min, max) so direction is irrelevant even if the
    # caller passes unsorted pairs.
    a_lo = torch.minimum(edges_a[..., 0], edges_a[..., 1]).to(torch.long)
    a_hi = torch.maximum(edges_a[..., 0], edges_a[..., 1]).to(torch.long)
    b_lo = torch.minimum(edges_b[..., 0], edges_b[..., 1]).to(torch.long)
    b_hi = torch.maximum(edges_b[..., 0], edges_b[..., 1]).to(torch.long)

    # Pack each (u, v) into a single int64 key: u * n + v. Safe for n < 2^31.
    a_keys = a_lo * n + a_hi
    b_keys = b_lo * n + b_hi

    # B is small (e.g. 64); a per-batch loop with `torch.unique` + `torch.isin`
    # is fast enough at the evaluation sizes we care about.
    out = torch.empty(B, device=device, dtype=torch.float32)
    for b in range(B):
        b_set = torch.unique(b_keys[b])
        out[b] = torch.isin(a_keys[b], b_set).sum().to(torch.float32) / n
    return out
