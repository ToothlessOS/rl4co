"""Standalone correctness tests for `utils/HK.py`.

Run with:

    .venv/bin/python test_hk.py

These tests do NOT depend on the POMO checkpoint or TSPEnv; they exercise the
HK module directly. The brute-force check enumerates all permutations for
``n in {4, 5, 6, 7}`` and asserts the HK 1-tree bound never exceeds the
true optimal tour length.
"""

from __future__ import annotations

import itertools

import torch

from utils.HK import edge_overlap, held_karp_one_tree_lower_bound, tour_edges


def _closed_tour_length(locs: torch.Tensor, tour: tuple) -> float:
    """Sum of L2 distances around a closed tour, given node indices."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = locs[tour[i]]
        b = locs[tour[(i + 1) % n]]
        total += float(torch.dist(a, b))
    return total


def _brute_force_optimal(locs: torch.Tensor) -> float:
    """Exact optimal TSP tour length by enumerating permutations (n <= 7)."""
    n = locs.shape[0]
    # Fix node 0 as the start to remove the n rotations of each cycle.
    rest = list(range(1, n))
    best = float("inf")
    for perm in itertools.permutations(rest):
        tour = (0,) + perm
        best = min(best, _closed_tour_length(locs, tour))
    return best


def test_tour_edges() -> None:
    """Tour 0 -> 1 -> 2 -> 0 yields edges {(0,1),(1,2),(0,2)}."""
    actions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    edges = tour_edges(actions)
    expected = torch.tensor([[[0, 1], [1, 2], [0, 2]]], dtype=torch.long)
    assert torch.equal(edges, expected), (edges, expected)
    print("test_tour_edges passed")


def test_overlap_self_and_disjoint() -> None:
    """Self overlap = 1.0; disjoint overlap = 0.0; direction-insensitive."""
    # Tour 0->1->2->3->4->0
    a = torch.tensor(
        [[[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]], dtype=torch.long
    )
    ov_self = edge_overlap(a, a)
    assert torch.allclose(ov_self, torch.tensor([1.0])), ov_self

    # Five edges on the same 5-node graph that share nothing with `a`:
    # C(5,2)=10 edges total minus a's 5 = {(0,2),(0,3),(1,3),(1,4),(2,4)}.
    b = torch.tensor(
        [[[0, 2], [0, 3], [1, 3], [1, 4], [2, 4]]], dtype=torch.long
    )
    ov_disj = edge_overlap(a, b)
    assert torch.allclose(ov_disj, torch.tensor([0.0])), ov_disj

    # Partial overlap: 2 of 5 edges shared.
    c = torch.tensor(
        [[[0, 1], [2, 4], [1, 3], [1, 4], [2, 3]]], dtype=torch.long
    )
    ov_partial = edge_overlap(a, c)
    # Shared edges: (0,1) and (2,3) -> 2/5.
    assert torch.allclose(ov_partial, torch.tensor([0.4])), ov_partial

    # Direction-insensitive: swapping u/v should not change the overlap.
    a_swap = a.flip(dims=(-1,))
    ov_swap = edge_overlap(a_swap, a)
    assert torch.allclose(ov_swap, torch.tensor([1.0])), ov_swap

    print("test_overlap_self_and_disjoint passed")


def test_hk_lower_bound() -> None:
    """HK bound <= brute-force optimal for random small instances."""
    torch.manual_seed(0)
    for n in (4, 5, 6, 7):
        for trial in range(3):
            locs = torch.rand(2, n, 2)               # B=2
            bound, edges = held_karp_one_tree_lower_bound(locs)
            # Sanity: bound is finite and has the expected shape.
            assert bound.shape == (2,), bound.shape
            assert edges.shape == (2, n, 2), edges.shape
            for b in range(2):
                opt = _brute_force_optimal(locs[b])
                assert bound[b].item() <= opt + 1e-4, (
                    f"n={n} trial={trial} b={b}: HK={bound[b].item():.6f} "
                    f"> opt={opt:.6f}"
                )
                # Bound should be strictly positive for non-degenerate locs.
                assert bound[b].item() > 0.0
    print("test_hk_lower_bound passed")


def test_hk_edges_count_and_no_self_loops() -> None:
    """Each 1-tree returns exactly `n` sorted edges with no self-loops."""
    torch.manual_seed(1)
    locs = torch.rand(4, 12, 2)
    bound, edges = held_karp_one_tree_lower_bound(locs)
    assert edges.shape == (4, 12, 2)
    # No self-loops: u != v on every edge.
    assert (edges[..., 0] != edges[..., 1]).all()
    # Edges are sorted per-row.
    assert (edges[..., 0] <= edges[..., 1]).all()
    # Cost recomputed from edges equals reported bound (within float32 noise).
    pairwise = torch.cdist(locs, locs, p=2.0)
    u = edges[..., 0]
    v = edges[..., 1]
    batch_idx = torch.arange(4).unsqueeze(1).expand(-1, 12)
    edge_costs = pairwise[batch_idx, u, v]
    recomputed = edge_costs.sum(dim=-1)
    assert torch.allclose(bound.cpu(), recomputed.cpu(), atol=1e-4), (
        bound.cpu() - recomputed.cpu()
    )
    print("test_hk_edges_count_and_no_self_loops passed")


def test_hk_general_root() -> None:
    """Non-zero root: bound is finite and ≥ minimal root-edge pair.

    Different root choices yield different 1-trees, so the bounds are not
    required to match. We just check the function runs and the result is
    positive and at least as large as the two shortest root edges (the MST
    adds non-negative weight on top).
    """
    torch.manual_seed(2)
    locs = torch.rand(2, 10, 2)

    bound_0, edges_0 = held_karp_one_tree_lower_bound(locs, root=0)
    bound_3, edges_3 = held_karp_one_tree_lower_bound(locs, root=3)
    assert bound_0.shape == (2,)
    assert bound_3.shape == (2,)
    assert edges_0.shape == (2, 10, 2)
    assert edges_3.shape == (2, 10, 2)
    # Every edge is sorted and has u != v.
    assert (edges_0[..., 0] <= edges_0[..., 1]).all()
    assert (edges_3[..., 0] <= edges_3[..., 1]).all()
    assert (edges_0[..., 0] != edges_0[..., 1]).all()
    assert (edges_3[..., 0] != edges_3[..., 1]).all()
    # Bounds are positive.
    assert (bound_0 > 0).all() and (bound_3 > 0).all()
    # No root=3 edge should have u=3 (root) -- wait, root edges ARE (3, x).
    # What we DO know: no edge should have both u=v.
    # Bounds should be at least the cost of the two shortest edges from
    # the respective root node (MST contributes ≥ 0).
    pairwise = torch.cdist(locs, locs, p=2.0)
    for b in range(2):
        row0 = pairwise[b, 0].clone(); row0[0] = float("inf")
        row3 = pairwise[b, 3].clone(); row3[3] = float("inf")
        two_cheapest_0 = torch.topk(row0, k=2, largest=False).values.sum()
        two_cheapest_3 = torch.topk(row3, k=2, largest=False).values.sum()
        assert bound_0[b].item() >= two_cheapest_0.item() - 1e-4, (
            f"b={b} root=0: bound={bound_0[b].item()} < 2cheapest={two_cheapest_0.item()}"
        )
        assert bound_3[b].item() >= two_cheapest_3.item() - 1e-4, (
            f"b={b} root=3: bound={bound_3[b].item()} < 2cheapest={two_cheapest_3.item()}"
        )
    print("test_hk_general_root passed")


if __name__ == "__main__":
    test_tour_edges()
    test_overlap_self_and_disjoint()
    test_hk_lower_bound()
    test_hk_edges_count_and_no_self_loops()
    test_hk_general_root()
    print("All tests passed.")
