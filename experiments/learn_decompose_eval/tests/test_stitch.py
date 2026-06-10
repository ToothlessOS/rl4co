"""Unit tests for the orchestrator's stitch logic.

The stitch combines per-subproblem LKH-3 routes (1-indexed in each
subproblem's own Uchoa CVRP convention) into a single master Uchoa
tour suitable for LKH-3's INITIAL_TOUR_FILE.

Format invariant: the output must be a flat permutation of
``[1..num_loc+1]`` (1 depot + ``num_loc`` customers) — no route
separators, no closing depot, no phantom depots.  LKH-3's ``ReadTour``
rejects entries that duplicate an id or fall outside ``[1..DIMENSION]``.
"""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from learn_decompose_eval.solvers.decomposition import Subproblem
from learn_decompose_eval.solvers.orchestration import _stitch_routes


def _dummy_td(num_loc: int = 10) -> TensorDict:
    """A minimal TensorDict matching CVRPEnv's expected schema."""
    return TensorDict(
        {
            "depot": torch.zeros(2, dtype=torch.float32),
            "locs": torch.zeros((num_loc, 2), dtype=torch.float32),
            "demand": torch.zeros(num_loc, dtype=torch.float32),
            "capacity": torch.tensor([1.0], dtype=torch.float32),
        },
        batch_size=[],
    )


def _sub(
    customer_ids: list[int], capacity: int = 50, depot_xy=None
) -> Subproblem:
    """Build a Subproblem for testing the stitch."""
    n = len(customer_ids)
    if depot_xy is None:
        depot_xy = np.array([0.0, 0.0], dtype=np.float32)
    return Subproblem(
        customer_ids=customer_ids,
        xy=np.zeros((n, 2), dtype=np.float32),
        demand=np.zeros(n, dtype=np.int64),
        capacity=capacity,
        depot_xy=depot_xy,
    )


def test_stitch_single_route():
    """A single subproblem with one route should produce a flat master
    tour starting with the depot followed by all customers in order.
    """
    sub = _sub(customer_ids=[3, 5, 7])
    # Subproblem-local Uchoa: depot=1, customers=2..4
    #   master 3 → sub id 2
    #   master 5 → sub id 3
    #   master 7 → sub id 4
    # Expected master Uchoa ids: 5, 7, 9
    routes = [[1, 2, 3, 4, 1]]  # depot, three customers, depot
    stitched = _stitch_routes([(sub, routes)], _dummy_td(num_loc=10))
    # Format: depot (1) followed by customers in route order.
    assert stitched == [1, 5, 7, 9]


def test_stitch_multi_route_in_subproblem():
    """A subproblem with two vehicles (two routes) should produce a flat
    sequence of all customers (the depot appears only once at the start).
    """
    sub = _sub(customer_ids=[3, 5, 7])
    # Two routes in the subproblem: [1, 2, 3, 1] and [1, 4, 1]
    #   first  → master customers 3, 5  → Uchoa 5, 7
    #   second → master customer 7      → Uchoa 9
    routes = [[1, 2, 3, 1], [1, 4, 1]]
    stitched = _stitch_routes([(sub, routes)], _dummy_td(num_loc=10))
    # Format: depot (1) followed by all customers in order, no route
    # markers.
    assert stitched == [1, 5, 7, 9]


def test_stitch_drops_phantom_depots():
    """Phantom depots (ids > subproblem DIMENSION) should be dropped, not
    counted as customers.
    """
    sub = _sub(customer_ids=[3, 5, 7])
    # sub_dim = 4; ids 5, 6, 7 are phantom depots (one per vehicle in
    # a 3-vehicle solution)
    routes = [[1, 2, 5, 3, 6, 4, 7, 1]]
    stitched = _stitch_routes([(sub, routes)], _dummy_td(num_loc=10))
    # 1 → drop (subproblem depot), 2 → 5, 5 → drop (phantom),
    # 3 → 7, 6 → drop, 4 → 9, 7 → drop, 1 → drop
    # Output: [1, 5, 7, 9]
    assert stitched == [1, 5, 7, 9]


def test_stitch_multi_subproblem():
    """Two subproblems with disjoint customer sets are stitched
    back-to-back, producing a single flat sequence.
    """
    sub_a = _sub(customer_ids=[2, 4])  # master 2, 4 → Uchoa 4, 6
    sub_b = _sub(customer_ids=[7, 9])  # master 7, 9 → Uchoa 9, 11
    routes_a = [[1, 2, 3, 1]]  # sub_a: depot, c1=2, c2=3, depot
    routes_b = [[1, 2, 3, 1]]  # sub_b: depot, c1=2, c2=3, depot
    stitched = _stitch_routes(
        [(sub_a, routes_a), (sub_b, routes_b)], _dummy_td(num_loc=12)
    )
    # Format: depot (1) followed by all customers from sub_a then sub_b.
    assert stitched == [1, 4, 6, 9, 11]


def test_stitch_output_is_permutation():
    """The output must be a permutation of [1..num_loc+1] with no
    duplicate ids (so LKH-3's ReadTour doesn't reject it).
    """
    sub_a = _sub(customer_ids=[0, 1, 2, 3, 4])
    sub_b = _sub(customer_ids=[5, 6, 7, 8, 9])
    routes_a = [[1, 2, 3, 4, 5, 6, 1]]  # depot, 5 customers, depot
    routes_b = [[1, 2, 3, 4, 5, 6, 1]]  # depot, 5 customers, depot
    stitched = _stitch_routes(
        [(sub_a, routes_a), (sub_b, routes_b)], _dummy_td(num_loc=10)
    )
    # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # (depot + master customers 0..9 in Uchoa 2..11)
    assert sorted(stitched) == list(range(1, 12))
    assert len(stitched) == len(set(stitched))  # no duplicates
    assert len(stitched) == 11  # 1 depot + 10 customers


def test_stitch_empty_input():
    """Empty input returns None (not a degenerate tour)."""
    assert _stitch_routes([], _dummy_td()) is None
    sub = _sub(customer_ids=[1, 2])
    assert _stitch_routes([(sub, [])], _dummy_td()) is None
    sub_empty = _sub(customer_ids=[])
    assert _stitch_routes([(sub_empty, [[1, 1]])], _dummy_td()) is None


def test_stitch_skips_subproblem_with_empty_routes():
    """A subproblem with empty routes should be skipped, not crash."""
    sub_ok = _sub(customer_ids=[3, 5])
    sub_empty_routes = _sub(customer_ids=[7, 9])
    stitched = _stitch_routes(
        [(sub_ok, [[1, 2, 3, 1]]), (sub_empty_routes, [])],
        _dummy_td(num_loc=10),
    )
    # Only sub_ok contributes: 2→5, 3→7 → [1, 5, 7]
    assert stitched == [1, 5, 7]


def test_stitch_skips_subproblem_with_no_customers():
    """A degenerate subproblem with empty customer_ids should be skipped."""
    sub = _sub(customer_ids=[])
    # Even a non-empty route is dropped because sub_dim=1 means all
    # ids > 1 are phantom depots.
    stitched = _stitch_routes([(sub, [[1, 1]])], _dummy_td())
    assert stitched is None


def test_stitch_id_mapping_for_non_contiguous_customers():
    """Customer ids in the subproblem are not necessarily contiguous
    0..n-1; the mapping should still produce the correct master Uchoa ids.
    """
    sub = _sub(customer_ids=[0, 1, 5, 7, 99])  # non-contiguous 0-indexed
    # Subproblem-local Uchoa: 2, 3, 4, 5, 6
    # Expected master Uchoa: 0+2, 1+2, 5+2, 7+2, 99+2 = 2, 3, 7, 9, 101
    routes = [[1, 2, 3, 4, 5, 6, 1]]
    stitched = _stitch_routes([(sub, routes)], _dummy_td(num_loc=100))
    assert stitched == [1, 2, 3, 7, 9, 101]
