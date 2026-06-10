"""Tests for BarycentreClusteringDecomposer."""
from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from learn_decompose_eval.solvers.decomposition import BarycentreClusteringDecomposer


def _make_td(num_loc: int, capacity: int = 30, seed: int = 0) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    depot = torch.rand(2, generator=g)
    locs = torch.rand(num_loc, 2, generator=g)
    # Integer demand in [1, 9] to mimic Kool
    demand = (torch.randint(1, 10, (num_loc,), generator=g) + 1).float()
    return TensorDict(
        {
            "depot": depot,
            "locs": locs,
            "demand": demand / capacity,
            "capacity": torch.tensor([capacity], dtype=torch.float32),
        },
        batch_size=[],
    )


def _greedy_routes(td: TensorDict) -> list[list[int]]:
    """Trivial feasible solution: round-robin assignment to ~sqrt(N) routes,
    starting a new route whenever the current one overflows. The exact
    assignment is unimportant for testing decomposition — we just need
    *some* feasible master solution to feed in.
    """
    n = td["locs"].shape[0]
    capacity = int(td["capacity"].item())
    demand_int = (td["demand"] * capacity).round().long().tolist()
    # Greedy: assign each customer to the route with the most free space that
    # still fits. If no route fits, open a new one.
    routes: list[list[int]] = [[]]
    loads: list[int] = [0]
    for i in range(n):
        d = demand_int[i]
        placed = False
        for v in range(len(routes)):
            if loads[v] + d <= capacity:
                routes[v].append(i)
                loads[v] += d
                placed = True
                break
        if not placed:
            routes.append([i])
            loads.append(d)
    return [r for r in routes if r]


def test_k1_subproblem():
    """With n=50 and target=200, k=1: all customers in one subproblem
    (capacity-respecting split disabled so we observe the k=1 path cleanly)."""
    td = _make_td(50, capacity=1000, seed=42)  # huge capacity → no split needed
    routes = _greedy_routes(td)
    decomposer = BarycentreClusteringDecomposer(
        target_max_subproblem_size=200, enforce_capacity=False
    )
    sps = decomposer.decompose(td, routes)
    assert len(sps) == 1
    assert sorted(sps[0].customer_ids) == list(range(50))


def test_k3_subproblems():
    """With n=500 and target=200, k=3: 3 subproblems, all customers covered, no overlap."""
    td = _make_td(500, capacity=50, seed=1)
    routes = _greedy_routes(td)
    decomposer = BarycentreClusteringDecomposer(target_max_subproblem_size=200)
    sps = decomposer.decompose(td, routes)
    assert len(sps) >= 1  # may be split further for capacity
    covered = sorted({c for sp in sps for c in sp.customer_ids})
    assert covered == list(range(500))
    # All subproblems are non-empty
    for sp in sps:
        assert sp.num_loc > 0
        # Capacity respected
        assert int(sp.demand.sum()) <= int(sp.capacity)


def test_empty_routes_handled():
    """Empty routes are distributed round-robin; decomposition still works."""
    td = _make_td(40, capacity=1000, seed=7)  # huge capacity
    routes = [[0, 1, 2], [], [3, 4, 5], [], [6, 7, 8, 9, 10, 11, 12, 13, 14]]
    decomposer = BarycentreClusteringDecomposer(
        target_max_subproblem_size=10, enforce_capacity=False
    )
    sps = decomposer.decompose(td, routes)
    covered = sorted({c for sp in sps for c in sp.customer_ids})
    assert covered == list(range(15))


def test_capacity_split_oversized_demand():
    """A cluster whose total demand exceeds capacity is greedily split."""
    td = _make_td(20, capacity=10, seed=3)
    # All customers have demand 9 (after rounding); capacity 10 fits only 1 each
    td["demand"] = (torch.full((20,), 9.0) / 10.0)
    routes = [list(range(20))]  # one big route
    decomposer = BarycentreClusteringDecomposer(
        target_max_subproblem_size=200, enforce_capacity=True
    )
    sps = decomposer.decompose(td, routes)
    # With k=1 and demand=9, capacity=10, only 1 customer per subproblem
    assert len(sps) == 20
    for sp in sps:
        assert int(sp.demand.sum()) <= 10
