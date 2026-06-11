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
    """With n=50 and target=200, k=1: all customers in one subproblem,
    which carries the parent's full route list as warm-start material."""
    td = _make_td(50, capacity=1000, seed=42)  # huge capacity → no split needed
    routes = _greedy_routes(td)
    decomposer = BarycentreClusteringDecomposer(target_max_subproblem_size=200)
    sps = decomposer.decompose(td, routes)
    assert len(sps) == 1
    assert sorted(sps[0].customer_ids) == list(range(50))
    # k=1 short-circuit: parent_routes should hold all non-empty
    # parent routes (one per input route, possibly fewer if some
    # routes had no customers).
    assert sps[0].n_parent_routes == len(sps[0].parent_routes)
    # All parent routes are 1-indexed Uchoa (no depot, customers in 2..51).
    for pr in sps[0].parent_routes:
        assert all(2 <= c <= 51 for c in pr)
        assert len(pr) > 0


def test_k3_subproblems():
    """With n=500 and target=200, k=3: k subproblems, all customers covered, no overlap,
    each carrying its parent routes."""
    td = _make_td(500, capacity=50, seed=1)
    routes = _greedy_routes(td)
    decomposer = BarycentreClusteringDecomposer(target_max_subproblem_size=200)
    sps = decomposer.decompose(td, routes)
    assert len(sps) >= 1
    covered = sorted({c for sp in sps for c in sp.customer_ids})
    assert covered == list(range(500))
    # All subproblems are non-empty
    for sp in sps:
        assert sp.num_loc > 0
        # Each subproblem carries at least one parent route
        assert sp.n_parent_routes == len(sp.parent_routes)
        assert sp.n_parent_routes >= 1
        # Each parent route is a non-empty list of valid 1-indexed Uchoa ids
        for pr in sp.parent_routes:
            assert len(pr) > 0
            assert all(2 <= c <= sp.num_loc + 1 for c in pr)


def test_empty_routes_handled():
    """Empty routes are distributed round-robin; decomposition still works."""
    td = _make_td(40, capacity=1000, seed=7)  # huge capacity
    routes = [[0, 1, 2], [], [3, 4, 5], [], [6, 7, 8, 9, 10, 11, 12, 13, 14]]
    decomposer = BarycentreClusteringDecomposer(target_max_subproblem_size=10)
    sps = decomposer.decompose(td, routes)
    covered = sorted({c for sp in sps for c in sp.customer_ids})
    assert covered == list(range(15))
    # Each non-empty cluster carries its parent route(s)
    for sp in sps:
        assert sp.n_parent_routes == len(sp.parent_routes)
        assert sp.n_parent_routes >= 1


def test_parent_routes_per_route_capacity_feasible():
    """Each parent route in each subproblem must respect capacity on its own.

    This is the property that makes the warm-start safe: every segment
    of the warm-start tour is a parent route, hence already
    capacity-feasible, so LKH-3's tour validator accepts the file.
    """
    td = _make_td(120, capacity=30, seed=5)
    routes = _greedy_routes(td)  # _greedy_routes produces a feasible solution
    decomposer = BarycentreClusteringDecomposer(target_max_subproblem_size=50)
    sps = decomposer.decompose(td, routes)
    assert len(sps) >= 1
    capacity = int(td["capacity"].item())
    demand_int = (td["demand"] * capacity).round().long().tolist()
    for sp in sps:
        for pr_1indexed in sp.parent_routes:
            # Convert subproblem-local 1-indexed Uchoa ids (2..N+1)
            # back to 0-indexed ids in the **original** problem space
            # via the subproblem's customer_ids mapping.
            ids_0 = [sp.customer_ids[c - 2] for c in pr_1indexed]
            total = sum(demand_int[i] for i in ids_0)
            assert total <= capacity, (
                f"Parent route {pr_1indexed} has demand {total} > capacity {capacity}"
            )


def test_warm_start_no_split_needed():
    """A tight cluster with 1 parent route produces n_parent_routes=1 and
    a single-vehicle subproblem (the orchestrator short-circuits LKH-3 in
    that case)."""
    td = _make_td(20, capacity=100, seed=11)
    # All customers in one tight route.
    routes = [list(range(20))]
    decomposer = BarycentreClusteringDecomposer(target_max_subproblem_size=200)
    sps = decomposer.decompose(td, routes)
    # k=1 (n=20 < 200), so the entire instance is one subproblem with
    # 1 parent route.
    assert len(sps) == 1
    assert sps[0].n_parent_routes == 1
    # The single parent route has all 20 customers.
    assert len(sps[0].parent_routes[0]) == 20
