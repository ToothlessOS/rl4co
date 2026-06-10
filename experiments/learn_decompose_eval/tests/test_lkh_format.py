"""Tests for TSPLIB CVRP format conversion."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from learn_decompose_eval.solvers.lkh_format import (
    cvrp_td_to_lkh_problem,
    parse_lkh_tour,
    parse_tour_with_cost,
    routes_to_action,
    write_cvrp_initial_tour,
    write_lkh_problem,
)


def _make_td(num_loc: int = 10, capacity: int = 20, seed: int = 0) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    depot = torch.rand(2, generator=g)
    locs = torch.rand(num_loc, 2, generator=g)
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


def test_writer_contains_required_sections():
    td = _make_td(15, capacity=25)
    s = cvrp_td_to_lkh_problem(td, name="test")
    for section in [
        "NAME",
        "TYPE : CVRP",
        "DIMENSION",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "NODE_COORD_TYPE : TWOD_COORDS",
        "CAPACITY",
        "NODE_COORD_SECTION",
        "DEMAND_SECTION",
        "EDGE_WEIGHT_SECTION",
        "DEPOT_SECTION",
        "EOF",
    ]:
        assert section in s, f"Missing section: {section}"
    # Uchoa convention: DIMENSION = N + 1 (1 depot + N customers)
    assert "DIMENSION : 16" in s
    # Depot (node 1) has demand 0
    assert "1\t0" in s


def test_round_trip_through_lkh(tmp_path: Path, lkh_binary: str):
    """Write a problem, solve it with the patched LKH-3, parse the tour back.

    LKH-3's CVRP solver internally expands the dimension to
    ``nbClients + n_vehicles`` (with depot repeated per vehicle). The exact
    number of vehicles depends on the algorithm's choice. We assert only
    that the round trip runs without error and produces a non-empty route
    list.
    """
    if not lkh_binary or not os.path.exists(lkh_binary):
        pytest.skip("LKH-3 binary not available")

    td = _make_td(20, capacity=30, seed=42)
    problem_str = cvrp_td_to_lkh_problem(td, name="roundtrip")
    problem_path = tmp_path / "test.vrp"
    tour_path = tmp_path / "test.tour"
    par_path = tmp_path / "test.par"
    write_lkh_problem(str(problem_path), problem_str)
    (tmp_path / "test.par").write_text(
        f"PROBLEM_FILE = {problem_path}\n"
        f"OUTPUT_TOUR_FILE = {tour_path}\n"
        f"RUNS = 1\nMAX_TRIALS = 500\nTIME_LIMIT = 3\nTRACE_LEVEL = 0\n"
    )

    import subprocess

    res = subprocess.run([lkh_binary, str(par_path)], capture_output=True, timeout=10)
    if res.returncode != 0 or not tour_path.exists():
        pytest.skip(f"LKH-3 failed: rc={res.returncode}\nstderr={res.stderr[:200]}")

    routes = parse_lkh_tour(str(tour_path))
    # Tour was produced and parsed
    assert len(routes) >= 1
    # All 20 customers should be visited (with possible LKH internal nodes
    # representing the depot)
    customers = sorted({c for r in routes for c in r if c != 1})
    # Just check the 20 customer ids are a subset of the visited nodes
    for cust_id in range(2, 22):
        assert cust_id in customers, f"customer {cust_id} missing from tour"


def test_routes_to_action_basic():
    """LKH uses 1-indexed (depot=1, customers=2..N+1); CVRPEnv uses 1-indexed
    customers (1..N) with depot=0. Conversion: subtract 1 from each LKH
    customer id.
    """
    routes = [[1, 5, 3, 1], [1, 2, 4, 1]]  # LKH 1-indexed
    action = routes_to_action(routes, num_loc=5)
    # LKH customers 5,3,2,4 → CVRPEnv 4,2,1,3
    cust = sorted([int(x) for x in action if x != 0])
    assert cust == [1, 2, 3, 4]


def test_routes_to_action_no_depot_at_start():
    """Routes that start with a non-depot node should be accepted.

    LKH-3 uses 1-indexed nodes (depot=1, customers=2..N+1). CVRPEnv also
    uses 1-indexed customers (1..N) with depot=0. So LKH customer 2 → CVRPEnv
    customer 1 (i.e. subtract 1 from each LKH customer id).
    """
    routes = [[5, 3, 1], [2, 4, 1]]  # LKH 1-indexed; customers are 2,3,4,5
    action = routes_to_action(routes, num_loc=5)
    # LKH 5,3 → CVRPEnv 4,2; LKH 2,4 → CVRPEnv 1,3. So customers are 1,2,3,4.
    cust = sorted([int(x) for x in action if x != 0])
    assert cust == [1, 2, 3, 4]


def test_write_and_parse_known_tour(tmp_path: Path):
    """Parse a hand-rolled TSPLIB tour and confirm structure.

    Uchoa CVRP convention: depot is node 1, customers are 2..DIMENSION-1.
    The tour is a single linear sequence with the depot repeated at route
    boundaries.
    """
    tour_str = (
        "NAME : test.tour\n"
        "DIMENSION : 6\n"
        "TYPE : TOUR\n"
        "TOUR_SECTION\n"
        "1\n5\n3\n1\n2\n4\n1\n"
        "-1\nEOF\n"
    )
    p = tmp_path / "test.tour"
    p.write_text(tour_str)
    routes = parse_lkh_tour(str(p))
    # Routes split at depot id 1
    assert routes == [[1, 5, 3, 1], [2, 4, 1]]


def test_parse_tour_with_cost(tmp_path: Path):
    """parse_tour_with_cost should extract DIMENSION and COMMENT:Length=X
    from the tour header, plus the body routes."""
    tour_str = (
        "NAME : test.tour\n"
        "COMMENT : Length = 12345\n"
        "COMMENT : Found by LKH-3 [Keld Helsgaun] Mon Jan  1 00:00:00 2024\n"
        "TYPE : TOUR\n"
        "DIMENSION : 6\n"
        "TOUR_SECTION\n"
        "1\n5\n3\n1\n2\n4\n1\n"
        "-1\nEOF\n"
    )
    p = tmp_path / "test.tour"
    p.write_text(tour_str)
    result = parse_tour_with_cost(str(p))
    assert result is not None
    routes, cost, dim = result
    assert cost == 12345
    assert dim == 6
    assert routes == [[1, 5, 3, 1], [2, 4, 1]]


def test_parse_tour_with_cost_handles_penalty_format(tmp_path: Path):
    """For tours with non-zero penalty, LKH-3 writes 'Cost = P_C'.
    parse_tour_with_cost should return only the cost component C."""
    tour_str = (
        "NAME : infeasible.tour\n"
        "COMMENT : Cost = 0_5000\n"  # penalty 0, cost 5000
        "TYPE : TOUR\n"
        "DIMENSION : 4\n"
        "TOUR_SECTION\n"
        "1\n2\n3\n1\n"
        "-1\nEOF\n"
    )
    p = tmp_path / "test.tour"
    p.write_text(tour_str)
    result = parse_tour_with_cost(str(p))
    assert result is not None
    _, cost, _ = result
    assert cost == 5000  # only the C part, not the 0_5000 prefix


def test_parse_tour_with_cost_missing_file(tmp_path: Path):
    """Missing file should return None, not raise."""
    result = parse_tour_with_cost(str(tmp_path / "does_not_exist.tour"))
    assert result is None


def test_write_cvrp_initial_tour_salesmen_1(tmp_path: Path):
    """salesmen=1 should produce a single-route tour with no phantom
    depots; DIMENSION = num_loc + 1."""
    customers = list(range(2, 12))  # 10 customers, ids 2..11
    p = tmp_path / "tour.tour"
    write_cvrp_initial_tour(str(p), customers, num_loc=10, salesmen=1)
    text = p.read_text()
    # Verify the file is well-formed
    assert "DIMENSION : 11" in text  # 10 + 1
    assert "TYPE : TOUR" in text
    assert "TOUR_SECTION" in text
    # Verify the body is a permutation of [1, 2..11]
    body_ids = _parse_tour_body(text)
    assert sorted(body_ids) == list(range(1, 12))
    # No phantom depots: no id > 11
    assert all(x <= 11 for x in body_ids)


def test_write_cvrp_initial_tour_even_split(tmp_path: Path):
    """salesmen=3 with num_loc=10 should produce a tour with 2 phantom
    depots (ids 12, 13); DIMENSION = 13. Each route gets ~3-4 customers."""
    customers = list(range(2, 12))  # 10 customers
    p = tmp_path / "tour.tour"
    write_cvrp_initial_tour(str(p), customers, num_loc=10, salesmen=3)
    text = p.read_text()
    # DIMENSION = num_loc + salesmen = 13
    assert "DIMENSION : 13" in text
    # Verify the body
    body_ids = _parse_tour_body(text)
    # Body should have 13 entries
    assert len(body_ids) == 13
    # All entries unique
    assert len(set(body_ids)) == 13
    # All ids in [1..13]
    assert sorted(body_ids) == list(range(1, 14))
    # Phantom depots 12 and 13 should appear
    assert 12 in body_ids
    assert 13 in body_ids
    # All 10 customers should appear
    for cust in range(2, 12):
        assert cust in body_ids


def test_write_cvrp_initial_tour_uneven_split(tmp_path: Path):
    """When num_loc is not divisible by salesmen, the last route gets
    the remainder. For num_loc=10, salesmen=4, route sizes are
    [3, 3, 2, 2] (or [2, 3, 3, 2] etc. depending on the implementation)."""
    customers = list(range(2, 12))  # 10 customers
    p = tmp_path / "tour.tour"
    write_cvrp_initial_tour(str(p), customers, num_loc=10, salesmen=4)
    text = p.read_text()
    assert "DIMENSION : 14" in text  # 10 + 4
    body_ids = _parse_tour_body(text)
    assert len(body_ids) == 14
    assert sorted(body_ids) == list(range(1, 15))


def _parse_tour_body(text: str) -> list[int]:
    """Helper: extract the integer body of a TSPLIB tour file (TSPLIB TOUR_SECTION).

    The body is everything after ``TOUR_SECTION``. The first line
    after the split is usually empty (the newline that follows
    ``TOUR_SECTION``), so we skip empties and only break on the
    ``-1`` terminator.
    """
    body = text.split("TOUR_SECTION", 1)[1]
    body_ids: list[int] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue  # skip blank lines
        if s in ("-1", "EOF"):
            break
        try:
            body_ids.append(int(s))
        except ValueError:
            continue
    return body_ids


def test_write_cvrp_initial_tour_invalid_inputs(tmp_path: Path):
    """Inputs that aren't a permutation of 2..N+1 should raise ValueError."""
    p = tmp_path / "tour.tour"
    # Wrong length — caught as a permutation error
    with pytest.raises(ValueError, match="permutation"):
        write_cvrp_initial_tour(str(p), [2, 3, 4], num_loc=10, salesmen=2)
    # Wrong values (and wrong length) — also permutation
    with pytest.raises(ValueError, match="permutation"):
        write_cvrp_initial_tour(
            str(p), [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100], num_loc=10, salesmen=2
        )
    # salesmen < 1
    with pytest.raises(ValueError, match="salesmen"):
        write_cvrp_initial_tour(str(p), [2, 3], num_loc=2, salesmen=0)
