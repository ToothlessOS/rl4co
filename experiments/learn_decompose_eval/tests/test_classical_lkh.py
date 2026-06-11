"""Tests for the LKH-3 CVRP solver classes (raw and BCC-decomposed)."""
from __future__ import annotations

import os
import subprocess as _subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from learn_decompose_eval.solvers.classical_lkh import (
    BarycentreLKH3CVRSolver,
    RawLKH3CVRSolver,
    _resolve_lkh_binary,
)
from learn_decompose_eval.solvers.lkh_format import routes_to_action


LKH_REQUIRED = pytest.mark.skipif(
    not os.environ.get("LDE_LKH_BINARY") and not os.path.exists(
        os.path.join(
            os.path.dirname(__file__), "..", "LKH-3.0.14", "LKH"
        )
    ),
    reason="LKH-3 binary not available",
)


def _build_env(num_loc: int = 20):
    # The cvrp subpackage is a namespace package in rl4co (no __init__.py),
    # so we import via the .env submodule to be safe.
    from rl4co.envs.routing.cvrp.env import CVRPEnv

    return CVRPEnv(generator_params={"num_loc": num_loc})


def _make_td(env, batch_size: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    return env.reset(batch_size=[batch_size])


@LKH_REQUIRED
def test_resolve_lkh_binary(lkh_binary: str):
    binary = _resolve_lkh_binary(lkh_binary)
    assert os.path.exists(binary)


# BCC k2 (n=500) is currently known-fragile: the orchestrator's behaviour
# on very large problems with multiple LKH sub-invocations has race
# conditions that can produce invalid tours. The test is documented but
# the body asserts to give a clear failure if conditions change.


@LKH_REQUIRED
def test_raw_solver_runs(lkh_binary: str):
    env = _build_env(20)
    td = _make_td(env, batch_size=2, seed=42)
    solver = RawLKH3CVRSolver(
        env, binary_path=lkh_binary, max_runtime_s=2.0, num_workers=1
    )
    out = solver.solve_batch(td)
    actions = torch.as_tensor(out, dtype=torch.int64)
    env.check_solution_validity(td, actions)


@LKH_REQUIRED
def test_bcc_solver_runs_k1(lkh_binary: str):
    """With target=200, n=50 → k=1: BCC should run end-to-end and return a valid action."""
    env = _build_env(50)
    td = _make_td(env, batch_size=1, seed=42)
    solver = BarycentreLKH3CVRSolver(
        env,
        binary_path=lkh_binary,
        max_total_s=4.0,
        decompose_every_s=2.0,
        num_workers=2,
        target_max_subproblem_size=200,
    )
    out = solver.solve_batch(td)
    actions = torch.as_tensor(out, dtype=torch.int64)
    env.check_solution_validity(td, actions)


@LKH_REQUIRED
def test_bcc_solver_runs_k2(lkh_binary: str):
    """With n=500 and target=200, k=3: the orchestrator should still finish.

    KNOWN ISSUE: The BCC orchestrator has a race condition for very
    large problems (n>=500) where parallel sub-LKH invocations can
    produce invalid tours. Marked as a known-failing test until the
    orchestrator's race is debugged. The decomposition and format
    helpers themselves are tested separately in test_decomposition and
    test_lkh_format.
    """
    env = _build_env(500)
    td = _make_td(env, batch_size=1, seed=42)
    solver = BarycentreLKH3CVRSolver(
        env,
        binary_path=lkh_binary,
        max_total_s=8.0,
        decompose_every_s=4.0,
        num_workers=2,
        target_max_subproblem_size=200,
    )
    out = solver.solve_batch(td)
    actions = torch.as_tensor(out, dtype=torch.int64)
    env.check_solution_validity(td, actions)


@LKH_REQUIRED
def test_bcc_solver_runs_k1(lkh_binary: str):
    """With target=200, n=50 → k=1: BCC should run end-to-end and return a valid action."""
    env = _build_env(50)
    td = _make_td(env, batch_size=1, seed=42)
    solver = BarycentreLKH3CVRSolver(
        env,
        binary_path=lkh_binary,
        max_total_s=4.0,
        decompose_every_s=2.0,
        num_workers=2,
        target_max_subproblem_size=200,
    )
    out = solver.solve_batch(td)
    assert out.shape[0] == 1
    # Authoritative check
    actions = torch.as_tensor(out, dtype=torch.int64)
    env.check_solution_validity(td, actions)


@LKH_REQUIRED
def test_bcc_solver_runs_k2(lkh_binary: str):
    """With n=500 and target=200, k=3: the orchestrator should still finish."""
    env = _build_env(500)
    td = _make_td(env, batch_size=1, seed=42)
    solver = BarycentreLKH3CVRSolver(
        env,
        binary_path=lkh_binary,
        max_total_s=8.0,
        decompose_every_s=4.0,
        num_workers=2,
        target_max_subproblem_size=200,
    )
    out = solver.solve_batch(td)
    assert out.shape[0] == 1
    # Authoritative check
    actions = torch.as_tensor(out, dtype=torch.int64)
    env.check_solution_validity(td, actions)


# ---------------------------------------------------------------------------
# Intermediate-tour fallback for the raw LKH-3 path
# ---------------------------------------------------------------------------
# The raw solver's watchdog can fire while LKH-3 is still searching.
# The LDE-patched LKH-3 binary writes the current best tour to
# INTERMEDIATE_TOUR_FILE on every improvement. The raw solver should
# fall back to that intermediate tour instead of returning [].
# These tests stub ``subprocess.call`` to exercise the fallback path
# deterministically without actually running LKH-3.
def _write_tour(path: str, body: list[int], cost: float | None = None) -> None:
    """Write a LKH-3 TSPLIB tour file in the format that ``WriteTour`` emits.

    ``body`` is the sequence of node ids inside ``TOUR_SECTION``, terminated
    by ``-1`` and ``EOF``. The depot (id 1) appears at route boundaries to
    mark depot returns.
    """
    dim = max(body) if body else 0
    lines: list[str] = [
        "NAME : test",
        (
            f"COMMENT : Cost = {cost:.4f}"
            if cost is not None
            else "COMMENT : Length = 0"
        ),
        "TYPE : TOUR",
        f"DIMENSION : {dim}",
        "TOUR_SECTION",
    ]
    for nid in body:
        lines.append(str(int(nid)))
    lines.extend(["-1", "EOF"])
    Path(path).write_text("\n".join(lines) + "\n")


def _make_td_raw(num_loc: int = 10, capacity: int = 30, seed: int = 0) -> TensorDict:
    """Build a minimal CVRP TensorDict for ``_solve_one_instance``.

    Mirrors ``test_decomposition._make_td`` but lives here to avoid an
    import into another test module (pytest's collection of sibling
    modules is brittle when conftest setup matters).
    """
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


def _stub_subprocess_call_timeout():
    """Return a mock for ``subprocess.call`` that raises ``TimeoutExpired``
    (simulating the Python watchdog killing LKH-3)."""

    def _fake_call(cmd, timeout=None, stdout=None, stderr=None, **kwargs):
        if hasattr(stdout, "write"):
            stdout.write(b"")
            stdout.flush()
        if hasattr(stderr, "write"):
            stderr.write(b"LKH-3 was running and got killed by watchdog\n")
            stderr.flush()
        raise _subprocess.TimeoutExpired(cmd=cmd, timeout=timeout or 0)

    return _fake_call


def _stub_subprocess_call_rc0(final_tour_body: list[int]):
    """Return a mock for ``subprocess.call`` that pretends LKH-3 self-exited
    with rc=0 and wrote ``final.tour`` to the directory of the .par file."""

    def _fake_call(cmd, timeout=None, stdout=None, stderr=None, **kwargs):
        par_path = cmd[1]
        final_tour = os.path.join(os.path.dirname(par_path), "final.tour")
        _write_tour(final_tour, final_tour_body, cost=42.0)
        if hasattr(stdout, "write"):
            stdout.write(b"")
            stdout.flush()
        if hasattr(stderr, "write"):
            stderr.write(b"")
            stderr.flush()
        return 0

    return _fake_call


def test_intermediate_fallback_on_timeout(tmp_path: Path):
    """Watchdog fires, but LKH-3 already wrote a usable intermediate tour:
    the solver should return the intermediate tour's routes."""
    from learn_decompose_eval.solvers.classical_lkh import _solve_one_instance

    td = _make_td_raw(num_loc=10, seed=0)
    tmp = tempfile.mkdtemp(prefix="lde_raw_test_", dir=tmp_path)
    intermediate_path = os.path.join(tmp, "intermediate.tour")
    # Pre-write a valid intermediate tour (4 routes through 10 customers).
    # LKH-3 customer ids are 2..N+1, depot is 1.
    _write_tour(
        intermediate_path,
        body=[1, 2, 3, 1, 1, 4, 5, 1, 1, 6, 7, 1, 1, 8, 9, 10, 1],
        cost=123.45,
    )
    with mock.patch(
        "learn_decompose_eval.solvers.classical_lkh.subprocess.call",
        new=_stub_subprocess_call_timeout(),
    ):
        routes, elapsed = _solve_one_instance(
            lkh_binary="/bin/true",  # not used; subprocess is stubbed
            td_instance=td,
            name="inst_fb",
            time_limit_s=1.0,
            tmpdir=tmp,
            cleanup_tmp=False,
        )
    # ``parse_lkh_tour`` returns 1-indexed routes with depot at
    # boundaries, so the 4 chunks in the body are returned as-is.
    assert routes == [
        [1, 2, 3, 1],
        [1, 4, 5, 1],
        [1, 6, 7, 1],
        [1, 8, 9, 10, 1],
    ], f"Expected intermediate routes, got {routes}"
    assert elapsed > 0


def test_intermediate_fallback_unparseable(tmp_path: Path):
    """Watchdog fires AND the intermediate file is missing: the solver
    should fall through to the existing ``[]`` behavior."""
    from learn_decompose_eval.solvers.classical_lkh import _solve_one_instance

    td = _make_td_raw(num_loc=10, seed=0)
    tmp = tempfile.mkdtemp(prefix="lde_raw_test_", dir=tmp_path)
    # No intermediate.tour file written at all.
    with mock.patch(
        "learn_decompose_eval.solvers.classical_lkh.subprocess.call",
        new=_stub_subprocess_call_timeout(),
    ):
        routes, elapsed = _solve_one_instance(
            lkh_binary="/bin/true",
            td_instance=td,
            name="inst_nofb",
            time_limit_s=1.0,
            tmpdir=tmp,
            cleanup_tmp=False,
        )
    assert routes == []
    assert elapsed > 0


def test_intermediate_fallback_empty_file(tmp_path: Path):
    """Watchdog fires AND the intermediate file exists but is empty
    (size 0): same as the missing case — return []."""
    from learn_decompose_eval.solvers.classical_lkh import _solve_one_instance

    td = _make_td_raw(num_loc=10, seed=0)
    tmp = tempfile.mkdtemp(prefix="lde_raw_test_", dir=tmp_path)
    intermediate_path = os.path.join(tmp, "intermediate.tour")
    Path(intermediate_path).write_text("")
    with mock.patch(
        "learn_decompose_eval.solvers.classical_lkh.subprocess.call",
        new=_stub_subprocess_call_timeout(),
    ):
        routes, _ = _solve_one_instance(
            lkh_binary="/bin/true",
            td_instance=td,
            name="inst_empty",
            time_limit_s=1.0,
            tmpdir=tmp,
            cleanup_tmp=False,
        )
    assert routes == []


def test_intermediate_fallback_malformed_file(tmp_path: Path, caplog):
    """Watchdog fires AND the intermediate file is unparseable: log
    ``raw_timeout_intermediate_parse_failed`` and return []."""
    from learn_decompose_eval.solvers.classical_lkh import _solve_one_instance

    td = _make_td_raw(num_loc=10, seed=0)
    tmp = tempfile.mkdtemp(prefix="lde_raw_test_", dir=tmp_path)
    intermediate_path = os.path.join(tmp, "intermediate.tour")
    # Garbage content that parse_lkh_tour will reject (no TOUR_SECTION).
    Path(intermediate_path).write_text("this is not a LKH-3 tour file")
    with mock.patch(
        "learn_decompose_eval.solvers.classical_lkh.subprocess.call",
        new=_stub_subprocess_call_timeout(),
    ):
        routes, _ = _solve_one_instance(
            lkh_binary="/bin/true",
            td_instance=td,
            name="inst_malformed",
            time_limit_s=1.0,
            tmpdir=tmp,
            cleanup_tmp=False,
        )
    assert routes == []
    # Confirm the parse-failed log was emitted (so we know the fallback
    # branch was actually entered, not the no-fallback raw_timeout branch).
    assert any(
        "raw_timeout_intermediate_parse_failed" in rec.getMessage()
        for rec in caplog.records
    )


def test_no_intermediate_when_clean_exit(tmp_path: Path):
    """LKH-3 self-exits cleanly with rc=0 and a final tour: the
    intermediate file is ignored and the final tour is returned."""
    from learn_decompose_eval.solvers.classical_lkh import _solve_one_instance

    td = _make_td_raw(num_loc=5, seed=0)
    tmp = tempfile.mkdtemp(prefix="lde_raw_test_", dir=tmp_path)
    intermediate_path = os.path.join(tmp, "intermediate.tour")
    # Intermediate says the search was at a worse cost; should be ignored.
    _write_tour(
        intermediate_path,
        body=[1, 2, 1, 1, 3, 4, 1, 1, 5, 6, 1],
        cost=9999.0,
    )
    # The final tour (the real success path) is written by the stub.
    with mock.patch(
        "learn_decompose_eval.solvers.classical_lkh.subprocess.call",
        new=_stub_subprocess_call_rc0([1, 2, 3, 1, 1, 4, 5, 1, 1, 6, 1]),
    ):
        routes, elapsed = _solve_one_instance(
            lkh_binary="/bin/true",
            td_instance=td,
            name="inst_clean",
            time_limit_s=1.0,
            tmpdir=tmp,
            cleanup_tmp=False,
        )
    # Should match the final.tour body, NOT the intermediate.
    assert routes == [[1, 2, 3, 1], [1, 4, 5, 1], [1, 6, 1]]
    assert elapsed > 0


def test_intermediate_cost_helper(tmp_path: Path):
    """``_intermediate_cost`` should return the COMMENT line from a LKH-3
    tour file, or None if no such line exists."""
    from learn_decompose_eval.solvers.classical_lkh import _intermediate_cost

    p = tmp_path / "tour.tour"
    _write_tour(str(p), [1, 2, 1, 1, 3, 1], cost=12.5)
    out = _intermediate_cost(str(p))
    assert out is not None
    assert "12.5" in out

    p_empty = tmp_path / "empty.tour"
    p_empty.write_text("")
    assert _intermediate_cost(str(p_empty)) is None

    p_no_cost = tmp_path / "no_cost.tour"
    p_no_cost.write_text(
        "NAME : x\nTYPE : TOUR\nDIMENSION : 1\nTOUR_SECTION\n1\n-1\nEOF\n"
    )
    assert _intermediate_cost(str(p_no_cost)) is None

    # LKH-3's WriteTour emits ``COMMENT : Length = <cost>`` when there
    # is no current penalty — which is the format the intermediate
    # tour file uses during a normal CVRP search. Confirm the helper
    # picks that up too (not just ``Cost =``).
    p_length = tmp_path / "length.tour"
    p_length.write_text(
        "NAME : x.1234.5678.tour\n"
        "COMMENT : Length = 1234.5678\n"
        "COMMENT : Found by LKH-3 [Keld Helsgaun] Wed Jun 11 00:00:00 2026\n"
        "TYPE : TOUR\n"
        "DIMENSION : 3\n"
        "TOUR_SECTION\n1\n2\n3\n-1\nEOF\n"
    )
    out_length = _intermediate_cost(str(p_length))
    assert out_length is not None
    assert "1234.5678" in out_length
    assert "Length" in out_length


# ---------------------------------------------------------------------------
# parse_lkh_tour: LKH-3 CVRP "monster tour" mode (phantom depots)
# ---------------------------------------------------------------------------
# LKH-3's CVRP solver writes a single circular walk of length
# num_loc + Salesmen, in which the real depot (id 1) appears exactly once
# and phantom depots (ids num_loc+2..num_loc+Salesmen) appear at each
# route boundary. Without ``num_loc``, the legacy parser sees the real
# depot only once and returns a single "route" of length num_loc+Salesmen
# — which is the bug we hit on CVRP-200. With ``num_loc``, the parser
# splits at phantom depots and returns the correct Salesmen routes.
def _write_cvrp_monster_tour(
    path: str,
    num_loc: int,
    salesmen: int,
    customers_per_route: int | None = None,
) -> None:
    """Write a synthetic LKH-3 CVRP monster tour.

    Customers: 2..num_loc+1 (Uchoa convention).
    Real depot: 1 (appears exactly once, at the start of the circular walk).
    Phantom depots: num_loc+2..num_loc+salesmen (one per route boundary).
    The depot is implicit at every route start/end in the cyclic tour.
    """
    if customers_per_route is None:
        # Distribute customers as evenly as possible.
        base, rem = divmod(num_loc, salesmen)
        per_route = [base + (1 if i < rem else 0) for i in range(salesmen)]
    else:
        per_route = [customers_per_route] * (salesmen - 1) + [
            num_loc - customers_per_route * (salesmen - 1)
        ]
    body: list[int] = [1]  # real depot at the start
    cid = 2
    for r in range(salesmen):
        if r > 0:
            body.append(num_loc + 1 + r)  # phantom depot for this route
        for _ in range(per_route[r]):
            body.append(cid)
            cid += 1
    assert cid == num_loc + 2
    assert len(body) == num_loc + salesmen
    dim = num_loc + salesmen
    lines = [
        "NAME : test_monster",
        "COMMENT : Length = 0",
        "TYPE : TOUR",
        f"DIMENSION : {dim}",
        "TOUR_SECTION",
        *(str(x) for x in body),
        "-1",
        "EOF",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def test_parse_lkh_tour_cvrp_monster_with_num_loc(tmp_path: Path):
    """With ``num_loc``, parse_lkh_tour splits the monster tour at
    phantom-depot boundaries and returns the actual number of CVRP routes.
    """
    from learn_decompose_eval.solvers.lkh_format import parse_lkh_tour

    num_loc, salesmen = 200, 15
    p = tmp_path / "monster.tour"
    _write_cvrp_monster_tour(str(p), num_loc=num_loc, salesmen=salesmen)
    routes = parse_lkh_tour(str(p), depot_id=1, num_loc=num_loc)
    assert len(routes) == salesmen, (
        f"Expected {salesmen} routes, got {len(routes)}: {routes[:3]}..."
    )
    # Each route should be bracketed by the real depot id.
    for r in routes:
        assert r[0] == 1 and r[-1] == 1, f"Route not depot-bracketed: {r}"
        # No phantom depots (ids > num_loc+1) should appear inside.
        assert all(c <= num_loc + 1 for c in r), (
            f"Phantom depot leaked into route: {r}"
        )
    # All customers should be covered exactly once.
    seen = [c for r in routes for c in r if 2 <= c <= num_loc + 1]
    assert sorted(seen) == list(range(2, num_loc + 2))


def test_parse_lkh_tour_cvrp_monster_without_num_loc_legacy(tmp_path: Path):
    """Without ``num_loc`` (legacy TSP-style), the parser sees the real
    depot only once and returns a single 'route' of length num_loc+salesmen.
    This documents the broken legacy behavior that motivated the fix.
    """
    from learn_decompose_eval.solvers.lkh_format import parse_lkh_tour

    num_loc, salesmen = 200, 15
    p = tmp_path / "monster.tour"
    _write_cvrp_monster_tour(str(p), num_loc=num_loc, salesmen=salesmen)
    routes = parse_lkh_tour(str(p), depot_id=1)  # num_loc=None (legacy)
    # Legacy parser collapses the monster tour to a single route, which
    # is the bug we are fixing.
    assert len(routes) == 1
    # Legacy parser sees the real depot only once, so the single "route"
    # contains num_loc + salesmen nodes plus one trailing depot id appended
    # by the legacy close-out logic.
    assert len(routes[0]) == num_loc + salesmen + 1


def test_parse_lkh_tour_legacy_tsp_with_depot_boundaries(tmp_path: Path):
    """Legacy TSP-style tours (no phantom depots) still work: routes are
    split at the real depot's occurrences and each route is depot-bracketed.
    """
    from learn_decompose_eval.solvers.lkh_format import parse_lkh_tour

    _write_tour(
        str(tmp_path / "tsp.tour"),
        body=[1, 2, 3, 1, 1, 4, 5, 1, 1, 6, 7, 1],
    )
    routes = parse_lkh_tour(str(tmp_path / "tsp.tour"), depot_id=1)
    assert routes == [
        [1, 2, 3, 1],
        [1, 4, 5, 1],
        [1, 6, 7, 1],
    ]
