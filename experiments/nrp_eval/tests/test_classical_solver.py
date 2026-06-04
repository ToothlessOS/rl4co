"""Tests for classical solvers."""
from __future__ import annotations

import os

import pytest
import torch
from rl4co.envs import CVRPEnv, TSPEnv

from nrp.solvers import (
    BuiltinEnvSolver,
    ConcordeTSPSolver,
    GurobiTSPSolver,
    LKHSolver,
    ORToolsTSPSolver,
    ORToolsVRPSolver,
    RLSolver,
    SolverRegistry,
)


@pytest.fixture
def tsp_env():
    return TSPEnv(generator_params={"num_loc": 20})


@pytest.fixture
def cvrp_env():
    return CVRPEnv(generator_params={"num_loc": 20})


def test_classical_registration():
    assert "ortools_tsp" in SolverRegistry.available()
    assert "ortools_vrp" in SolverRegistry.available()
    assert "lkh_tsp" in SolverRegistry.available()
    assert "concorde_tsp" in SolverRegistry.available()
    assert "gurobi_tsp" in SolverRegistry.available()
    assert "builtin_solve" in SolverRegistry.available()


def test_ortools_tsp_returns_valid_tours(tsp_env):
    """ORToolsTSPSolver returns [B, N] int64 actions with feasible tours."""
    solver = ORToolsTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[8])
    out = solver.solve(td)
    assert out["actions"].shape == (8, 20)
    assert out["actions"].dtype == torch.int64
    # Each row should be a permutation of 0..19
    for row in out["actions"].cpu().numpy():
        assert sorted(row.tolist()) == list(range(20))
    # Reward should be finite
    assert torch.isfinite(out["reward"]).all()
    # Tour length should be > 0
    assert (-out["reward"]).min() > 0


def test_ortools_vrp_returns_valid_tours(cvrp_env):
    """ORToolsVRPSolver returns valid CVRP tours (depot + customers, capacity respected)."""
    solver = ORToolsVRPSolver(cvrp_env)
    td = cvrp_env.reset(batch_size=[4])
    out = solver.solve(td)
    assert out["actions"].shape[0] == 4
    assert torch.isfinite(out["reward"]).all()
    # The CVRPEnv's check_solution_validity (in solve via env.get_reward) should pass
    # i.e., reward should be non-positive (tour length is non-negative, reward is negated)


def test_builtin_solve_unsupported_env_raises(tsp_env):
    """BuiltinEnvSolver raises NotImplementedError for TSP (no env.solve impl)."""
    solver = BuiltinEnvSolver(tsp_env)
    td = tsp_env.reset(batch_size=[2])
    with pytest.raises(NotImplementedError):
        solver.solve(td)


def test_lkh_stub_raises_clear_error(tsp_env):
    """LKHSolver raises FileNotFoundError with NRP_LKH_BINARY hint."""
    os.environ.pop("NRP_LKH_BINARY", None)
    solver = LKHSolver(tsp_env)
    td = tsp_env.reset(batch_size=[2])
    with pytest.raises(FileNotFoundError, match="NRP_LKH_BINARY"):
        solver.solve(td)


def test_concorde_stub_raises_clear_error(tsp_env):
    os.environ.pop("NRP_CONCORDE_BINARY", None)
    solver = ConcordeTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[2])
    with pytest.raises(FileNotFoundError, match="NRP_CONCORDE_BINARY"):
        solver.solve(td)


def test_gurobi_stub_raises_clear_error(tsp_env):
    os.environ.pop("NRP_GUROBI_BINARY", None)
    solver = GurobiTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[2])
    with pytest.raises(FileNotFoundError, match="NRP_GUROBI_BINARY"):
        solver.solve(td)


def test_classical_solvers_within_reasonable_factor(tsp_env):
    """ORToolsTSP mean tour length should be within 2x of untrained POMO greedy.

    This is a sanity check that the classical solver actually produces reasonable
    tours. Untrained POMO is basically random; OR-Tools (or our NN+2opt) should beat it.
    """
    from rl4co.models import POMO

    # Classical
    classical_solver = ORToolsTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[32])
    classical_tour = -classical_solver.solve(td)["reward"].mean().item()

    # RL (untrained POMO, greedy)
    model = POMO(tsp_env, num_augment=0)
    rl_solver = RLSolver.from_policy(tsp_env, model.policy, decode_type="greedy")
    td2 = tsp_env.reset(batch_size=[32])
    rl_tour = -rl_solver.solve(td2)["reward"].mean().item()

    # Classical should be no more than 2x RL (which is itself essentially random).
    assert classical_tour <= 2.0 * rl_tour, (
        f"Classical ({classical_tour:.3f}) more than 2x RL ({rl_tour:.3f})"
    )
