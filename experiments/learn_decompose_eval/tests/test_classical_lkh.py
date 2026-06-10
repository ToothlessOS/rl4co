"""Tests for the LKH-3 CVRP solver classes (raw and BCC-decomposed)."""
from __future__ import annotations

import os
import time

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
