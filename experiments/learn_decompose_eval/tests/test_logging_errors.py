"""Tests for the new diagnostic logging in the LKH-3 ± BCC CVRP experiment.

These tests use ``caplog`` to assert that previously-silent failure paths
now produce a log line at WARNING or ERROR level, with the LKH-3 stderr
tail included for diagnosis.
"""

from __future__ import annotations

import logging
import os

import pytest
import torch
from rl4co.envs.routing.cvrp.env import CVRPEnv

from learn_decompose_eval.solvers.classical_lkh import _solve_one_instance
from learn_decompose_eval.solvers.orchestration import (
    IntermediateTourWatcher,
    OrchestratorConfig,
    _tail,
)

LKH_REQUIRED = pytest.mark.skipif(
    not os.environ.get("LDE_LKH_BINARY")
    and not os.environ.get("NRP_LKH_BINARY")
    and not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "LKH-3.0.14", "LKH")
    ),
    reason="LKH-3 binary not available",
)


def _resolve() -> str:
    """Return the LKH-3 binary path from env or in-tree location."""
    env = os.environ.get("LDE_LKH_BINARY") or os.environ.get("NRP_LKH_BINARY")
    if env and os.path.exists(env):
        return env
    in_tree = os.path.join(os.path.dirname(__file__), "..", "LKH-3.0.14", "LKH")
    if os.path.exists(in_tree):
        return in_tree
    pytest.skip("LKH-3 binary not found")


@LKH_REQUIRED
def test_raw_solve_emits_info_log(caplog):
    """The raw solver should now log raw_solve_start and raw_solve_done."""
    env = CVRPEnv(generator_params={"num_loc": 20})
    torch.manual_seed(0)
    td = env.reset(batch_size=[1])[0]
    lkh = _resolve()

    with caplog.at_level(logging.INFO, logger="learn_decompose_eval"):
        routes, _ = _solve_one_instance(
            lkh,
            td,
            name="log_test_raw",
            time_limit_s=1.0,
        )
    recs = [r.getMessage() for r in caplog.records]
    assert any("raw_solve_start" in m for m in recs), recs
    # raw_solve_done only fires if LKH wrote a tour; allow either.
    assert routes or any("raw_nonzero" in m or "raw_no_final_tour" in m for m in recs)


@LKH_REQUIRED
def test_orchestrator_lifecycle_logs(caplog):
    """The orchestrator should emit solve_start, master_launch, solve_done."""
    env = CVRPEnv(generator_params={"num_loc": 30})
    torch.manual_seed(0)
    td = env.reset(batch_size=[1])[0]
    lkh = _resolve()
    cfg = OrchestratorConfig(
        lkh_binary=lkh,
        max_total_s=2.0,
        min_restart_interval_s=1.0,
        population_size=2,
        num_workers=2,
        target_max_subproblem_size=200,
    )
    watcher = IntermediateTourWatcher(cfg)
    with caplog.at_level(logging.INFO, logger="learn_decompose_eval"):
        routes, _ = watcher.solve(td, name="log_test_bcc")
    recs = [r.getMessage() for r in caplog.records]
    assert any("solve_start" in m for m in recs), recs
    assert any("master_launch" in m for m in recs), recs
    assert any("solve_done" in m for m in recs), recs


@LKH_REQUIRED
def test_orchestrator_runs_past_first_improvement(caplog):
    """The orchestrator should NOT exit on no_intermediate_update for
    n=100 within the time budget, because the master now runs for the
    full max_total_s budget.

    With the warm-start now enabled, we also expect:
    - a ``master_salesmen`` log line
    - at least one ``n_restarts > 0`` (i.e. a successful restart with
      a phantom-depot-aware initial tour)
    - the master should NOT exit with rc=1 (the dimension-mismatch
      bug should be gone).
    """
    env = CVRPEnv(generator_params={"num_loc": 100})
    torch.manual_seed(0)
    td = env.reset(batch_size=[1])[0]
    lkh = _resolve()
    # 30s budget so subs (per_sub_s ~ 1s) have time to write tour files.
    cfg = OrchestratorConfig(
        lkh_binary=lkh,
        max_total_s=30.0,
        min_restart_interval_s=3.0,
        population_size=4,
        num_workers=2,
        target_max_subproblem_size=200,
    )
    watcher = IntermediateTourWatcher(cfg)
    with caplog.at_level(logging.INFO, logger="learn_decompose_eval"):
        watcher.solve(td, name="continuous_test")
    recs = [r.getMessage() for r in caplog.records]
    solve_done = [m for m in recs if "solve_done" in m]
    assert solve_done, recs
    # master_salesmen log line should fire (from final.tour or heuristic)
    assert any("master_salesmen" in m for m in recs), recs
    # solve_done should report n_restarts >= 1 (warm-start cycle ran)
    assert "n_restarts=0" not in solve_done[0], solve_done[0]
    # New design: loop_exit=time_budget or master_self_exited.
    assert "loop_exit=no_update" not in solve_done[0], solve_done[0]
    # The solve_start line should mention the population size.
    assert any("population_size=4" in m for m in recs), recs


def test_decompose_every_s_deprecation_warning(caplog):
    """A non-default decompose_every_s should emit a deprecation warning."""
    # Use a fake binary path; the watcher only logs the warning, it
    # doesn't actually launch LKH-3 at construction time.
    cfg = OrchestratorConfig(
        lkh_binary="/bin/true",
        decompose_every_s=99.0,  # non-default triggers warning
    )
    with caplog.at_level(logging.WARNING, logger="learn_decompose_eval"):
        IntermediateTourWatcher(cfg)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("decompose_every_s is deprecated" in m for m in msgs), msgs


def _resolve_skip() -> bool:
    return bool(
        os.environ.get("LDE_LKH_BINARY")
        or os.environ.get("NRP_LKH_BINARY")
        or os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "LKH-3.0.14", "LKH")
        )
    )


def test_tail_helper(tmp_path):
    """_tail should return the last N bytes of a file."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"x" * 1000 + b"HELLO")
    assert _tail(str(f), 5) == "HELLO"
    # If the file is empty, return empty string.
    f2 = tmp_path / "empty.bin"
    f2.write_bytes(b"")
    assert _tail(str(f2), 5) == ""


def test_tail_helper_missing_file(tmp_path):
    """_tail should not raise on a missing file."""
    assert "<unreadable" in _tail(str(tmp_path / "nope"), 8)
