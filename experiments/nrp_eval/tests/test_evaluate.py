"""Tests for nrp.harness.evaluate."""
import math

import pytest
from rl4co.envs import TSPEnv
from rl4co.models import POMO

from nrp.harness.evaluate import EvaluationResult, evaluate
from nrp.solvers import ORToolsTSPSolver, RLSolver


@pytest.fixture
def tsp_env():
    return TSPEnv(generator_params={"num_loc": 20})


def test_evaluate_classical_greedy(tsp_env):
    """End-to-end: evaluate ORToolsTSPSolver on a small batch."""
    solver = ORToolsTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[8])
    result = evaluate(solver, tsp_env, td, method="greedy", warmup=True)
    assert isinstance(result, EvaluationResult)
    assert len(result.per_instance) == 8
    assert math.isfinite(result.summary["mean"]), result.summary
    assert result.summary["feasible_ratio"] == 1.0
    assert result.summary["method"] == "greedy"
    assert result.summary["solver_name"] == "ortools_tsp"


def test_evaluate_rl_greedy(tsp_env):
    """End-to-end: evaluate RLSolver (untrained POMO) on a small batch."""
    model = POMO(tsp_env, num_augment=0)
    solver = RLSolver.from_policy(tsp_env, model.policy, decode_type="greedy")
    td = tsp_env.reset(batch_size=[8])
    result = evaluate(solver, tsp_env, td, method="greedy", warmup=True)
    assert isinstance(result, EvaluationResult)
    assert len(result.per_instance) == 8
    assert math.isfinite(result.summary["mean"]), result.summary


def test_evaluate_with_optima(tsp_env):
    """When optima are provided, gap_to_opt_pct is computed."""
    solver = ORToolsTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[4])
    # Use the actual tour lengths as fake optima to test the wiring
    out = solver.solve(td)
    optima = (-out["reward"]).flatten().tolist()
    result = evaluate(
        solver, tsp_env, td, method="greedy", warmup=False, optima=optima
    )
    assert "mean_gap_to_opt_pct" in result.summary
    assert result.summary["mean_gap_to_opt_pct"] == pytest.approx(0.0, abs=1e-3)


def test_evaluate_unknown_method_raises(tsp_env):
    solver = ORToolsTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[2])
    with pytest.raises(ValueError, match="Unknown eval method"):
        evaluate(solver, tsp_env, td, method="nonsense")


def test_evaluate_saves_pickle(tsp_env, tmp_path):
    solver = ORToolsTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[4])
    evaluate(
        solver,
        tsp_env,
        td,
        method="greedy",
        warmup=False,
        save_dir=tmp_path,
        run_id="test_run",
    )
    assert (tmp_path / "test_run.pkl").exists()
    # Verify it can be loaded back
    from nrp.utils.pkl import load_versioned

    loaded = load_versioned(tmp_path / "test_run.pkl", expected_schema="evaluation_result")
    assert loaded.summary["solver_name"] == "ortools_tsp"


def test_evaluate_metadata(tsp_env):
    solver = ORToolsTSPSolver(tsp_env)
    td = tsp_env.reset(batch_size=[4])
    result = evaluate(solver, tsp_env, td, method="greedy", warmup=False)
    assert result.metadata["solver_type"] == "ORToolsTSPSolver"
    assert result.metadata["env_name"] == "tsp"
    assert result.metadata["num_loc"] == 20
    assert result.metadata["method"] == "greedy"
