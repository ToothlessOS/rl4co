"""Tests for :class:`nrp.solvers.RLSolver`."""
from __future__ import annotations

import pytest
import torch
from rl4co.envs import CVRPEnv, TSPEnv
from rl4co.models import POMO, AttentionModel

from nrp.solvers import RLSolver, SolverRegistry


@pytest.fixture
def tsp_env():
    return TSPEnv(generator_params={"num_loc": 20})


@pytest.fixture
def cvrp_env():
    return CVRPEnv(generator_params={"num_loc": 20})


def test_rl_solver_registration():
    assert "pomo" in SolverRegistry.available()
    assert "am" in SolverRegistry.available()
    assert SolverRegistry.supports("pomo", "tsp")
    assert SolverRegistry.supports("pomo", "cvrp")


def test_rl_solver_from_policy_returns_valid_actions(tsp_env):
    """RLSolver.from_policy wraps a fresh (untrained) POMO policy and runs greedy."""
    model = POMO(tsp_env, num_augment=0)  # untrained, no augmentation
    solver = RLSolver.from_policy(tsp_env, model.policy, decode_type="greedy")
    td = tsp_env.reset(batch_size=[4])
    out = solver.solve(td)
    assert "actions" in out.keys()
    assert "reward" in out.keys()
    assert out["actions"].shape[0] == 4
    assert out["actions"].shape[1] == 20
    # env.get_reward returns shape (B,); we accept either (B,) or (B, 1)
    assert out["reward"].shape in {(4,), (4, 1)}
    assert torch.isfinite(out["reward"]).all()


def test_rl_solver_from_checkpoint_with_fake_ckpt(tmp_path, tsp_env):
    """Save a POMO checkpoint (Lightning format), then load via RLSolver.from_checkpoint."""
    from rl4co.models import POMO
    from rl4co.utils.trainer import RL4COTrainer

    model = POMO(
        tsp_env,
        num_augment=0,
        batch_size=2,
        train_data_size=4,
        val_data_size=4,
        test_data_size=4,
    )
    # Save as a proper Lightning checkpoint
    ckpt_path = tmp_path / "fake.ckpt"
    trainer = RL4COTrainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(str(ckpt_path))

    solver = RLSolver.from_checkpoint(
        tsp_env, str(ckpt_path), model_name="pomo", decode_type="greedy"
    )
    assert solver.model_name == "pomo"
    td = tsp_env.reset(batch_size=[2])
    out = solver.solve(td)
    assert out["actions"].shape == (2, 20)


def test_rl_solver_warmup(tsp_env):
    """warmup() runs without raising."""
    model = POMO(tsp_env, num_augment=0)
    solver = RLSolver.from_policy(tsp_env, model.policy, decode_type="greedy")
    td = tsp_env.reset(batch_size=[4])
    solver.warmup(td)  # should not raise


def test_rl_solver_clone_to_device(tsp_env):
    """to(device) moves the policy."""
    model = POMO(tsp_env, num_augment=0)
    solver = RLSolver.from_policy(tsp_env, model.policy, decode_type="greedy")
    solver.to("cpu")
    assert solver.device.type == "cpu"


def test_rl_solver_from_policy_with_cvrp(cvrp_env):
    """RLSolver wraps a POMO policy for CVRP and returns valid actions."""
    model = POMO(cvrp_env, num_augment=0)
    solver = RLSolver.from_policy(cvrp_env, model.policy, decode_type="greedy")
    td = cvrp_env.reset(batch_size=[2])
    out = solver.solve(td)
    # CVRP returns one action per step including depot returns, so the
    # exact length is variable; just check we got *something* finite.
    assert out["actions"].shape[0] == 2
    assert torch.isfinite(out["reward"]).all()


def test_rl_solver_with_attention_model(tsp_env):
    """RLSolver wraps a non-POMO zoo model (AttentionModel)."""
    model = AttentionModel(tsp_env)
    solver = RLSolver.from_policy(
        tsp_env, model.policy, model_name="am", decode_type="greedy"
    )
    td = tsp_env.reset(batch_size=[2])
    out = solver.solve(td)
    assert out["actions"].shape[0] == 2
    assert out["actions"].shape[1] == 20
