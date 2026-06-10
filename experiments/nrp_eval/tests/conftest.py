"""Shared pytest fixtures for the nrp_eval test suite."""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict


@pytest.fixture
def tsp_env():
    """A TSPEnv with 20 locations."""
    from rl4co.envs import TSPEnv

    return TSPEnv(generator_params={"num_loc": 20})


@pytest.fixture
def random_td_tsp(tsp_env):
    """A small (batch=4) batch of random TSP instances."""
    return tsp_env.reset(batch_size=[4])


@pytest.fixture
def tmp_results_dir(tmp_path):
    """A temporary ``results/`` directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


# Re-export TensorDict for tests that need it via the conftest.
__all__ = ["TensorDict", "torch"]
