"""Tests for dataset / benchmark stubs."""
from __future__ import annotations

import pytest

from nrp.data.benchmarks import load_tsplib
from nrp.data.dataset import DatasetSpec, build_eval_dataset
from nrp.envs.factory import build_env


def test_synthetic_dataset_shape():
    spec = DatasetSpec(
        env_name="tsp",
        num_instances=8,
        generator_params={"num_loc": 20},
        seed=42,
    )
    env = build_env(spec.env_name, generator_params=spec.generator_params)
    td = build_eval_dataset(env, spec)
    assert td.batch_size[0] == 8
    assert "locs" in td.keys()
    assert td["locs"].shape == (8, 20, 2)


def test_tsplib_loader_stub_raises():
    with pytest.raises(NotImplementedError, match="stage 2"):
        load_tsplib("dummy.tsp")
