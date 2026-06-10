"""Build a :class:`TensorDictDataset` from a :class:`DatasetSpec`."""
from __future__ import annotations

import torch
from rl4co.data.dataset import TensorDictDataset

from nrp.data.dataset import DatasetSpec
from nrp.envs.factory import build_env


def build_dataset_from_spec(spec: DatasetSpec) -> TensorDictDataset:
    """Build a :class:`TensorDictDataset` for evaluation from a :class:`DatasetSpec`.

    The env is constructed via :func:`build_env` and reset with the
    requested batch size to materialise the instances on the fly.

    Args:
        spec: Dataset spec describing the env, batch size, and seed.

    Returns:
        A :class:`TensorDictDataset` wrapping the freshly generated instances.
    """
    env = build_env(
        spec.env_name,
        generator_params=spec.generator_params,
        dataset_params=spec.dataset_params,
    )
    torch.manual_seed(spec.seed)
    td = env.reset(batch_size=[spec.num_instances])
    return TensorDictDataset(td)
