"""Dataset spec + builders for evaluation datasets.

The :class:`DatasetSpec` is a plain dataclass describing a synthetic
dataset. For Stage 1 we generate on the fly via ``env.reset``; later stages
will add TSPLIB / CVRPLIB loaders behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tensordict import TensorDict


@dataclass
class DatasetSpec:
    """Specification for an evaluation dataset.

    Attributes:
        env_name: Registered RL4CO environment name (e.g. ``"tsp"``).
        num_instances: Number of instances to materialise.
        generator_params: Keyword arguments forwarded to the env's generator.
        dataset_params: Extra kwargs forwarded to the env's ``__init__``
            (e.g. ``data_dir``, ``test_file``).
        seed: Random seed used to generate the data.
        name: Short tag used when naming cached files.
    """

    env_name: str
    num_instances: int = 1000
    generator_params: dict = field(default_factory=dict)
    dataset_params: dict = field(default_factory=dict)
    seed: int = 1234
    name: str = "test"


def build_eval_dataset(env, spec: DatasetSpec, phase: str = "test") -> TensorDict:
    """Build an eval dataset by calling ``env.reset``.

    For stage 1, the function always generates a fresh batch on the fly. If
    later stages want to load from disk, the env's ``dataset_params`` will
    determine that behaviour.

    Args:
        env: An instantiated RL4CO env.
        spec: A :class:`DatasetSpec` describing what to generate.
        phase: Unused in stage 1; reserved for stage 2 to dispatch on
            ``train/val/test`` file resolution.

    Returns:
        A :class:`TensorDict` with ``batch_size=[num_instances]``.
    """
    del phase  # reserved for stage 2
    torch.manual_seed(spec.seed)
    td = env.reset(batch_size=[spec.num_instances])
    return td
