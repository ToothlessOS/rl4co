"""Thin wrapper around :func:`rl4co.envs.get_env` with metadata defaults."""
from __future__ import annotations

from rl4co.envs import RL4COEnvBase, get_env

from nrp.envs.env_registry import get_env_info


def build_env(
    name: str,
    generator_params: dict | None = None,
    dataset_params: dict | None = None,
    **env_kwargs,
) -> RL4COEnvBase:
    """Build an RL4CO env by name with optional generator/dataset overrides.

    Args:
        name: Registered env name (must be in :data:`ENV_INFO`).
        generator_params: Forwarded to the env's generator. If ``None`` we
            use the env's ``default_generator_params`` from the registry.
        dataset_params: Forwarded to the env constructor (e.g. ``data_dir``,
            ``train_file``, ``val_file``, ``test_file``).
        **env_kwargs: Any other keyword arguments forwarded to ``get_env``.

    Returns:
        The instantiated :class:`RL4COEnvBase`.
    """
    info = get_env_info(name)  # raises KeyError with a helpful message
    if generator_params is None:
        generator_params = info.get("default_generator_params", {})
    return get_env(
        name,
        generator_params=generator_params,
        **(dataset_params or {}),
        **env_kwargs,
    )
