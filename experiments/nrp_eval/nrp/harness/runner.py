"""Loop over a list of (solver, env, size) tuples. Used by `nrp sweep`."""
from __future__ import annotations

from collections.abc import Sequence
from itertools import product

from omegaconf import DictConfig, OmegaConf

from nrp.harness.train import train_and_evaluate
from nrp.utils.reproducibility import seed_everything


def run_sweep(
    base_cfg: DictConfig,
    solver_names: Sequence[str],
    env_names: Sequence[str],
    num_locs: Sequence[int],
    seeds: Sequence[int] = (42,),
) -> list[dict]:
    """Cartesian product of (solver, env, num_loc, seed). Runs each combo.

    Returns a list of result dicts (one per combo).
    """
    results = []
    for solver_name, env_name, num_loc, seed in product(
        solver_names, env_names, num_locs, seeds
    ):
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.solver_name = solver_name
        if "env" not in cfg:
            cfg.env = OmegaConf.create({})
        cfg.env.name = env_name
        if "generator_params" not in cfg.env:
            cfg.env.generator_params = OmegaConf.create({})
        cfg.env.generator_params.num_loc = num_loc
        cfg.seed = seed
        cfg.run_id = f"{solver_name}-{env_name}-{num_loc}-seed{seed}"
        seed_everything(seed)
        result = train_and_evaluate(cfg)
        results.append({"cfg": OmegaConf.to_container(cfg, resolve=True), "result": result})
    return results
