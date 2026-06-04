"""Thin glue around rl4co.tasks.train.train."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nrp.data.dataset import DatasetSpec, build_eval_dataset
from nrp.envs.factory import build_env
from nrp.solvers import SolverRegistry


def train_and_evaluate(cfg: DictConfig):
    """Dispatch on cfg.mode.

    - 'train': call rl4co.tasks.train.train(cfg).
    - 'eval': build solver from cfg, run evaluate().
    - 'train_and_eval': train first, then evaluate the best checkpoint.
    """
    mode = cfg.get("mode", "train")
    if mode == "train":
        from rl4co.tasks.train import train as rl4co_train

        return rl4co_train(cfg)
    if mode == "eval":
        return _eval_only(cfg)
    if mode == "train_and_eval":
        from rl4co.tasks.train import train as rl4co_train

        # Suppress RL4CO's built-in trainer.test(): we do our own eval below.
        # This also avoids the PyTorch weights_only=True UnpicklingError on
        # newer PyTorch when the Lightning checkpoint contains non-tensor state.
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.test = False
        rl4co_train(cfg)
        ckpt_path = _find_best_checkpoint(cfg)
        if ckpt_path is None:
            raise FileNotFoundError(
                "No checkpoint found under paths.output_dir after training."
            )
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.ckpt_path = ckpt_path
        cfg.mode = "eval"
        return _eval_only(cfg)
    raise ValueError(f"Unknown mode '{mode}'. Use 'train', 'eval', or 'train_and_eval'.")


def _eval_only(cfg: DictConfig):
    from nrp.harness.evaluate import evaluate

    env_name = (
        cfg.env.name
        if "name" in cfg.env
        else cfg.get("solver_name", "tsp")
    )
    generator_params = (
        OmegaConf.to_container(cfg.env.generator_params, resolve=True)
        if "generator_params" in cfg.env
        else {}
    )
    env = build_env(env_name, generator_params=generator_params)

    spec = DatasetSpec(
        env_name=env.name,
        num_instances=cfg.evaluate.get("num_instances", 1000),
        generator_params=generator_params,
        seed=cfg.get("seed", 1234),
    )
    td = build_eval_dataset(env, spec)

    solver_name = cfg.get("solver_name", "pomo")
    solver_kwargs = {}
    if "solver" in cfg and solver_name in cfg.solver:
        solver_kwargs = OmegaConf.to_container(cfg.solver[solver_name], resolve=True)

    if cfg.get("ckpt_path"):
        if solver_name in ("pomo", "am", "symnco", "matnet", "ptrnet"):
            # RL solver: use from_checkpoint classmethod
            solver_cls = SolverRegistry.registry[solver_name]
            solver = solver_cls.from_checkpoint(
                env, cfg.ckpt_path, model_name=solver_name, **solver_kwargs
            )
        else:
            solver = SolverRegistry.build(
                solver_name, env, ckpt_path=cfg.ckpt_path, **solver_kwargs
            )
    else:
        if solver_name in (
            "ortools_tsp",
            "ortools_vrp",
            "builtin_solve",
            "lkh_tsp",
            "concorde_tsp",
            "gurobi_tsp",
        ):
            solver = SolverRegistry.build(solver_name, env, **solver_kwargs)
        else:
            raise ValueError(
                f"Eval mode for RL solver '{solver_name}' requires cfg.ckpt_path. "
                "Use mode='train_and_eval' or pass ckpt_path=... on the CLI."
            )

    wandb_run = None
    if cfg.get("logger", {}).get("wandb"):
        from nrp.utils.logging import init_wandb

        wb_cfg = cfg.logger.wandb
        wandb_run = init_wandb(
            project=wb_cfg.get("project", "rl4co"),
            name=wb_cfg.get("name"),
            group=wb_cfg.get("group"),
            tags=wb_cfg.get("tags"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    save_dir = None
    if "paths" in cfg and cfg.paths.get("output_dir"):
        save_dir = cfg.paths.output_dir
    run_id = cfg.get("run_id")
    if run_id is None:
        # Auto-generate a run_id from solver + env + num_loc
        run_id = f"{solver_name}-{env.name}-{generator_params.get('num_loc', '?')}"
    method = cfg.evaluate.get("method", "augment_dihedral_8")
    return evaluate(
        solver, env, td,
        method=method,
        wandb_run=wandb_run,
        save_dir=save_dir,
        run_id=run_id,
    )


def _find_best_checkpoint(cfg: DictConfig) -> str | None:
    """Find the best checkpoint under cfg.paths.output_dir."""
    output_dir = Path(
        cfg.paths.get("output_dir", "outputs") if "paths" in cfg else "outputs"
    )
    if not output_dir.exists():
        # Also try hydra's runtime config dir if present
        return None
    ckpts = list(output_dir.rglob("*.ckpt"))
    if not ckpts:
        return None
    for c in ckpts:
        if "best" in c.name:
            return str(c)
    return str(ckpts[0])
