"""Argparse CLI for the learn_decompose_eval pipeline.

Subcommand `eval` forwards remaining args to a Hydra entry that builds the
env + dataset, then calls nrp_eval's `evaluate()` directly. We bypass
`nrp.harness.train.train_and_evaluate` because its classical-solver
allow-list is hardcoded and does not include our raw_lkh_cvrp / bcc_lkh_cvrp
solvers.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

# Configure root logger before any submodules log so that the package's
# `log = logging.getLogger(__name__)` instances emit at INFO.  Hydra's own
# job_logging will add a per-run file handler so these lines also land in
# `${hydra.runtime.output_dir}/${name}.log`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Force our package's loggers to INFO even if a parent config dialed them down.
logging.getLogger("learn_decompose_eval").setLevel(logging.INFO)


def _config_dir() -> str:
    return str(Path(__file__).resolve().parent.parent / "configs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="learn_decompose_eval",
        description="LKH-3 ± Barycenter-Clustering CVRP evaluation",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("eval", help="Evaluate a solver on a CVRP dataset")
    return p


def _compose(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    return cfg


def _ensure_nrp_eval_on_path() -> None:
    """The harness lives in the sibling `nrp_eval` experiment; add it to sys.path."""
    import sys

    here = Path(__file__).resolve().parent
    nrp_eval = here.parent.parent / "nrp_eval"
    if nrp_eval.exists() and str(nrp_eval) not in sys.path:
        sys.path.insert(0, str(nrp_eval))


def _eval_only(cfg) -> int:
    """Build env + dataset, build solver, run nrp_eval's evaluate() loop."""
    _ensure_nrp_eval_on_path()
    from nrp.data.dataset import DatasetSpec, build_eval_dataset
    from nrp.envs.factory import build_env
    from nrp.harness.evaluate import evaluate
    from nrp.solvers import SolverRegistry
    from nrp.utils.logging import init_wandb

    env_name = cfg.env.name if "name" in cfg.env else cfg.get("solver_name", "tsp")
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
    solver = SolverRegistry.build(solver_name, env, **solver_kwargs)

    wandb_run = None
    if cfg.get("logger", {}).get("wandb"):
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
        run_id = f"{solver_name}-{env.name}-{generator_params.get('num_loc', '?')}"
    method = cfg.evaluate.get("method", "augment_dihedral_8")
    evaluate(
        solver,
        env,
        td,
        method=method,
        wandb_run=wandb_run,
        save_dir=save_dir,
        run_id=run_id,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.version:
        from learn_decompose_eval import __version__

        print(__version__)
        return 0
    if args.command == "eval":
        # Ensure the LKH binary env var is set if the user has it
        if "LDE_LKH_BINARY" in os.environ and "NRP_LKH_BINARY" not in os.environ:
            os.environ["NRP_LKH_BINARY"] = os.environ["LDE_LKH_BINARY"]
        cfg = _compose(["mode=eval", *remaining])
        return _eval_only(cfg)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
