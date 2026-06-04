"""Argparse CLI for the nrp evaluation pipeline.

Subcommands: train, eval, sweep. Each forwards remaining args to the
appropriate Hydra entry (rl4co.tasks.train.train for train, our own
hydra-based dispatch for eval/sweep).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _config_dir() -> str:
    return str(Path(__file__).resolve().parent.parent / "configs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nrp", description="NRP evaluation pipeline")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("train", help="Train a model (delegates to rl4co.tasks.train.train)")
    sub.add_parser("eval", help="Evaluate a solver on a dataset")
    sub.add_parser("sweep", help="Run a cartesian-product sweep")
    return p


def _compose(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.version:
        from nrp import __version__

        print(__version__)
        return 0
    if args.command == "train":
        from rl4co.tasks.train import train as rl4co_train

        return rl4co_train(remaining)
    if args.command == "eval":
        from nrp.harness.train import train_and_evaluate

        cfg = _compose(["mode=eval", *remaining])
        train_and_evaluate(cfg)
        return 0
    if args.command == "sweep":
        from nrp.harness.runner import run_sweep

        # expect: --solvers pomo,am --envs tsp,cvrp --num_locs 50,100 --seeds 42
        kwargs = {}
        new_remaining = []
        i = 0
        while i < len(remaining):
            tok = remaining[i]
            if tok == "--solvers" and i + 1 < len(remaining):
                kwargs["solver_names"] = remaining[i + 1].split(",")
                i += 2
            elif tok == "--envs" and i + 1 < len(remaining):
                kwargs["env_names"] = remaining[i + 1].split(",")
                i += 2
            elif tok == "--num_locs" and i + 1 < len(remaining):
                kwargs["num_locs"] = [int(x) for x in remaining[i + 1].split(",")]
                i += 2
            elif tok == "--seeds" and i + 1 < len(remaining):
                kwargs["seeds"] = [int(x) for x in remaining[i + 1].split(",")]
                i += 2
            else:
                new_remaining.append(tok)
                i += 1
        cfg = _compose(new_remaining)
        run_sweep(cfg, **kwargs)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
