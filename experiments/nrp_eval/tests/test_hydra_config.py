"""Test that the Hydra config tree resolves correctly."""
from __future__ import annotations

import os

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


def test_hydra_main_composes():
    """`hydra.compose('main')` resolves all defaults; cfg.env._target_ is set."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="main")
    assert cfg.env is not None
    assert "_target_" in cfg.env
    assert cfg.env._target_ == "rl4co.envs.TSPEnv"
    assert cfg.env.name == "tsp"


def test_hydra_experiment_composes():
    """A specific experiment override resolves cleanly."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=["experiment=nrp_eval/ortools_tsp"],
        )
    assert cfg.solver_name == "ortools_tsp"
    assert cfg.env.name == "tsp"
    assert cfg.evaluate.method == "greedy"


def test_hydra_logger_csv_resolves():
    """Switching logger to csv works (avoids W&B for tests)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=["logger=csv"],
        )
    assert cfg.logger.csv is not None
