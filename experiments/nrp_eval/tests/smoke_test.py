"""End-to-end smoke test: train POMO-TSP-20 for 1 epoch, eval 50 instances.

Skipped by default (set NRP_RUN_SMOKE=1 to enable). The default test run
should be quick — this test is marked slow.
"""
import os

import pytest


@pytest.mark.skipif(
    os.environ.get("NRP_RUN_SMOKE") != "1",
    reason="Smoke test disabled by default; set NRP_RUN_SMOKE=1 to run",
)
def test_smoke_pomo_tsp():
    """End-to-end: train a tiny POMO-TSP-20, then eval 50 instances."""
    from hydra import compose, initialize_config_dir

    from nrp.harness.train import train_and_evaluate

    CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=nrp_eval/pomo_tsp",
                "trainer.max_epochs=1",
                "model.batch_size=4",
                "model.train_data_size=16",
                "model.val_data_size=16",
                "model.test_data_size=16",
                "env.generator_params.num_loc=20",
                "evaluate.num_instances=50",
                "evaluate.method=greedy",
                "logger=csv",
            ],
        )
        cfg.mode = "train_and_eval"
        result = train_and_evaluate(cfg)

    assert result.summary["num_instances"] == 50
    assert result.summary["feasible_ratio"] >= 0.95
