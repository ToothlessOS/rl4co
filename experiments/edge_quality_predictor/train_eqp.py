"""Hydra entrypoint for the edge quality predictor.

Mirrors ``experiments/POMO_eval/train_pomo_tsp.py``: compose a config from
``configs/eqp_main.yaml``, instantiate the Lightning data module + module,
configure the ``RL4COTrainer`` with model-checkpoint / progress / wandb
callbacks, and run ``trainer.fit`` followed by ``trainer.test``.

CLI overrides follow standard Hydra syntax, e.g.::

    python train_eqp.py data.batch_size=64 optim.lr=3e-4 train=false test=true

The script uses ``hydra.initialize_config_dir`` + ``hydra.compose`` so it
can be run from anywhere (the ``PROJECT_ROOT`` env var anchors ``paths``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import lightning.pytorch as pl

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eqp.lightning import EdgeQualityDataModule, EdgeQualityModule
from rl4co.utils import (
    RL4COTrainer,
    instantiate_loggers,
    log_hyperparameters,
)


_HERE = Path(__file__).resolve().parent
_CONFIG_DIR = _HERE / "configs"


def _build_trainer(cfg) -> RL4COTrainer:
    """Build the RL4COTrainer with the configured callbacks / loggers."""
    # ModelCheckpoint callback.
    from lightning.pytorch.callbacks import (
        LearningRateMonitor,
        ModelCheckpoint,
        RichModelSummary,
    )

    ckpt_cfg = cfg.callbacks.model_checkpoint
    if ckpt_cfg.get("dirpath") is None:
        ckpt_cfg.dirpath = str(Path(cfg.paths.output_dir) / "checkpoints")
    if ckpt_cfg.get("filename") is None:
        ckpt_cfg.filename = "epoch_{epoch:03d}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_cfg.dirpath),
        filename=str(ckpt_cfg.filename),
        monitor=ckpt_cfg.get("monitor"),
        mode=str(ckpt_cfg.get("mode", "max")),
        save_top_k=int(ckpt_cfg.get("save_top_k", 1)),
        save_last=bool(ckpt_cfg.get("save_last", True)),
        auto_insert_metric_name=bool(ckpt_cfg.get("auto_insert_metric_name", True)),
    )
    summary_cb = RichModelSummary(max_depth=int(cfg.callbacks.model_summary.max_depth))
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_cb, summary_cb, lr_cb]

    loggers = instantiate_loggers(cfg.get("logger"), _DummyModule())

    trainer = RL4COTrainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices) if isinstance(cfg.trainer.devices, int) else cfg.trainer.devices,
        logger=loggers or cfg.trainer.logger,
        callbacks=callbacks,
        gradient_clip_val=float(cfg.trainer.get("gradient_clip_val", 1.0)),
        precision=str(cfg.trainer.get("precision", "16-mixed")),
        default_root_dir=str(cfg.trainer.default_root_dir),
    )
    return trainer, loggers, callbacks


class _DummyModule:
    """Used so instantiate_loggers can read hparams; replaced after."""

    def __init__(self):
        self.hparams = OmegaConf.create({"cfg": {}})


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PROJECT_ROOT", str(_HERE))

    # Pass through CLI overrides to Hydra. Strip the script name if present.
    overrides: list[str] = []
    if argv is not None:
        overrides = list(argv)
    else:
        # Pick up everything after the script name from sys.argv.
        overrides = sys.argv[1:]

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="eqp_main.yaml", overrides=overrides)

    if cfg.get("print_config", True):
        print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed") is not None:
        pl.seed_everything(int(cfg.seed), workers=True)

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    optim_cfg = OmegaConf.to_container(cfg.optim, resolve=True)

    datamodule = EdgeQualityDataModule(data_cfg)
    module = EdgeQualityModule(model_cfg, optim_cfg)

    trainer, loggers, callbacks = _build_trainer(cfg)

    if loggers:
        log_hyperparameters(
            {"cfg": cfg, "model": module, "trainer": trainer, "callbacks": callbacks,
             "logger": loggers}
        )

    if cfg.get("train", True):
        trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test", True):
        trainer.test(module, datamodule=datamodule, ckpt_path="best")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))