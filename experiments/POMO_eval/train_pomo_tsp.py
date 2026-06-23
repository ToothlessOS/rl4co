from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary
import lightning as L

from rl4co.envs import TSPEnv
from rl4co.models import POMO
from rl4co.utils import RL4COTrainer, instantiate_loggers, log_hyperparameters

from hydra import compose, initialize
from omegaconf import OmegaConf

from utils.gmm_tsp import GMMSampler

# Hydra configs
ROOT_DIR = "./"
with initialize(version_base=None, config_path=ROOT_DIR + "configs/"):
    cfg = compose(config_name="pomo_tsp_main.yaml")

# Seed Python/NumPy/PyTorch (CPU+CUDA) and DataLoader workers.
# Set cfg.seed in the YAML or via CLI (e.g. `python train.py seed=42`).
# Leave `seed: null` for sweeps so the sweep launcher controls it.
if cfg.get("seed") is not None:
    L.seed_everything(cfg.seed, workers=True)

print(OmegaConf.to_yaml(cfg))

# RL4CO env based on TorchRL — GMM-sampled locations instead of uniform.
# ``OmegaConf.to_container`` returns a plain Python dict — `dict(cfg...)`
# preserves OmegaConf's struct flag and rejects unknown keys.
gen_params = OmegaConf.to_container(cfg.env.generator_params, resolve=True)
gen_params["loc_sampler"] = GMMSampler(
    num_modes=5, std=0.1, seed=cfg.get("seed") or 0
)
env = TSPEnv(generator_params=gen_params)

# RL Model: REINFORCE and greedy rollout baseline
model = POMO(
    env,
    # Use default AttentionModelPolicy with default hyperparameters
    baseline=cfg.model.baseline,
    num_augment=cfg.model.num_augment,
    batch_size=cfg.model.batch_size,
    train_data_size=cfg.model.train_data_size,
    val_data_size=cfg.model.val_data_size,
    optimizer_kwargs=dict(cfg.model.optimizer_kwargs),
)

# Pytorch Lightning callbacks for checkpointing and model summary

# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(
    dirpath=cfg.callbacks.model_checkpoint.dirpath,
    filename=cfg.callbacks.model_checkpoint.filename,
    save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
    save_last=cfg.callbacks.model_checkpoint.save_last,
    monitor=cfg.callbacks.model_checkpoint.monitor,
    mode=cfg.callbacks.model_checkpoint.mode,
)

# Print model summary
rich_model_summary = RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth)

# Callbacks list
callbacks = [checkpoint_callback, rich_model_summary]

# Loggers (e.g. wandb from configs/logger/wandb.yaml)
loggers = instantiate_loggers(cfg.get("logger"), model)

# Trainer
trainer = RL4COTrainer(
    max_epochs=cfg.trainer.max_epochs,
    accelerator=cfg.trainer.accelerator,
    devices=cfg.trainer.devices,
    logger=loggers or cfg.trainer.logger,
    callbacks=callbacks,
)

# Upload the full resolved config + parameter counts to wandb's run panel
if loggers:
    log_hyperparameters(
        {
            "cfg": cfg,
            "model": model,
            "trainer": trainer,
            "callbacks": callbacks,
            "logger": loggers,
        }
    )

# Train
trainer.fit(model)
