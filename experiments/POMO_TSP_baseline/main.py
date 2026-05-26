import torch
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary

from rl4co.envs import TSPEnv
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils.trainer import RL4COTrainer

# RL4CO env based on TorchRL
env = TSPEnv(generator_params={"num_loc": 100})


# RL Model: REINFORCE and greedy rollout baseline
model = POMO(
    env,
    # Use default AttentionModelPolicy with default hyperparameters
    baseline="shared",
    batch_size=128,
    train_data_size=100_000,
    val_data_size=10_000,
    optimizer_kwargs={"lr": 1e-4},
)

# Pytorch Lightning callbacks for checkpointing and model summary

# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",  # save to checkpoints/
    filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
    save_top_k=1,  # save only the best model
    save_last=True,  # save the last model
    monitor="val/reward",  # monitor validation reward
    mode="max",
)  # maximize validation reward

# Print model summary
rich_model_summary = RichModelSummary(max_depth=3)

# Callbacks list
callbacks = [checkpoint_callback, rich_model_summary]

# Trainer
trainer = RL4COTrainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    logger=True,
    callbacks=callbacks,
)

# Train
trainer.fit(model)
