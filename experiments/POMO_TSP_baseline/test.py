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
    baseline="shared",
    batch_size=128,
    train_data_size=100_000,
    val_data_size=10_000,
    optimizer_kwargs={"lr": 1e-4},
)

ckpt = POMO.load_from_checkpoint(
    "checkpoints/last-v1.ckpt", strict=False, weights_only=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preview / Testing
td_init = env.reset(batch_size=[3]).to(device)
policy = model.policy.to(device)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_trained = out["actions"].cpu().detach()

# Plotting
import matplotlib.pyplot as plt

for i, td in enumerate(td_init):
    fig, axs = plt.subplots(1, 1, figsize=(11, 5))
    env.render(td, actions_trained[i], ax=axs)
    axs.set_title(r"Trained $\pi_\theta$" + f"| Cost = {-out['reward'][i].item():.3f}")
    fig.savefig(f"trained_{i}.png")
