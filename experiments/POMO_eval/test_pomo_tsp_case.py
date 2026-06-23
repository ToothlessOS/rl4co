import glob
import os
import sys

import lightning as L
import torch

from hydra import compose, initialize
from omegaconf import OmegaConf

from rl4co.envs import TSPEnv
from rl4co.models import POMO

from utils.gmm_tsp import GMMSampler

# Hydra configs (CLI overrides supported, e.g.
# python test_pomo_tsp.py seed=99 ckpt_path=/abs/path/last.ckpt)
ROOT_DIR = "./"
with initialize(version_base=None, config_path=ROOT_DIR + "configs/"):
    cfg = compose(config_name="pomo_tsp_main.yaml", overrides=sys.argv[1:])

print(OmegaConf.to_yaml(cfg))

# Seed before env.reset so the test batch and local search are reproducible
if cfg.get("seed") is not None:
    L.seed_everything(cfg.seed, workers=True)


def resolve_ckpt_path() -> str:
    """Pick a checkpoint: explicit `ckpt_path=` override, otherwise the most
    recent `*.ckpt` under `paths.output_dir`."""
    for arg in sys.argv[1:]:
        if arg.startswith("ckpt_path="):
            return arg.split("=", 1)[1]
    candidates = sorted(
        glob.glob(os.path.join(cfg.paths.output_dir, "**", "*.ckpt"), recursive=True),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {cfg.paths.output_dir}. "
            "Train first or pass one explicitly: "
            "python test_pomo_tsp.py ckpt_path=/abs/path/last.ckpt"
        )
    if len(candidates) > 1:
        print(
            f"Multiple checkpoints found; using most recent ({len(candidates)} total)."
        )
        for c in candidates[-5:]:
            print(f"  {c}")
    return candidates[-1]


# Load model checkpoint
ckpt_path = resolve_ckpt_path()
print(f"Using checkpoint: {ckpt_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
# ``OmegaConf.to_container`` returns a plain Python dict — `dict(cfg...)`
# preserves OmegaConf's struct flag and rejects unknown keys.
gen_params = OmegaConf.to_container(cfg.env.generator_params, resolve=True)
gen_params["loc_sampler"] = GMMSampler(
    num_modes=5, std=0.1, seed=cfg.get("seed") or 0
)
env = TSPEnv(generator_params=gen_params)
model = POMO.load_from_checkpoint(ckpt_path, env=env, weights_only=False).to(device)

# Run inference on test data (n=3)
td_init = env.reset(batch_size=[3]).to(device)
out = model(td_init.clone(), phase="test", decode_type="greedy")
actions = out["actions"]

td_init = td_init.to("cpu")
actions = actions.to("cpu")

## Test 1: Check and visualize the results of 2-opt improvement

# Improve solutions using LocalSearch
improved_actions = env.local_search(td_init, actions)
improved_rewards = env.get_reward(td_init, improved_actions)

# Plotting
import matplotlib

matplotlib.use("Agg")  # non-interactive; plt.show() below is a no-op then
import matplotlib.pyplot as plt

for i, td in enumerate(td_init):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    env.render(td, actions[i], ax=axs[0])
    env.render(td, improved_actions[i], ax=axs[1])
    axs[0].set_title(f"Before improvement | Cost = {-out['reward'][i].item():.3f}")
    axs[1].set_title(f"After improvement | Cost = {-improved_rewards[i].item():.3f}")
    plt.close(fig)


## Test 2: Compare and visualize the results to solutions from LKH-3

import os

import matplotlib.pyplot as plt
import torch

from utils.lkh_tsp import _resolve_lkh_binary, solve_lkh3_tsp_batch

# ---- LKH-3 on the same 3-instance Test-1 batch -----------------------------
# Reuses `td_init["locs"]` (the canonical POMO test instances) and
# `actions` / `improved_actions` from Test 1 above, so the route panels
# compare all three methods on identical instances.

n_fig = 3
binary = _resolve_lkh_binary(
    binary_path="/home/toothlessos/Projects/nrp/rl4co/experiments/POMO_eval/LKH-3.0.14/LKH"
)
print(f"Using LKH-3 binary: {binary}")

fig_perms, fig_lengths = solve_lkh3_tsp_batch(
    td_init["locs"],  # [3, n, 2]
    binary_path=binary,
    max_trials=100_000,
    n_workers=4,
)
# Convert LKH-3 permutations to a torch tensor for the renderer.
lkh_actions = torch.tensor(
    [p if p is not None else list(range(td_init["locs"].shape[1])) for p in fig_perms],
    dtype=torch.long,
)
lkh_lengths = torch.tensor(
    [l if l != float("inf") else float("nan") for l in fig_lengths],
    dtype=torch.float32,
)

# ---- Figure A: 3 (instances) x 3 (methods) route grid ----------------------
fig, axs = plt.subplots(n_fig, 3, figsize=(13, 4 * n_fig))
methods = [
    ("POMO greedy", actions, -out["reward"].to("cpu")),
    ("POMO + 2-opt", improved_actions, -improved_rewards.to("cpu")),
    ("LKH-3", lkh_actions, lkh_lengths),
]
for r in range(n_fig):
    for c, (label, acts, lens) in enumerate(methods):
        ax = axs[r, c] if n_fig > 1 else axs[c]
        env.render(td_init[r], acts[r], ax=ax)
        ax.set_title(f"{label} | L = {lens[r]:.3f}")
    # Row label on the left-most panel.
    (axs[r, 0] if n_fig > 1 else axs[0]).set_ylabel(
        f"instance {r}", fontsize=11, rotation=90, labelpad=10
    )
fig.suptitle(
    f"POMO vs LKH-3 routes (n={td_init['locs'].shape[1]} per instance, "
    f"LKH-3 = stock 3.0.14, MAX_TRIALS=100k)",
    fontsize=13,
)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/pomo_vs_lkh3_routes.png", dpi=120, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.5)

# ---- Figure B: length distribution histogram on a larger batch -------------
# HIST_BATCH is intentionally modest: 2-opt (numba) on CPU is the slow
# step, not the LKH-3 calls. 16 instances is enough to show a
# distribution and keeps total wall-clock under a few minutes.
HIST_BATCH = 8
hist_td = env.reset(batch_size=[HIST_BATCH]).to("cpu")
print(f"Running histogram batch ({HIST_BATCH} instances): LKH-3 sweep...")
hist_perms, hist_lengths = solve_lkh3_tsp_batch(
    hist_td["locs"],
    binary_path=binary,
    max_trials=100_000,
    n_workers=4,
)
print(f"  LKH-3 done. Running POMO greedy + 2-opt on {HIST_BATCH} instances...")
out_h = model(hist_td.clone().to(device), phase="test", decode_type="greedy")
greedy_actions_h = out_h["actions"].to("cpu")
opt_actions_h = env.local_search(hist_td, greedy_actions_h)
greedy_l = -out_h["reward"].to("cpu")
opt_l = -env.get_reward(hist_td, opt_actions_h).to("cpu")
lkh_l = torch.tensor(
    [l if l != float("inf") else float("nan") for l in hist_lengths],
    dtype=torch.float32,
)
print(f"  All three methods done for the histogram batch.")

fig, ax = plt.subplots(figsize=(8, 5))
bins = 20
ax.hist(greedy_l.numpy(), bins=bins, alpha=0.5, label="POMO greedy", color="C0")
ax.hist(opt_l.numpy(), bins=bins, alpha=0.5, label="POMO + 2-opt", color="C1")
ax.hist(lkh_l.numpy(), bins=bins, alpha=0.5, label="LKH-3", color="C2")
for arr, label, color in [
    (greedy_l, "POMO greedy", "C0"),
    (opt_l, "POMO + 2-opt", "C1"),
    (lkh_l, "LKH-3", "C2"),
]:
    m = torch.nanmean(arr).item()
    ax.axvline(
        m, color=color, linestyle="--", linewidth=1.5, label=f"mean {label} = {m:.3f}"
    )
ax.set_xlabel("Tour length (Euclidean)")
ax.set_ylabel("Count")
ax.set_title(
    f"POMO vs LKH-3 length distribution "
    f"(n={HIST_BATCH} instances, TSP-{td_init['locs'].shape[1]})"
)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("figures/pomo_vs_lkh3_histogram.png", dpi=120, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.5)


# ---- Console summary --------------------------------------------------------
def _stats(arr: torch.Tensor) -> str:
    finite = arr[~torch.isnan(arr)]
    if finite.numel() == 0:
        return "no finite samples"
    return f"mean = {finite.mean():.3f}  std = {finite.std():.3f}  (n={finite.numel()})"


print(
    f"\n=== Test 2: POMO vs LKH-3 ({HIST_BATCH} instances, "
    f"TSP-{td_init['locs'].shape[1]}) ==="
)
for label, arr in [
    ("POMO greedy", greedy_l),
    ("POMO + 2-opt", opt_l),
    ("LKH-3", lkh_l),
]:
    print(f"  {label:14s}  {_stats(arr)}")

# Pairwise win/tie/loss: POMO+2opt vs LKH-3, only over finite pairs.
both_finite = ~torch.isnan(opt_l) & ~torch.isnan(lkh_l)
pomo_wins = ((opt_l < lkh_l) & both_finite).sum().item()
ties = ((opt_l == lkh_l) & both_finite).sum().item()
lkh_wins = ((opt_l > lkh_l) & both_finite).sum().item()
n_both = both_finite.sum().item()
print(
    f"  POMO+2opt wins: {pomo_wins}   ties: {ties}   "
    f"LKH-3 wins: {lkh_wins}   (over {n_both} finite pairs of {HIST_BATCH})"
)
