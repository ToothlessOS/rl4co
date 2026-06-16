import glob
import os
import sys

import lightning as L
import torch

from hydra import compose, initialize
from omegaconf import OmegaConf

from rl4co.envs import TSPEnv
from rl4co.models import POMO

from utils.HK import held_karp_one_tree_lower_bound, tour_edges, edge_overlap

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
env = TSPEnv(generator_params=dict(cfg.env.generator_params))
model = POMO.load_from_checkpoint(ckpt_path, env=env, weights_only=False).to(device)

# Run inference on test data (n=64)
td_init = env.reset(batch_size=[64]).to(device)
out = model(td_init.clone(), phase="test", decode_type="greedy")
actions = out["actions"]

td_init = td_init.to("cpu")
actions = actions.to("cpu")

## Test 1: Can 2-opt LocalSearch improve the solutions?

# Improve solutions using LocalSearch
improved_actions = env.local_search(td_init, actions)
improved_rewards = env.get_reward(td_init, improved_actions)

# Compute percent+-stddev of improvement
improvement = (improved_rewards - out["reward"].to("cpu")) / (-out["reward"].to("cpu"))
print(f"Improvement: {improvement.mean().item():.2%} ± {improvement.std().item():.2%}")

## Test 2: Compare solution to HK lower bound

# Held-Karp 1-tree per instance (B=64, n=100).
hk_lower_bound, hk_edges = held_karp_one_tree_lower_bound(td_init["locs"])

# Positive tour lengths (rl4co reward = -length). `out["reward"]` lives on
# the model device; move to CPU for arithmetic with the CPU-side HK bound.
greedy_lengths = -out["reward"].to("cpu")
opt_lengths = -improved_rewards

greedy_gap = (greedy_lengths - hk_lower_bound) / hk_lower_bound
opt_gap = (opt_lengths - hk_lower_bound) / hk_lower_bound

print(
    f"Greedy gap to HK bound:        "
    f"{greedy_gap.mean().item():.2%} +/- {greedy_gap.std().item():.2%}"
)
print(
    f"Greedy+2-opt gap to HK bound:  "
    f"{opt_gap.mean().item():.2%} +/- {opt_gap.std().item():.2%}"
)

# Edge overlap between tours and the 1-tree (both have exactly n=100 edges).
greedy_overlap = edge_overlap(tour_edges(actions), hk_edges)
opt_overlap = edge_overlap(tour_edges(improved_actions), hk_edges)
print(
    f"Greedy vs HK 1-tree edge overlap:        "
    f"{greedy_overlap.mean().item():.2%} +/- {greedy_overlap.std().item():.2%}"
)
print(
    f"Greedy+2-opt vs HK 1-tree edge overlap:  "
    f"{opt_overlap.mean().item():.2%} +/- {opt_overlap.std().item():.2%}"
)
