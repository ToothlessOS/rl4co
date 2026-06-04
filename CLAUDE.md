# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL4CO is a unified PyTorch Lightning-based benchmark/library for **Reinforcement Learning for Combinatorial Optimization (CO)**. Built on top of **TorchRL** (vectorized GPU envs), **TensorDict** (heterogeneous state containers), and **Hydra** (config composition). Python ≥ 3.10.

This repo is a fork of RL4CO and we will be running experiments for research based on it. The focus is experiments based on RL4CO, rather than modifying the repo itself. `wandb` is used for logging.

## Project Structure

We mainly work on the 3 folders - `experiments`, `docs/content` and `examples`:
- `experiments` - the place where we keep the experiment code, model checkpoints, configs and results; Each experiment has a seperate folder.
- `docs/content` - docs and api ref for RL4CO in markdown.
- `examples` - example implementations with RL4CO.

## Common Commands

All commands assume the repo root as cwd. The project uses `uv` for env management.

### Setup
```bash
uv sync --all-extras         # create .venv with all extras (routing, graph, dev, docs)
source .venv/bin/activate
pre-commit install           # install git hooks (black + ruff)
```

### Lint / Format
`pre-commit` runs `black` then `ruff` on every commit (config in `.pre-commit-config.yaml`). Manual:
```bash
ruff check rl4co tests                  # lint (line-length 100, target py310; rules F/E/W/I001/UP)
ruff check --fix rl4co tests            # autofix
black rl4co tests                       # format
```

## Important Caveats

(Currently Empty)