"""Reproducibility: seed every relevant RNG.

A single entry point that seeds Python's ``random``, ``numpy``, ``torch`` (CPU
and CUDA) and sets ``PYTHONHASHSEED``. When ``deterministic=True`` the
backends are also configured for bitwise-reproducible runs (slower).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed python, numpy, torch, and CUDA RNGs.

    Args:
        seed: The seed to use everywhere.
        deterministic: If True, also configure cuDNN for bitwise-reproducible
            runs (note: this can be slow and is not always possible).

    Returns:
        The seed used (for logging).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
