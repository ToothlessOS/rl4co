"""Tests for :func:`nrp.utils.reproducibility.seed_everything`."""
from __future__ import annotations

import numpy as np
import torch

from nrp.utils.reproducibility import seed_everything


def test_seed_everything_idempotent():
    seed_everything(42)
    a = torch.rand(10)
    b = np.random.rand(10)
    seed_everything(42)
    a2 = torch.rand(10)
    b2 = np.random.rand(10)
    assert torch.allclose(a, a2)
    assert np.allclose(b, b2)
