"""Train/val/test split helpers, including optional optimum tours."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SplitSpec:
    """Specification for a (train/val/test) split of a generated dataset.

    Stage 1 keeps this simple: generate three independent batches with three
    fixed seeds (the RL4CO convention: train=1234, val=4321, test=1234).
    """

    env_name: str
    num_loc: int
    num_train: int = 100_000
    num_val: int = 10_000
    num_test: int = 10_000
    seed_train: int = 1234
    seed_val: int = 4321
    seed_test: int = 1234
    loc_distribution: str = "uniform"

    def as_dict(self) -> dict:
        return {
            "env_name": self.env_name,
            "num_loc": self.num_loc,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "num_test": self.num_test,
            "seed_train": self.seed_train,
            "seed_val": self.seed_val,
            "seed_test": self.seed_test,
            "loc_distribution": self.loc_distribution,
        }


def split_filename(env_name: str, num_loc: int, phase: str, seed: int) -> str:
    """Return the canonical .npz filename for a phase+seed.

    Example: split_filename('tsp', 50, 'test', 1234) -> 'tsp50_test_seed1234.npz'
    """
    return f"{env_name}{num_loc}_{phase}_seed{seed}.npz"
