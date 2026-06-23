"""GMM-based TSP location sampler for rl4co's TSPGenerator.

rl4co's ``TSPGenerator._generate`` calls ``loc_sampler.sample(size)``
where ``size = (*batch_size, num_loc, 2)`` and expects a tensor of
shape ``(*batch_size, num_loc, 2)``. A vanilla
``torch.distributions.MixtureSameFamily`` does not satisfy that
interface: calling ``gmm.sample((B, num_loc, 2))`` returns shape
``(B, num_loc, 2, event_dim)`` because the trailing ``2`` ends up in
front of the event dims rather than being absorbed.

This wrapper exposes the minimal ``sample(size)`` interface the
generator expects. Internally it uses ``MixtureSameFamily`` with
``Independent(Normal(loc=(num_modes, 2), scale=(num_modes, 2)), 2)``:
``MixtureSameFamily.sample((B, num_loc))`` already returns shape
``(B, num_loc, 2)`` because the mixture collapses the component
batch dim internally, so we drop the trailing ``2`` from ``size``
before delegating.

Centers are drawn once at construction time from a deterministic
``torch.Generator`` so successive calls produce consistent modes
(important when comparing across train / test runs). If you want
stochastic centers, leave ``seed=None``.
"""

from __future__ import annotations

import torch
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    Normal,
)


class GMMSampler:
    """GMM location sampler compatible with rl4co's TSPGenerator.

    Args:
        num_modes: number of mixture components (clusters).
        low: minimum coordinate value (per axis).
        high: maximum coordinate value (per axis).
        std: isotropic per-mode standard deviation. Keep small enough
            that modes stay visually separated (~0.05-0.15 for unit
            square).
        seed: optional seed for the center generator. Centers are
            sampled once at construction; pass ``None`` for fresh
            random centers per ``__init__`` call.
        device: torch device for the sampler buffers.

    Example:
        >>> sampler = GMMSampler(num_modes=5, std=0.1, seed=0)
        >>> locs = sampler.sample((4, 100, 2))  # [4, 100, 2]

    Wire it into rl4co via TSPEnv with the ``loc_sampler`` kwarg
    (which bypasses the class-string dispatch in ``get_sampler``)::

        gen_params["loc_sampler"] = GMMSampler(num_modes=5, std=0.1, seed=0)
        env = TSPEnv(generator_params=gen_params)
    """

    def __init__(
        self,
        num_modes: int = 5,
        low: float = 0.0,
        high: float = 1.0,
        std: float = 0.1,
        seed: int | None = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_modes = num_modes
        self.low = low
        self.high = high
        self.std = std

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

        # Uniform mixture weights (one logit per mode).
        mix = Categorical(torch.ones(num_modes))

        # Random centers in [low, high]^2, isotropic std per mode.
        loc = (
            torch.rand((num_modes, 2), generator=gen)
            * (high - low)
            + low
        )
        scale = torch.full((num_modes, 2), float(std))

        # Independent(..., 1) → batch_shape=(num_modes,), event_shape=(2,).
        # Using reinterpreted_batch_ndims=2 would collapse *both* dims into
        # the event, leaving batch_shape=() and breaking MixtureSameFamily.
        comp = Independent(Normal(loc.to(device), scale.to(device)), 1)

        # MixtureSameFamily collapses the component batch dim on sample,
        # so .sample((B, n)) returns shape (B, n, 2).
        self._gmm: MixtureSameFamily = MixtureSameFamily(mix, comp)

    def sample(self, size: tuple[int, ...]) -> torch.Tensor:
        """Sample locations. ``size == (batch_size, num_loc, 2)``;
        trailing ``2`` is dropped because MixtureSameFamily already
        produces an ``(B, n, 2)`` tensor.
        """
        if len(size) != 3 or size[-1] != 2:
            raise ValueError(
                f"GMMSampler.sample expects size=(B, num_loc, 2); got {tuple(size)}"
            )
        batch_size, num_loc, _ = size
        return self._gmm.sample((batch_size, num_loc))