"""RL solver: thin adapter between any RL4CO zoo policy and our ``Solver`` API.

The :class:`RLSolver` is the most general ``Solver`` in the nrp_eval pipeline.
It takes an RL4CO ``LightningModule`` (POMO, AM, SymNCO, MatNet, PointerNetwork)
or its underlying ``policy`` ``nn.Module`` and exposes the standard
:meth:`solve(td) -> TensorDict(actions, reward)` contract.

Decoding strategies supported: ``greedy``, ``sampling``,
``multistart_greedy``, ``multistart_sampling``, ``beam_search``, and the
augment-aware variants (``augment_dihedral_8`` is realised by passing the
TensorDict through :class:`rl4co.data.transforms.StateAugmentation` before
calling the policy).
"""
from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from .base import Solver, SolverRegistry

if TYPE_CHECKING:  # pragma: no cover
    from rl4co.envs.common.base import RL4COEnvBase
    from rl4co.models.rl.reinforce.reinforce import REINFORCE

# Lazy import of RL4CO models. Done at module import time but inside a
# try/except so that a missing optional dependency does not break the
# rest of the registry.
try:
    from rl4co.models import (
        POMO,
        AttentionModel,
        MatNet,
        PointerNetwork,
        SymNCO,
    )

    ZOO: dict[str, type] = {
        "pomo": POMO,
        "am": AttentionModel,
        "symnco": SymNCO,
        "matnet": MatNet,
        "ptrnet": PointerNetwork,
    }
except Exception as exc:  # pragma: no cover - very defensive
    warnings.warn(f"Failed to import RL4CO zoo models: {exc}", stacklevel=2)
    ZOO = {}


# Environment names that the zoo policies nominally support. Mirrors the
# list in rl4co/envs/__init__.py. Used only to populate the registry's
# ``env_names`` field — it is the user's responsibility to ensure the
# loaded checkpoint matches the target env.
RL_ENV_NAMES: tuple[str, ...] = (
    "tsp",
    "cvrp",
    "sdvrp",
    "cvrptw",
    "svrp",
    "pctsp",
    "spctsp",
    "op",
    "mtsp",
    "atsp",
    "pdp",
    "pdp_ruin_repair",
    "tsp_kopt",
    "smtwtp",
    "mdcpdp",
    "mtvrp",
    "shpp",
    "cvrpmvc",
    "mcp",
    "dpp",
    "mdpp",
    "flp",
)


# Decoding strategies that require augmentation. The classical / hybrid
# solvers cannot honour these, so we treat them as RL-only.
_AUGMENT_DECODES = {"augment_dihedral_8", "augment_symmetric"}


class RLSolver(Solver):
    """Adapter that wraps an RL4CO zoo policy as a :class:`Solver`.

    Construction paths:

    - :meth:`from_checkpoint` — load a Lightning module from a ``.ckpt`` file
      via ``ZOO[model_name].load_from_checkpoint(...)``.
    - :meth:`from_policy` — wrap an already-instantiated ``policy``
      ``nn.Module`` (e.g. from a freshly-constructed untrained model in tests).
    - :meth:`__init__` — pass a fully-loaded Lightning ``model`` directly.

    Args:
        env: An RL4CO env instance.
        model: A Lightning module exposing ``.policy`` (e.g. ``POMO``).
        model_name: Display name (e.g. ``"pomo"``).
        ckpt_path: Original checkpoint path (kept for logging/metadata).
        decode_type: One of the strategies supported by the policy
            (see :class:`rl4co.utils.decoding`).
        num_starts: Number of parallel rollouts (for ``multistart_*``); ``0``
            to disable multistart.
        num_augment: Number of dihedral augmentations (for augment-aware
            decoders); ``0`` to disable.
        augmentation: Pre-built :class:`rl4co.data.transforms.StateAugmentation`
            instance, or ``None`` to construct one on demand.
        device: Torch device to run the policy on.
        **kwargs: Forwarded to the :class:`Solver` base.
    """

    is_trainable: bool = True
    is_differentiable: bool = False

    def __init__(
        self,
        env: RL4COEnvBase,
        model: REINFORCE | None = None,
        model_name: str = "unknown",
        ckpt_path: str | None = None,
        decode_type: str = "greedy",
        num_starts: int = 0,
        num_augment: int = 0,
        augmentation: Any = None,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        if model is None:
            raise ValueError(
                "`model` is required. Use `RLSolver.from_checkpoint(...)` or "
                "`RLSolver.from_policy(...)` for the standard construction paths."
            )

        self.model = model
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.decode_type = decode_type
        self.num_starts = num_starts
        self.num_augment = num_augment
        # `self.policy` is the actual nn.Module we call.
        self.policy = model.policy
        self.augmentation = augmentation

        # If the user asked for augmentation but didn't pass a transform,
        # construct one. We pick the dihedral-8 family only when the
        # decode_type says so; otherwise default to the 4x symmetric rotation
        # family used by the AM paper.
        if (
            self.augmentation is None
            and self.num_augment
            and self.num_augment > 1
        ):
            from rl4co.data.transforms import StateAugmentation

            aug_fn = (
                "dihedral8"
                if self.decode_type == "augment_dihedral_8"
                else "symmetric"
            )
            self.augmentation = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=aug_fn,
            )

        # Move policy to the requested device.
        self.policy.to(self.device)
        self.name = model_name

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        env: RL4COEnvBase,
        ckpt_path: str,
        model_name: str = "pomo",
        map_location: str | torch.device | None = None,
        **kwargs: Any,
    ) -> RLSolver:
        """Load a zoo Lightning module from a checkpoint.

        Mirrors the pattern in ``experiments/POMO_TSP_baseline/test.py`` and
        :meth:`rl4co.models.rl.reinforce.REINFORCE.load_from_checkpoint`.
        """
        if not ZOO:
            raise RuntimeError(
                "RL4CO zoo models are not importable; cannot load from checkpoint."
            )
        key = model_name.lower()
        if key not in ZOO:
            raise KeyError(
                f"Unknown model_name '{model_name}'. Available: {sorted(ZOO)}"
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = ZOO[key].load_from_checkpoint(
            ckpt_path,
            map_location=map_location,
            load_baseline=False,
            strict=False,
            weights_only=False,
        )
        return cls(
            env=env,
            model=model,
            model_name=model_name,
            ckpt_path=ckpt_path,
            **kwargs,
        )

    @classmethod
    def from_policy(
        cls,
        env: RL4COEnvBase,
        policy: torch.nn.Module,
        model_name: str = "wrapped_policy",
        **kwargs: Any,
    ) -> RLSolver:
        """Wrap a pre-existing policy ``nn.Module`` (e.g. ``POMO(env).policy``).

        Useful in tests where a freshly-constructed (untrained) model is
        available and a full checkpoint is not.
        """
        # Build a tiny shim so the Solver can keep a single ``self.model``
        # attribute for logging. The shim is intentionally not a Lightning
        # module — it just exposes ``.policy``.
        shim = _PolicyShim(policy=policy, model_name=model_name)
        return cls(
            env=env,
            model=shim,  # type: ignore[arg-type]
            model_name=model_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------
    def solve(self, td: TensorDict) -> TensorDict:
        """Run the policy on a batch of instances.

        Steps:
        1. Move ``td`` to ``self.device`` and clone (don't mutate caller's td).
        2. Optionally apply dihedral/symmetric augmentation. This expands the
           batch by ``num_augment``; we then call the policy and pass the
           augmented ``td`` to ``env.get_reward`` so the reward shape matches
           the actions. The stage-1 contract returns whatever the policy
           produced — the caller is responsible for ``select_best`` reduction
           if they want a per-instance best across augmentations.
        3. Call ``self.policy(td, env=self.env, phase="test", ...)``.
        4. Return ``TensorDict(actions, reward, batch_size=actions.shape[:1])``.
        """
        td = td.to(self.device).clone()
        if self.augmentation is not None:
            td = self.augmentation(td)

        out = self.policy(
            td,
            env=self.env,
            phase="test",
            decode_type=self.decode_type,
            num_starts=self.num_starts,
        )
        actions = out["actions"]

        # Compute reward on the (possibly augmented) td. The reward shape
        # matches `actions.shape[:1]`; downstream EvalBase wrappers will
        # `select_best` across the num_augment dimension.
        reward = self.env.get_reward(td, actions)

        return TensorDict(
            actions=actions,
            reward=reward,
            batch_size=actions.shape[:1],
        )

    def to(self, device: str | torch.device) -> RLSolver:
        """Move the policy to ``device`` (also updates ``self.device``)."""
        self.device = torch.device(device)
        if self.policy is not None:
            self.policy.to(self.device)
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RLSolver(model={self.model_name!r}, decode={self.decode_type!r}, "
            f"num_starts={self.num_starts}, num_augment={self.num_augment}, "
            f"device={self.device})"
        )


class _PolicyShim:
    """Tiny stand-in for a Lightning module that exposes ``.policy``.

    Used by :meth:`RLSolver.from_policy` so the solver can keep a single
    ``self.model`` attribute (for logging / metadata) while the caller
    supplies only the underlying ``policy`` ``nn.Module``.
    """

    def __init__(self, policy: torch.nn.Module, model_name: str = "wrapped_policy"):
        self.policy = policy
        self.model_name = model_name

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"_PolicyShim(name={self.model_name!r})"


# ----------------------------------------------------------------------
# Registry: one entry per zoo model. Each is bound to the same RLSolver
# class, dispatched on the model name. MatNet's sparse-graph 2-opt
# requires `(num_loc + 1) % 2 == 0` — that's the user's responsibility to
# honour when constructing the env.
# ----------------------------------------------------------------------
for _name in ("pomo", "am", "symnco", "matnet", "ptrnet"):
    SolverRegistry.register(_name, env_names=RL_ENV_NAMES)(RLSolver)
