"""Keystone abstractions: ``Solver`` ABC and ``SolverRegistry``.

Every other module in the ``nrp_eval`` pipeline depends on these. A
:class:`Solver` is anything that turns a ``TensorDict`` of problem instances
into a ``TensorDict`` with an ``actions`` (int64) and ``reward`` (float) key,
suitable for ``env.get_reward(td, actions)``.

The :class:`SolverRegistry` provides a decorator-based registration mechanism
so that concrete solver implementations can be looked up by string name from
Hydra config files.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict


class Solver(ABC):
    """Abstract base class for all solvers in the nrp_eval pipeline.

    Subclasses must implement :meth:`solve`. Concrete implementations live in
    ``nrp.solvers.rl``, ``nrp.solvers.classical`` and ``nrp.solvers.hybrid``
    (added in wave 2 by sub-agent B).

    Attributes:
        name: Human-readable name of the solver.
        is_trainable: Whether the solver is trainable (i.e. has parameters that
            can be optimised).
        is_differentiable: Whether the solver's ``solve`` is differentiable
            w.r.t. its parameters.
    """

    name: str = "abstract"
    is_trainable: bool = False
    is_differentiable: bool = False

    def __init__(self, env, device: str | torch.device = "cpu", **kwargs):
        self.env = env
        self.device = torch.device(device)
        self.kwargs = kwargs

    @abstractmethod
    def solve(self, td: TensorDict) -> TensorDict:
        """Solve a batch of problem instances.

        Args:
            td: Input ``TensorDict`` describing a batch of problem instances.

        Returns:
            A ``TensorDict`` with at least ``actions`` (int64) and ``reward``
            (float) keys, with ``batch_size == actions.shape[:1]``.
        """
        raise NotImplementedError

    def warmup(self, td: TensorDict) -> None:
        """Best-effort warmup run on a single instance.

        Used by the evaluation harness to amortise the cost of CUDA kernel
        compilation and lazy initialisation. Errors are swallowed because some
        warmup operations are inherently best-effort.
        """
        try:
            self.solve(td[:1].clone())
        except Exception:
            pass

    def to(self, device: str | torch.device) -> Solver:
        """Move the solver to a device (in-place update of ``self.device``)."""
        self.device = torch.device(device)
        return self


class SolverRegistry:
    """Decorator-based registry for :class:`Solver` subclasses.

    Usage::

        @SolverRegistry.register("ortools_tsp", env_names=("tsp",))
        class ORToolsTSPSolver(ClassicalSolver):
            ...

    The class can be looked up later via :meth:`SolverRegistry.build` and
    filtered by environment support via :meth:`SolverRegistry.available`.
    """

    registry: dict[str, type[Solver]] = {}
    env_support: dict[str, tuple[str, ...]] = {}

    @classmethod
    def register(cls, name: str, env_names: tuple[str, ...] = ()):
        """Decorator that registers a :class:`Solver` subclass under ``name``.

        Args:
            name: String name to register the solver under.
            env_names: Tuple of environment names that the solver supports.
                An empty tuple means the solver is environment-agnostic
                (e.g. ``builtin_solve``).
        """

        def deco(klass: type[Solver]) -> type[Solver]:
            cls.registry[name] = klass
            cls.env_support[name] = tuple(env_names)
            klass._registered_name = name
            klass._supported_envs = tuple(env_names)
            return klass

        return deco

    @classmethod
    def build(cls, name: str, env, **kwargs) -> Solver:
        """Instantiate a registered solver by name.

        Args:
            name: Registered solver name.
            env: The environment to attach the solver to.
            **kwargs: Forwarded to the solver constructor.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        if name not in cls.registry:
            available = sorted(cls.registry.keys())
            raise ValueError(
                f"Unknown solver '{name}'. Available: {available}"
            )
        return cls.registry[name](env=env, **kwargs)

    @classmethod
    def available(cls, env_name: str | None = None) -> list[str]:
        """List available solver names, optionally filtered by environment.

        Args:
            env_name: If provided, only return solvers that either support
                no specific environment (empty tuple) or list this env name.
        """
        if env_name is None:
            return sorted(cls.registry.keys())
        return sorted(
            name
            for name, envs in cls.env_support.items()
            if not envs or env_name in envs
        )

    @classmethod
    def supports(cls, name: str, env_name: str) -> bool:
        """Whether a registered solver supports a given environment.

        An environment-agnostic solver (empty ``env_names``) supports any env.
        """
        envs = cls.env_support.get(name, ())
        return not envs or env_name in envs
