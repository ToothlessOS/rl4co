"""Per-environment metadata used by the nrp_eval pipeline.

Mirrors :data:`rl4co.envs.ENV_REGISTRY` and adds metadata needed by the
evaluation harness (``supports_improvement``, ``default_generator_params``,
``default_num_loc``).

Each entry in :data:`ENV_INFO` is a dict with the following keys:

- ``cls``: the env class.
- ``supports_improvement``: whether the env inherits
  :class:`rl4co.envs.common.base.ImprovementEnvBase`.
- ``default_generator_params``: a dict of safe default generator params.
- ``default_num_loc``: the canonical problem size (20) used by the
  configs in ``rl4co/configs/env``.
"""
from __future__ import annotations

import warnings
from typing import Any

# We deliberately import classes from their concrete modules (not via
# `get_env`) so that the registry carries class objects, not instances.

try:
    from rl4co.envs.common.base import ImprovementEnvBase
except Exception:  # pragma: no cover - extremely defensive
    ImprovementEnvBase = None  # type: ignore[assignment]

# Routing envs
try:
    from rl4co.envs.routing import ATSPEnv as _ATSPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping ATSPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _ATSPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import CVRPEnv as _CVRPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping CVRPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _CVRPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import CVRPMVCEnv as _CVRPMVCEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping CVRPMVCEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _CVRPMVCEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import CVRPTWEnv as _CVRPTWEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping CVRPTWEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _CVRPTWEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import MDCPDPEnv as _MDCPDPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping MDCPDPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _MDCPDPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import MTSPEnv as _MTSPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping MTSPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _MTSPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import MTVRPEnv as _MTVRPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping MTVRPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _MTVRPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import OPEnv as _OPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping OPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _OPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import PCTSPEnv as _PCTSPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping PCTSPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _PCTSPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import PDPEnv as _PDPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping PDPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _PDPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import PDPRuinRepairEnv as _PDPRuinRepairEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping PDPRuinRepairEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _PDPRuinRepairEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import SDVRPEnv as _SDVRPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping SDVRPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _SDVRPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import SPCTSPEnv as _SPCTSPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping SPCTSPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _SPCTSPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import SVRPEnv as _SVRPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping SVRPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _SVRPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import TSPEnv as _TSPEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping TSPEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _TSPEnv = None  # type: ignore[assignment]

try:
    from rl4co.envs.routing import TSPkoptEnv as _TSPkoptEnv
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Skipping TSPkoptEnv in nrp ENV_INFO: {exc}", stacklevel=2)
    _TSPkoptEnv = None  # type: ignore[assignment]


# Default problem size used by RL4CO's configs/env/*.yaml files.
DEFAULT_NUM_LOC = 20

# Safe default generator params per environment. These mirror the values
# used by RL4CO's reference configs and are good fallbacks for the harness.
_DEFAULT_GENERATOR_PARAMS: dict[str, dict[str, Any]] = {
    "tsp": {"num_loc": DEFAULT_NUM_LOC},
    "atsp": {"num_loc": DEFAULT_NUM_LOC},
    "cvrp": {"num_loc": DEFAULT_NUM_LOC},
    "cvrptw": {"num_loc": DEFAULT_NUM_LOC},
    "cvrpmvc": {"num_loc": DEFAULT_NUM_LOC},
    "sdvrp": {"num_loc": DEFAULT_NUM_LOC},
    "svrp": {"num_loc": DEFAULT_NUM_LOC},
    "pctsp": {"num_loc": DEFAULT_NUM_LOC},
    "spctsp": {"num_loc": DEFAULT_NUM_LOC},
    "op": {"num_loc": DEFAULT_NUM_LOC},
    "mtsp": {"num_loc": DEFAULT_NUM_LOC},
    "mtvrp": {"num_loc": DEFAULT_NUM_LOC},
    "pdp": {"num_loc": DEFAULT_NUM_LOC},
    "mdcpdp": {"num_loc": DEFAULT_NUM_LOC},
    "tsp_kopt": {"num_loc": DEFAULT_NUM_LOC},
    "pdp_ruin_repair": {"num_loc": DEFAULT_NUM_LOC},
}


def _supports_improvement(cls) -> bool:
    if cls is None or ImprovementEnvBase is None:
        return False
    try:
        return issubclass(cls, ImprovementEnvBase)
    except TypeError:
        return False


def _build_info(cls, name: str) -> dict[str, Any]:
    return {
        "cls": cls,
        "supports_improvement": _supports_improvement(cls),
        "default_generator_params": dict(_DEFAULT_GENERATOR_PARAMS.get(name, {})),
        "default_num_loc": DEFAULT_NUM_LOC,
    }


# Build ENV_INFO. Skip any env whose class failed to import.
_ENV_CANDIDATES: list[tuple[str, type | None]] = [
    ("tsp", _TSPEnv),
    ("atsp", _ATSPEnv),
    ("cvrp", _CVRPEnv),
    ("cvrptw", _CVRPTWEnv),
    ("cvrpmvc", _CVRPMVCEnv),
    ("sdvrp", _SDVRPEnv),
    ("svrp", _SVRPEnv),
    ("pctsp", _PCTSPEnv),
    ("spctsp", _SPCTSPEnv),
    ("op", _OPEnv),
    ("mtsp", _MTSPEnv),
    ("mtvrp", _MTVRPEnv),
    ("pdp", _PDPEnv),
    ("mdcpdp", _MDCPDPEnv),
    ("tsp_kopt", _TSPkoptEnv),
    ("pdp_ruin_repair", _PDPRuinRepairEnv),
]

ENV_INFO: dict[str, dict[str, Any]] = {
    name: _build_info(cls, name) for name, cls in _ENV_CANDIDATES if cls is not None
}


def list_envs() -> list[str]:
    """Return a sorted list of environment names registered in :data:`ENV_INFO`."""
    return sorted(ENV_INFO.keys())


def get_env_info(name: str) -> dict[str, Any]:
    """Look up the info dict for an environment by name.

    Raises:
        KeyError: With a helpful message listing available envs.
    """
    if name not in ENV_INFO:
        available = list_envs()
        raise KeyError(f"Unknown environment '{name}'. Available: {available}")
    return ENV_INFO[name]


def supports_improvement(name: str) -> bool:
    """Whether the named env supports iterative improvement (k-opt, ruin-repair)."""
    return bool(get_env_info(name).get("supports_improvement", False))
