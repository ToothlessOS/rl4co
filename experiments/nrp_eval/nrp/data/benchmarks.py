"""TSPLIB / CVRPLIB benchmark loaders.

Stage 1 ships stubs that raise ``NotImplementedError`` with a clear pointer
to the planned stage-2 API. Real implementations will live here in stage 2
and use ``vrplib.read_instance(path)`` to parse the standard formats.
"""
from __future__ import annotations

import warnings
from pathlib import Path


def load_tsplib(path: str | Path) -> dict:
    """Load a TSPLIB instance (``.tsp`` file).

    Stage 1 stub: raises ``NotImplementedError``. Stage 2 will return a dict
    with at least ``"node_coord"`` (an ``(N, 2)`` ndarray) and convert to a
    :class:`TensorDict` consumable by ``TSPEnv``.
    """
    warnings.warn(
        "TSPLIB loading is a stage-2 feature. Install `vrplib` and see "
        "experiments/nrp_eval/scripts/install_classical_solvers.sh for the planned API.",
        stacklevel=2,
    )
    raise NotImplementedError(
        "TSPLIB loading is deferred to stage 2. "
        "See nrp/data/benchmarks.py for the planned interface."
    )


def load_cvrplib(path: str | Path) -> dict:
    """Load a CVRPLIB instance (``.vrp`` file). Stage 1 stub."""
    raise NotImplementedError("CVRPLIB loading is deferred to stage 2.")


def load_tsplib_optimum(path: str | Path) -> list[int]:
    """Load a TSPLIB optimum tour (``.opt.tour`` or ``.tour`` file). Stage 1 stub."""
    raise NotImplementedError("TSPLIB optimum loading is deferred to stage 2.")
