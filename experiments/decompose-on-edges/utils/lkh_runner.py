"""Minimal LKH-2 wrapper for plain (symmetric Euclidean) TSP.

This is a self-contained, trimmed re-implementation of the LKH-2 plumbing
in ``experiments/POMO_eval/utils/lkh_tsp.py`` (which targets LKH-3). It
differs from that wrapper in three ways:

* No dependency on ``learn_decompose_eval`` or ``POMO_eval``. The
  ``.tsp`` / ``.par`` / ``.tour`` writers and parsers are inlined below.
* Operates on ``np.ndarray`` of shape ``(n, 2)`` rather than torch /
  TensorDict — the downstream pipeline (FI -> LKH -> alpha-nearness) is
  numpy end-to-end.
* Targets the stock LKH-2.0.11 binary; no CVRP / monster-tour / patching
  logic. The plain-TSP file format is identical between LKH-2 and LKH-3
  for the sections we use.

Usage::

    from utils.lkh_runner import solve_lkh_tsp
    perm, length = solve_lkh_tsp(coords)

Caveats (see the docstring of :func:`solve_lkh_tsp` for details):

* ``TIME_LIMIT = 0`` makes LKH-2 exit immediately with no tour, so we
  floor the value at 1 (memory ``lkh-time-limit-truncation``).
* LKH-2 writes a ``.tour`` file even on partial success. We don't trust
  the subprocess return code — only the existence of a non-empty
  ``.tour``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "LKH_SCALING_FACTOR",
    "DEFAULT_BINARY",
    "solve_lkh_tsp",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Multiplier applied to float distances before integerizing them into
#: the LKH ``EDGE_WEIGHT_SECTION``. Matches the convention used by the
#: POMO_eval LKH-3 wrapper and the rl4co mtvrp baseline. For uniform
#: ``[0, 1]`` coordinates with ``n <= 200`` the maximum scaled distance is
#: well within int32 range.
LKH_SCALING_FACTOR: int = 100_000

#: Default path to the bundled LKH-2.0.11 binary in this experiment.
DEFAULT_BINARY: str = (
    "/home/toothlessos/Projects/nrp/rl4co/experiments/"
    "decompose-on-edges/LKH-2.0.11/LKH"
)


# ---------------------------------------------------------------------------
# TSPLIB .tsp writer
# ---------------------------------------------------------------------------


def _pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    """``(n, n)`` Euclidean distance matrix; ``D[i, i] = 0``."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def _write_tsp(coords: np.ndarray, path: str, name: str = "inst") -> None:
    """Write a TSPLIB ``.tsp`` file for a plain Euclidean TSP.

    Format mirrors what LKH-2 and LKH-3 expect for ``TYPE : TSP`` with
    ``EXPLICIT / FULL_MATRIX``. Critical details:

    * ``NODE_COORD_TYPE : TWOD_COORDS`` is required even with
      ``EXPLICIT`` distances — LKH-2's ``ReadProblem`` checks this.
    * The ``EDGE_WEIGHT_SECTION`` body is **values only** — no leading
      row index per line. LKH-2's ``Read_EDGE_WEIGHT_SECTION`` for
      ``FULL_MATRIX`` consumes exactly ``n*n`` doubles via
      ``fscanf("%lf")``; a row index would shift the count and silently
      corrupt the matrix.

    Args:
        coords: ``(n, 2)`` Euclidean coordinates.
        path: Output ``.tsp`` path.
        name: ``NAME :`` field in the TSPLIB header.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    n = coords.shape[0]

    D = _pairwise_euclidean(coords)

    lines: list[str] = [
        f"NAME : {name}",
        "TYPE : TSP",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "NODE_COORD_TYPE : TWOD_COORDS",
        "",
        "NODE_COORD_SECTION",
    ]
    # 1-indexed node coordinates. ``.6f`` matches the POMO_eval writer.
    for i, (x, y) in enumerate(coords.tolist(), 1):
        lines.append(f"{i} {x:.6f} {y:.6f}")
    lines.append("")
    lines.append("EDGE_WEIGHT_SECTION")
    for i in range(n):
        row = " ".join(
            str(int(round(LKH_SCALING_FACTOR * D[i, j]))) for j in range(n)
        )
        lines.append(row)
    lines.append("")
    lines.append("EOF")
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# LKH .par writer
# ---------------------------------------------------------------------------


def _write_par(
    par_path: str,
    tsp_path: str,
    tour_path: str,
    *,
    max_trials: int = 10_000,
    seed: int = 1,
    time_limit_s: float = 30.0,
) -> None:
    """Write a minimal LKH-2 ``.par`` file for one plain-TSP run.

    Required keywords: ``PROBLEM_FILE``, ``OUTPUT_TOUR_FILE``, ``RUNS``,
    ``MAX_TRIALS``, ``TIME_LIMIT``, ``SEED``, ``TRACE_LEVEL``. The other
    LKH-2 defaults (e.g. ``CANDIDATE_SET_TYPE = ALPHA``,
    ``INITIAL_TOUR_ALGORITHM = WALK``) are fine for our use and not
    overridden.

    Args:
        par_path: Output ``.par`` path.
        tsp_path: Absolute path to the ``.tsp`` problem file.
        tour_path: Absolute path to the ``.tour`` output file.
        max_trials: ``MAX_TRIALS = <int>``; bounds the LKH-2 search.
        seed: ``SEED = <int>``; the LKH random seed.
        time_limit_s: Per-run time limit in seconds. ``LKH-2`` treats
            ``TIME_LIMIT = 0`` as "exit immediately", so the value is
            floored at ``max(1, int(round(time_limit_s)))`` (memory
            ``lkh-time-limit-truncation``).
    """
    # Floor TIME_LIMIT at 1: LKH-2 expects >= 0 and treats 0 as "no
    # time". Without this guard LKH-2 exits before producing a tour.
    safe_time_limit = max(1, int(round(time_limit_s)))

    body = "\n".join(
        [
            f"PROBLEM_FILE = {tsp_path}",
            f"OUTPUT_TOUR_FILE = {tour_path}",
            "RUNS = 1",
            f"MAX_TRIALS = {int(max_trials)}",
            f"TIME_LIMIT = {safe_time_limit}",
            f"SEED = {int(seed)}",
            "TRACE_LEVEL = 0",
            "",
        ]
    )
    Path(par_path).write_text(body)


# ---------------------------------------------------------------------------
# LKH-2 .tour parser
# ---------------------------------------------------------------------------


def _parse_tour(path: str) -> Optional[tuple[list[int], int]]:
    """Parse an LKH-2 ``.tour`` file.

    Reads the ``COMMENT : Length = X`` line for the scaled tour cost and
    the body after ``TOUR_SECTION`` for the 1-indexed node permutation.
    LKH-2's plain-TSP tour does NOT close the cycle explicitly — the
    implicit closing edge runs from the last listed node back to the
    first. The ``n``-length permutation is therefore a valid TSP tour
    for RL4CO's ``actions`` convention.

    Args:
        path: Path to the ``.tour`` file.

    Returns:
        ``(perm_1idx, length_scaled)`` on success, ``None`` if the file
        is missing or cannot be parsed. ``perm_1idx`` is length ``n``
        and is a permutation of ``1..n`` (1-indexed).
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return None

    cost_scaled: int | None = None
    dimension: int | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("DIMENSION"):
            try:
                dimension = int(s.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif s.startswith("COMMENT") and "Length" in s:
            # "COMMENT : Length = 12345"
            try:
                cost_scaled = int(s.rsplit("=", 1)[1].strip())
            except (ValueError, IndexError):
                pass

    tour_idx = text.find("TOUR_SECTION")
    if tour_idx < 0 or cost_scaled is None or dimension is None:
        return None
    body = text[tour_idx:].split("\n", 1)[1]

    perm: list[int] = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s in ("EOF",):
            continue
        try:
            nid = int(s.split()[0])
        except ValueError:
            continue
        if nid == -1:
            break
        perm.append(nid)
    if len(perm) != dimension:
        return None
    return perm, cost_scaled


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def solve_lkh_tsp(
    coords: np.ndarray,
    binary_path: str = DEFAULT_BINARY,
    *,
    max_trials: int = 10_000,
    seed: int = 1,
    time_limit_s: float = 30.0,
    timeout_s: int = 120,
    tmpdir: Optional[str] = None,
) -> tuple[Optional[list[int]], float]:
    """Solve a single Euclidean TSP instance with the LKH-2 binary.

    Writes a ``.tsp`` + ``.par`` to a temporary directory, invokes
    ``binary_path <par>`` as a subprocess, and parses the resulting
    ``.tour``. Returns a 0-indexed tour permutation and the float tour
    length in the input units.

    Args:
        coords: ``(n, 2)`` Euclidean coordinates on any scale (the
            canonical convention is ``[0, 1]``).
        binary_path: Absolute path to the LKH-2 executable.
        max_trials: ``MAX_TRIALS`` in the ``.par`` file.
        seed: ``SEED`` in the ``.par`` file.
        time_limit_s: Per-run ``TIME_LIMIT``. Floored at ``1`` internally.
        timeout_s: Hard wall-clock cap on the subprocess. Defaults to
            ``120 s``.
        tmpdir: If provided, write the ``.tsp`` / ``.par`` / ``.tour``
            files here; the caller is responsible for cleanup.
            Otherwise a ``tempfile.mkdtemp`` is used and cleaned up
            before return.

    Returns:
        ``(perm_0idx, length)``. ``perm_0idx`` is a length-``n`` list
        containing a permutation of ``0..n-1``. ``length`` is the
        Euclidean tour length in the same units as ``coords``. On
        failure (LKH-2 timeout, missing or empty ``.tour``, parse
        failure), returns ``(None, float("inf"))``.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    n = coords.shape[0]
    if not Path(binary_path).is_file():
        raise FileNotFoundError(f"LKH binary not found: {binary_path}")

    cleanup = tmpdir is None
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix=f"doe_lkh_{n}_")
    else:
        os.makedirs(tmpdir, exist_ok=True)
    tsp_path = os.path.join(tmpdir, "instance.tsp")
    tour_path = os.path.join(tmpdir, "instance.tour")
    par_path = os.path.join(tmpdir, "instance.par")

    try:
        _write_tsp(coords, tsp_path, name="inst")
        _write_par(
            par_path,
            tsp_path,
            tour_path,
            max_trials=max_trials,
            seed=seed,
            time_limit_s=time_limit_s,
        )

        try:
            subprocess.run(
                [binary_path, par_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, float("inf")

        tour_p = Path(tour_path)
        if not tour_p.exists() or tour_p.stat().st_size == 0:
            return None, float("inf")

        parsed = _parse_tour(tour_path)
        if parsed is None:
            return None, float("inf")

        perm_1idx, cost_scaled = parsed
        perm_0idx = [p - 1 for p in perm_1idx]
        length = cost_scaled / LKH_SCALING_FACTOR
        return perm_0idx, length
    finally:
        if cleanup:
            shutil.rmtree(tmpdir, ignore_errors=True)