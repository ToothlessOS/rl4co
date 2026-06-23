"""LKH-3 wrapper for plain (symmetric Euclidean) TSP.

Reuses the stdlib-only building blocks from
``learn_decompose_eval.solvers.lkh_format`` (``LKHParameters``,
``write_lkh_problem``, ``parse_tour_with_cost``, ``LKH_SCALING_FACTOR``) and
``learn_decompose_eval.solvers.classical_lkh`` (``_resolve_lkh_binary``).
The LDE writer ``cvrp_td_to_lkh_problem`` is CVRP-only and is *not* used
here; we emit a TSPLIB ``TYPE : TSP`` problem string ourselves.

Required environment (matching the existing ``learn_decompose_eval`` CLI
convention)::

    export PYTHONPATH=/home/toothlessos/Projects/nrp/rl4co/experiments/learn_decompose_eval:${PYTHONPATH:-}
    export NRP_LKH_BINARY=/home/toothlessos/Projects/nrp/rl4co/experiments/POMO_eval/LKH-3.0.14/LKH
    source /home/toothlessos/Projects/nrp/rl4co/.venv/bin/activate

Caveats (also documented inline):

* The bundled LKH-3.0.14 binary is the *stock* upstream build (no
  ``INTERMEDIATE_TOUR_FILE`` patch). We never set that field; the LDE
  ``LKHParameters`` writer omits the line when the value is ``None``, so
  stock LKH-3 just ignores it.
* ``LKHParameters.to_par_string()`` floors ``TIME_LIMIT`` at 1 (see
  [[lkh-time-limit-truncation]]). To mean "no time cap" we pass a huge
  ``time_limit_s`` and rely on ``MAX_TRIALS`` to bound the work.
* On a 100-node unit-square Euclidean TSP, expected tour length is in
  the rough range 8-12 (a tour of 100 random points in the unit square
  has expected length ~ sqrt(100 * 0.5) * 0.9 ~ 7-9 for an optimal tour;
  heuristic solutions are slightly worse).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tensordict import TensorDict

# LDE building blocks. Imported at module load with a try/except so a
# missing PYTHONPATH fails loudly at the first LKH-3 call (not at import).
def _import_lde():
    """Import the LDE building blocks, bootstrapping ``sys.path`` if needed.

    The caller is expected to either export ``PYTHONPATH=.../learn_decompose_eval``
    (the LDE convention) or invoke this module from a script whose working
    directory is a sibling of ``learn_decompose_eval`` (the POMO_eval
    convention). We try the explicit PYTHONPATH route first, then fall
    back to deriving the path from this file's location — so the module
    "just works" regardless of the user's shell setup.
    """
    try:
        from learn_decompose_eval.solvers.lkh_format import (  # type: ignore
            LKHParameters,
            write_lkh_problem,
            parse_tour_with_cost,
            LKH_SCALING_FACTOR,
        )
        from learn_decompose_eval.solvers.classical_lkh import (  # type: ignore
            _resolve_lkh_binary,
        )
        return (
            LKHParameters,
            write_lkh_problem,
            parse_tour_with_cost,
            LKH_SCALING_FACTOR,
            _resolve_lkh_binary,
            None,
        )
    except ImportError as err:
        # Fallback: derive the LDE root from this file's path. This file lives at
        # ``<repo>/experiments/POMO_eval/utils/lkh_tsp.py``; LDE lives at the
        # sibling ``<repo>/experiments/learn_decompose_eval/``.
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent.parent / "learn_decompose_eval",  # sibling experiment dir
        ]
        for cand in candidates:
            if cand.is_dir():
                sys.path.insert(0, str(cand))
        try:
            from learn_decompose_eval.solvers.lkh_format import (  # type: ignore
                LKHParameters,
                write_lkh_problem,
                parse_tour_with_cost,
                LKH_SCALING_FACTOR,
            )
            from learn_decompose_eval.solvers.classical_lkh import (  # type: ignore
                _resolve_lkh_binary,
            )
            return (
                LKHParameters,
                write_lkh_problem,
                parse_tour_with_cost,
                LKH_SCALING_FACTOR,
                _resolve_lkh_binary,
                None,
            )
        except ImportError as err2:
            return None, None, None, 100_000, None, err2


(
    LKHParameters,
    write_lkh_problem,
    parse_tour_with_cost,
    LKH_SCALING_FACTOR,
    _resolve_lkh_binary,
    _LDE_IMPORT_ERROR,
) = _import_lde()
_LDE_AVAILABLE = _resolve_lkh_binary is not None


def _require_lde():
    """Raise a helpful error if LDE is not on PYTHONPATH."""
    if not _LDE_AVAILABLE:
        raise ImportError(
            "learn_decompose_eval.solvers is not importable. Set "
            "PYTHONPATH=/home/toothlessos/Projects/nrp/rl4co/experiments/"
            f"learn_decompose_eval. Original error: {_LDE_IMPORT_ERROR}"
        )
    return (
        LKHParameters,
        write_lkh_problem,
        parse_tour_with_cost,
        LKH_SCALING_FACTOR,
        _resolve_lkh_binary,
    )


__all__ = [
    "tsp_td_to_lkh_problem",
    "parse_tsp_tour",
    "solve_lkh3_tsp_one",
    "solve_lkh3_tsp_batch",
    "_resolve_lkh_binary",
]


# ---------------------------------------------------------------------------
# TSPLIB TSP problem writer
# ---------------------------------------------------------------------------


def tsp_td_to_lkh_problem(
    td: TensorDict, name: str = "inst", scaling_factor: int = 100_000
) -> str:
    """Convert a single TSP instance to a TSPLIB problem string.

    Mirrors ``learn_decompose_eval.solvers.lkh_format.cvrp_td_to_lkh_problem``
    (lkh_format.py:35-110) but emits ``TYPE : TSP`` and omits all CVRP-only
    sections (``CAPACITY``, ``DEMAND_SECTION``, ``DEPOT_SECTION``). The
    ``EDGE_WEIGHT_SECTION`` is written as values only — no leading row
    index on each line — because LKH-3's ``Read_EDGE_WEIGHT_SECTION`` for
    ``FULL_MATRIX`` consumes exactly ``n*n`` doubles via ``fscanf("%lf")``
    (see [[lkh-3-patch-conventions]]). ``NODE_COORD_TYPE : TWOD_COORDS``
    is required when using ``EXPLICIT / FULL_MATRIX``.

    Args:
        td: ``TensorDict`` with a ``"locs"`` key of shape ``[n, 2]``
            (Euclidean coordinates, any scale; the canonical RL4CO
            convention is ``[0, 1]``).
        name: ``NAME : <name>`` in the TSPLIB header.
        scaling_factor: Distances are multiplied by this and rounded to
            ints; LKH-3 reads them as doubles and multiplies back.

    Returns:
        A single string in TSPLIB ``.tsp`` format, ready to be written
        verbatim to disk for LKH-3 to consume.
    """
    locs = td["locs"]
    if locs.dim() != 2 or locs.shape[-1] != 2:
        raise ValueError(
            f"locs must have shape [n, 2], got {tuple(locs.shape)}"
        )
    n = locs.shape[0]

    # Pairwise Euclidean distance matrix.
    d = torch.cdist(locs, locs, p=2.0)

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
    # 1-indexed node coords.
    for i, (x, y) in enumerate(locs.tolist(), 1):
        lines.append(f"{i} {x:.6f} {y:.6f}")
    lines.append("")
    lines.append("EDGE_WEIGHT_SECTION")
    # Values only — no leading "i" (would shift the fscanf("%lf") count).
    for i in range(n):
        row = " ".join(
            str(int(round(scaling_factor * d[i, j].item()))) for j in range(n)
        )
        lines.append(row)
    lines.append("")
    lines.append("EOF")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LKH-3 TSP tour file parser (plain TSP, no depot)
# ---------------------------------------------------------------------------


def parse_tsp_tour(path: str) -> tuple[list[int], int, int] | None:
    """Parse a stock LKH-3 TSP ``.tour`` file.

    Unlike ``learn_decompose_eval.solvers.lkh_format.parse_tour_with_cost``,
    which assumes a depot-bracketed tour (CVRP monster tour with phantom
    depots, or legacy TSP where the tour closes with the depot id), this
    reads plain LKH-3 TSP output, where the file lists the ``n`` nodes in
    cycle order and ends with ``-1``. The cycle is implicit: the
    permutation is just the list of node ids in the order LKH-3 wrote
    them, and the implicit closing edge runs from the last node back to
    the first.

    Returns ``(permutation_1idx, cost_scaled, dimension)`` or ``None`` on
    parse failure. ``permutation_1idx`` is length ``n`` and is a
    permutation of ``1..n`` (the writer uses 1-indexed node ids).
    """
    try:
        with open(path) as f:
            text = f.read()
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
        if not s or s in ("-1", "EOF"):
            if s == "-1":
                break
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
    return perm, cost_scaled, dimension


# ---------------------------------------------------------------------------
# Single-instance LKH-3 invocation
# ---------------------------------------------------------------------------


def solve_lkh3_tsp_one(
    locs_2d: torch.Tensor,
    *,
    name: str,
    binary_path: str,
    max_trials: int = 10_000_000,
    seed: int = 1,
    tmpdir: str | None = None,
    timeout_s: int = 600,
) -> tuple[list[int] | None, float]:
    """Solve a single Euclidean TSP instance with the LKH-3 binary.

    Writes a ``.tsp`` + ``.par`` to a temp directory, invokes the binary
    (with a hard wall-clock ``timeout_s`` safety net), and parses the
    output ``.tour`` file. Returns the 0-indexed tour permutation and
    the tour length in the same units as ``locs_2d`` (Euclidean,
    unscaled).

    Args:
        locs_2d: ``[n, 2]`` Euclidean coordinates on any device.
        name: Instance name; used for the temp-dir prefix and the
            TSPLIB ``NAME :`` field.
        binary_path: Absolute path to the LKH-3 executable. Pass
            ``_resolve_lkh_binary()``'s output.
        max_trials: ``MAX_TRIALS = <this>`` in the ``.par`` file.
        seed: ``SEED = <this>`` in the ``.par`` file.
        tmpdir: If provided, write the ``.tsp`` / ``.par`` / ``.tour``
            here (caller manages cleanup). Otherwise a
            ``tempfile.mkdtemp`` is used and cleaned up before return.
        timeout_s: Wall-clock safety net for the subprocess call. The
            binary is invoked with a per-run ``TIME_LIMIT`` of 999_999
            seconds (effectively uncapped) but we still want a
            hard fallback in case the binary hangs.

    Returns:
        ``(perm_0idx, length)``. On parse failure, returns
        ``(None, float("inf"))``. ``perm_0idx`` is a list of length
        ``n`` containing a permutation of ``0..n-1``. ``length`` is in
        the same Euclidean units as ``locs_2d``.
    """
    LKHParameters, write_lkh_problem, _, LKH_SCALING_FACTOR, _ = _require_lde()

    if locs_2d.dim() != 2 or locs_2d.shape[-1] != 2:
        raise ValueError(
            f"locs_2d must have shape [n, 2], got {tuple(locs_2d.shape)}"
        )
    n = locs_2d.shape[0]

    cleanup = tmpdir is None
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix=f"pomo_lkh_{name}_")
    else:
        os.makedirs(tmpdir, exist_ok=True)
    problem_path = os.path.join(tmpdir, "instance.tsp")
    tour_path = os.path.join(tmpdir, "instance.tour")
    par_path = os.path.join(tmpdir, "instance.par")

    try:
        td = TensorDict({"locs": locs_2d.detach().cpu()}, batch_size=[n])
        write_lkh_problem(problem_path, tsp_td_to_lkh_problem(td, name=name))
        LKHParameters(
            problem_file=problem_path,
            output_tour_file=tour_path,
            runs=1,
            max_trials=max_trials,
            time_limit_s=999_999.0,  # effectively uncapped; MAX_TRIALS bounds it
            seed=seed,
            trace_level=0,
        ).write(par_path)

        try:
            subprocess.call(
                [binary_path, par_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, float("inf")

        if not Path(tour_path).exists() or Path(tour_path).stat().st_size == 0:
            return None, float("inf")

        parsed = parse_tsp_tour(tour_path)
        if parsed is None:
            return None, float("inf")
        perm_1idx, cost_scaled, _dim = parsed
        # LKH-3 TSP output: nodes in cycle order, no closing repeat.
        # The implicit closing edge runs from the last node back to the
        # first, so the n-length permutation is already a valid TSP tour
        # for RL4CO's ``actions`` format. Convert 1-indexed -> 0-indexed.
        perm_0idx = [p - 1 for p in perm_1idx]
        length = cost_scaled / LKH_SCALING_FACTOR
        return perm_0idx, length
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Batched LKH-3 invocation
# ---------------------------------------------------------------------------


def solve_lkh3_tsp_batch(
    locs_batch: torch.Tensor,
    *,
    binary_path: str | None = None,
    max_trials: int = 10_000_000,
    seed: int = 1,
    n_workers: int = 4,
) -> tuple[list[list[int] | None], list[float]]:
    """Solve a batch of TSP instances with LKH-3 in parallel.

    Uses a ``ThreadPoolExecutor`` because ``subprocess.call`` releases
    the GIL while the child process is running — threads are lighter
    than processes and avoid the cost of re-importing torch in each
    worker. The LDE orchestrator uses the same pattern.

    Args:
        locs_batch: ``[B, n, 2]`` Euclidean coordinates.
        binary_path: Absolute path to the LKH-3 binary. If ``None``,
            resolves via ``_resolve_lkh_binary()`` (which honors
            ``LDE_LKH_BINARY`` / ``NRP_LKH_BINARY`` env vars).
        max_trials: Per-instance ``MAX_TRIALS``.
        seed: Per-instance ``SEED`` (kept identical for reproducibility;
            a future version could mix the batch index into the seed).
        n_workers: Thread-pool size.

    Returns:
        ``(perms, lengths)`` — both length-``B``. Failed instances have
        ``perms[i] is None`` and ``lengths[i] == float("inf")``.
    """
    _, _, _, _, _resolve_lkh_binary = _require_lde()
    if binary_path is None:
        binary_path = _resolve_lkh_binary(None)
    if locs_batch.dim() != 3 or locs_batch.shape[-1] != 2:
        raise ValueError(
            f"locs_batch must have shape [B, n, 2], got {tuple(locs_batch.shape)}"
        )
    B = locs_batch.shape[0]
    locs_cpu = locs_batch.detach().cpu()

    def _one(b: int) -> tuple[list[int] | None, float]:
        return solve_lkh3_tsp_one(
            locs_cpu[b],
            name=f"b{b:03d}",
            binary_path=binary_path,
            max_trials=max_trials,
            seed=seed,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_one, range(B)))

    perms = [r[0] for r in results]
    lengths = [r[1] for r in results]
    return perms, lengths
