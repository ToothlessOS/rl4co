"""TSP dataset parsers.

Three file shapes are supported:

1. **Train/test .txt** (e.g. ``train_TSP100_n100w-002.txt``,
   ``test_TSP100_n1w.txt``). Each line: ``x1 y1 x2 y2 ... xn yn output
   tour1 tour2 ... tourn``. Coordinates are floats; tours are 1-indexed
   integers (we convert to 0-indexed). The token ``output`` separates the
   coordinate block from the tour block. ``n`` is derived as
   ``line.index("output") // 2``.

2. **Variable-n test .txt** (e.g. ``test_TSP200_n128.txt``). Same format as
   above; ``n`` varies per line.

3. **TSPlib .txt** (e.g. ``TSPlib_70instances.txt``). Each line is a Python
   list literal ``['name', cost, x1, y1, x2, y2, ...]``. ``opt_tour`` is
   not provided — only the optimal cost is known. Used for transfer eval.

References: ``ref/env/TSPEnv.py::load_raw_data`` (line format),
``ref/env/TSPEnv_inTSPlib.py::make_tsplib_data`` (TSPlib format).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TSPInstance:
    """A single TSP instance.

    Attributes:
        coords: ``(n, 2)`` float64 Euclidean coordinates.
        opt_tour: ``(n,)`` int64 0-indexed permutation, or ``None`` when
            only the optimal cost is known (TSPlib).
        name: human-readable instance identifier; empty string if anonymous.
    """

    coords: np.ndarray
    opt_tour: np.ndarray | None
    name: str = ""


def _parse_train_line(line: str) -> TSPInstance:
    """Parse one line of the train/test file format.

    Mirrors ``ref/env/TSPEnv.py::load_raw_data`` line parsing, but more
    defensive: strips trailing whitespace, handles possible blank lines.
    """
    tokens = line.split()
    # The token 'output' separates coords from tour. Use the first occurrence.
    try:
        out_idx = tokens.index("output")
    except ValueError as e:
        raise ValueError(
            f"Line is missing the 'output' separator: {line[:80]}..."
        ) from e
    if out_idx % 2 != 0:
        raise ValueError(
            f"Coordinate block has odd length {out_idx} on line: {line[:80]}..."
        )
    n = out_idx // 2
    if n < 2:
        raise ValueError(f"Instance has n={n} < 2 on line: {line[:80]}...")

    coords = np.asarray(tokens[:out_idx], dtype=np.float64).reshape(n, 2)
    # Tour tokens are after 'output'. Strip a possible trailing newline artifact.
    tour_tokens = tokens[out_idx + 1:]
    if len(tour_tokens) < n:
        raise ValueError(
            f"Tour has only {len(tour_tokens)} tokens, expected {n}: "
            f"{line[:80]}..."
        )
    # 1-indexed → 0-indexed
    opt_tour = np.asarray(tour_tokens[:n], dtype=np.int64) - 1
    return TSPInstance(coords=coords, opt_tour=opt_tour)


def load_train_file(path: str | Path, max_instances: int | None = None) -> list[TSPInstance]:
    """Parse a train/test .txt file (fixed n).

    Stops after ``max_instances`` instances if given. Skips blank lines.
    """
    path = Path(path)
    instances: list[TSPInstance] = []
    with path.open("r") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                inst = _parse_train_line(line)
            except ValueError as e:
                # Defensive: skip malformed lines instead of crashing.
                # The training set is large; one bad line is acceptable.
                # We log via stderr via a print so it's visible in logs.
                print(f"[load_train_file] {path.name}:{line_no}: skip ({e})")
                continue
            instances.append(inst)
            if max_instances is not None and len(instances) >= max_instances:
                break
    return instances


def load_test_file(path: str | Path, max_instances: int | None = None) -> list[TSPInstance]:
    """Parse a test .txt file.

    Same as ``load_train_file`` but kept as a separate symbol for clarity
    at the call site (the file format is identical).
    """
    return load_train_file(path, max_instances=max_instances)


def load_tsplib_file(path: str | Path, max_instances: int | None = None) -> list[TSPInstance]:
    """Parse a TSPlib .txt file.

    Each line is a Python list literal: ``['name', cost, x1, y1, x2, y2, ...]``.
    ``opt_tour`` is set to ``None`` because the tour is not stored.
    """
    import ast

    path = Path(path)
    instances: list[TSPInstance] = []
    with path.open("r") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = ast.literal_eval(line)
            except (ValueError, SyntaxError) as e:
                print(f"[load_tsplib_file] {path.name}:{line_no}: skip ({e})")
                continue
            if len(parsed) < 4:
                print(
                    f"[load_tsplib_file] {path.name}:{line_no}: skip "
                    f"(too few tokens: {len(parsed)})"
                )
                continue
            name = str(parsed[0])
            # parsed[1] is the optimal cost (unused by the trainer; recorded
            # in ``name`` for traceability, e.g. ``"a280@2579"``).
            try:
                cost = float(parsed[1])
            except (TypeError, ValueError):
                cost = float("nan")
            coords_flat = np.asarray(parsed[2:], dtype=np.float64)
            if coords_flat.size % 2 != 0:
                print(
                    f"[load_tsplib_file] {path.name}:{line_no}: skip "
                    f"(odd coord count: {coords_flat.size})"
                )
                continue
            n = coords_flat.size // 2
            coords = coords_flat.reshape(n, 2)
            instances.append(
                TSPInstance(coords=coords, opt_tour=None, name=f"{name}@{cost:g}")
            )
            if max_instances is not None and len(instances) >= max_instances:
                break
    return instances