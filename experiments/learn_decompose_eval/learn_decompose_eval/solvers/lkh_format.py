"""TSPLIB CVRP format conversion.

Converts between RL4CO CVRP TensorDict instances and the LKH-3 input/output
file formats. Reference: `rl4co/rl4co/envs/routing/mtvrp/baselines/lkh.py`.

We use EDGE_WEIGHT_TYPE: EXPLICIT / EDGE_WEIGHT_FORMAT: FULL_MATRIX with a
scaling factor of 100_000 (matching mtvrp/baselines/constants.py). This is
the only safe way to feed LKH-3 from a Python environment that samples
coordinates in [0, 1] without risking int32 overflow in squared distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from tensordict import TensorDict

# Scaling factor to convert from floating-point coordinates/demands to the
# integer representation LKH-3 expects. With coords in [0, 1] and demand in
# [1, 9] and capacity up to ~150, the max scaled value is ~1.5e7, well
# within int32 range (2.1e9). The max squared distance is ~1e14, which DOES
# overflow int32 — hence we must use the EXPLICIT matrix form (precomputed
# distances) rather than EUC_2D.
LKH_SCALING_FACTOR = 100_000


# ---------------------------------------------------------------------------
# RL4CO TensorDict -> TSPLIB CVRP problem string
# ---------------------------------------------------------------------------


def cvrp_td_to_lkh_problem(
    td: TensorDict,
    name: str = "cvrp",
    scaling_factor: int = LKH_SCALING_FACTOR,
) -> str:
    """Build a TSPLIB CVRP problem string from a single-instance TensorDict.

    The TensorDict must be 1-D (a single instance), with keys
    ``depot``, ``locs``, ``demand``, ``capacity``. ``locs`` may either be
    ``[N, 2]`` (no depot) or ``[N+1, 2]`` (depot at index 0) — both are
    handled.

    Returns a TSPLIB CVRP string in the **Uchoa convention** (used by the
    Uchoa CVRP benchmark instances and accepted by LKH-3's CVRP solver):
        - DIMENSION = N + 1 (1 depot + N customers)
        - depot id = 1
        - customer ids = 2..N+1
    """
    if td.batch_size != torch.Size([]) and len(td.batch_size) > 1:
        raise ValueError(f"Expected a single-instance TensorDict, got {td.batch_size}")

    if "locs" in td.keys() and "depot" in td.keys():
        locs = td["locs"]
        if locs.shape[-2] == td["demand"].shape[-1] + 1:
            # depot already prepended at index 0
            depot_xy = locs[0]
            customer_xy = locs[1:]
        else:
            depot_xy = td["depot"].reshape(2)
            customer_xy = locs
    else:
        raise KeyError("TensorDict must have 'locs' and 'depot' keys")

    n = customer_xy.shape[0]
    # Uchoa ordering: depot first (id 1), then customers (ids 2..N+1).
    all_locs = torch.cat([depot_xy.unsqueeze(0), customer_xy], dim=0)  # [N+1, 2]

    demand_norm = td["demand"].reshape(-1)
    capacity = float(td["capacity"].reshape(()).item())
    demand_int = (demand_norm * capacity).round().long()  # [N]

    coords = (all_locs * scaling_factor).round().long()  # [N+1, 2]

    diffs = all_locs.unsqueeze(0) - all_locs.unsqueeze(1)
    dmat = torch.sqrt((diffs * diffs).sum(-1)) * scaling_factor
    dmat = dmat.round().long()

    lines: list[str] = []
    lines.append(f"NAME : {name}")
    lines.append("TYPE : CVRP")
    lines.append(f"DIMENSION : {n + 1}")
    lines.append("EDGE_WEIGHT_TYPE : EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append("NODE_COORD_TYPE : TWOD_COORDS")
    lines.append(f"CAPACITY : {int(capacity)}")
    lines.append("")
    lines.append("NODE_COORD_SECTION")
    for i in range(n + 1):
        x, y = int(coords[i, 0].item()), int(coords[i, 1].item())
        lines.append(f"{i + 1}\t{x}\t{y}")
    lines.append("")
    lines.append("DEMAND_SECTION")
    lines.append("1\t0")
    for i, d in enumerate(demand_int.tolist(), 2):
        lines.append(f"{i}\t{d}")
    lines.append("")
    lines.append("EDGE_WEIGHT_SECTION")
    for i in range(n + 1):
        row = " ".join(str(int(dmat[i, j].item())) for j in range(n + 1))
        lines.append(row)
    lines.append("")
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")
    return "\n".join(lines)


def write_lkh_problem(path: str, problem_str: str) -> None:
    """Write a TSPLIB problem string to disk."""
    with open(path, "w") as f:
        f.write(problem_str)


# ---------------------------------------------------------------------------
# LKH-3 tour file parser
# ---------------------------------------------------------------------------


def parse_lkh_tour(
    path: str,
    depot_id: int = 1,
) -> list[list[int]]:
    """Parse a TSPLIB tour file into routes (1-indexed).

    The depot (id = ``depot_id``, default 1 in the Uchoa CVRP convention)
    appears at route boundaries. Routes are 1-indexed: ``[1, c1, c2, 1]``,
    ``[1, c3, c4, 1]``, etc.
    """
    with open(path, "r") as f:
        text = f.read()

    tour_idx = text.find("TOUR_SECTION")
    if tour_idx < 0:
        raise ValueError(f"No TOUR_SECTION found in {path}")
    body = text[tour_idx:].split("\n", 1)[1]

    nodes: list[int] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line in ("-1", "EOF"):
            if line == "-1":
                break
            continue
        nodes.append(int(line.split()[0]))

    routes: list[list[int]] = []
    cur: list[int] = []
    for nid in nodes:
        cur.append(nid)
        if nid == depot_id and len(cur) > 1:
            routes.append(cur)
            cur = []
    if cur:
        if cur[-1] != depot_id:
            cur.append(depot_id)
        routes.append(cur)
    return routes


def parse_tour_with_cost(
    path: str, depot_id: int = 1
) -> tuple[list[list[int]], int, int] | None:
    """Parse a TSPLIB tour file into (routes, cost_scaled, dimension).

    The cost is parsed from the ``COMMENT : Length = X`` line in the
    header.  For feasible tours (the only case the orchestrator
    compares) LKH-3 writes the *unscaled* integer edge cost multiplied
    by ``LKH_SCALING_FACTOR`` — divide by the scaling factor (default
    100_000) to get the float cost in the same units RL4CO's
    ``CVRPEnv.get_reward`` returns.

    For tours with a non-zero capacity penalty, LKH-3 writes
    ``COMMENT : Cost = P_C`` (penalty and cost).  This function returns
    only the cost component ``C``; the penalty is not currently
    captured.  Callers that need to handle infeasible tours should
    re-evaluate via ``CVRPEnv.get_reward``.

    The dimension is parsed from the ``DIMENSION : X`` line.  For
    CVRP, this is ``num_loc + Salesmen`` (LKH-3 expands the
    problem's dimension by ``Salesmen - 1`` to account for phantom
    depots; see ``LKH-3.0.14/SRC/MTSP2TSP.c:37``).

    Returns ``None`` if the file cannot be parsed.
    """
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError:
        return None

    # Parse header fields.
    cost_scaled: int | None = None
    dimension: int | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("DIMENSION"):
            try:
                dimension = int(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("COMMENT") and "Length" in line:
            # COMMENT : Length = XXXXX
            try:
                cost_scaled = int(line.rsplit("=", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("COMMENT") and line.startswith("COMMENT : Cost ="):
            # COMMENT : Cost = P_C
            try:
                cost_scaled = int(line.rsplit("_", 1)[1].strip())
            except (ValueError, IndexError):
                pass

    # Parse the body (same logic as parse_lkh_tour, but inline since
    # we already have the text).
    tour_idx = text.find("TOUR_SECTION")
    if tour_idx < 0 or cost_scaled is None or dimension is None:
        return None
    body = text[tour_idx:].split("\n", 1)[1]
    nodes: list[int] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line in ("-1", "EOF"):
            if line == "-1":
                break
            continue
        try:
            nodes.append(int(line.split()[0]))
        except ValueError:
            continue
    routes: list[list[int]] = []
    cur: list[int] = []
    for nid in nodes:
        cur.append(nid)
        if nid == depot_id and len(cur) > 1:
            routes.append(cur)
            cur = []
    if cur:
        if cur[-1] != depot_id:
            cur.append(depot_id)
        routes.append(cur)
    return routes, cost_scaled, dimension


def write_cvrp_initial_tour(
    path: str,
    customer_seq_1indexed: Sequence[int],
    num_loc: int,
    salesmen: int,
    name: str = "stitched",
) -> None:
    """Write a TSPLIB TOUR file in CVRP format with phantom depots.

    The output matches the master's expanded ``DimensionSaved =
    num_loc + salesmen`` so LKH-3's CVRP ``ReadTour`` will accept
    it.  Phantom depots have ids ``num_loc + 2, num_loc + 3, ...,
    num_loc + salesmen`` (one per additional vehicle after the
    first) and are inserted at route boundaries.

    Body layout::

        depot(1) + (customers + phantom_depot) per route
                  ... + last route's customers (no trailing depot)

    Customers are distributed **evenly** across ``salesmen`` routes:
    each route gets ``num_loc // salesmen`` customers, with the last
    route getting the remainder.  This is a heuristic — the master
    LKH-3 will re-partition based on capacity anyway, so we just
    need a valid permutation of all customers that LKH-3 can read.

    Args:
        path: where to write the .tour file.
        customer_seq_1indexed: a permutation of ``[2..num_loc+1]``
            (Uchoa customer ids).  May be a list or any Sequence.
        num_loc: number of customers in the problem.
        salesmen: number of vehicles the master will use.  Must be
            >= 1.  ``salesmen=1`` produces a single-route tour
            with no phantom depots (DIMENSION = num_loc + 1).
        name: NAME field value (defaults to "stitched").
    """
    if salesmen < 1:
        raise ValueError(f"salesmen must be >= 1, got {salesmen}")
    customers = list(customer_seq_1indexed)
    # Check permutation first (catches both wrong-length and
    # wrong-values inputs with a single, more informative error).
    if sorted(customers) != list(range(2, num_loc + 2)):
        raise ValueError(
            f"customer_seq is not a permutation of 2..{num_loc + 1} "
            f"(got {len(customers)} entries, expected {num_loc})"
        )

    # Distribute customers evenly across salesmen routes.
    base_size, remainder = divmod(num_loc, salesmen)
    sizes = [base_size + (1 if i < remainder else 0) for i in range(salesmen)]

    # Build the body: depot(1) + (customers + phantom_depot) per route
    # for routes 0..salesmen-2, then last route's customers with no
    # trailing phantom (the master breaks the circular walk at the
    # real depot, which is at the end of the body in this layout).
    body: list[int] = [1]  # start with the real depot
    cursor = 0
    for i in range(salesmen):
        chunk = customers[cursor : cursor + sizes[i]]
        cursor += sizes[i]
        body.extend(chunk)
        if i < salesmen - 1:
            # Append a phantom depot at this route's end.
            # Phantom depots have ids num_loc+2..num_loc+salesmen
            # (one per additional vehicle after the first).
            body.append(num_loc + 2 + i)

    dimension = num_loc + salesmen
    lines: list[str] = [
        f"NAME : {name}",
        f"COMMENT : Length = 0",  # placeholder; LKH-3 will recompute
        f"TYPE : TOUR",
        f"DIMENSION : {dimension}",
        "TOUR_SECTION",
    ]
    for nid in body:
        lines.append(str(int(nid)))
    lines.extend(["-1", "EOF"])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# LKH-3 routes -> RL4CO action
# ---------------------------------------------------------------------------


def routes_to_action(
    routes: Sequence[Sequence[int]],
    num_loc: int,
    depot_id: int = 1,
) -> np.ndarray:
    """Convert a list of LKH routes (Uchoa 1-indexed) to the RL4CO CVRP
    action format.

    Uchoa CVRP convention: depot = 1, customers = 2..(num_loc+1), phantom
    depots (one per vehicle) = (num_loc+2)..(num_loc+Salesmen).

    CVRPEnv convention: depot = 0, customers = 1..num_loc (1-indexed).

    So we:
        - drop the depot (id 1) at route boundaries
        - drop phantom depots (ids > num_loc+1)
        - convert customer id ``n`` to ``n - 1`` (Uchoa 2..N+1 → 1..N)
        - emit a 0 at each route boundary (real or phantom depot)
    """
    out: list[int] = []
    first = True
    for route in routes:
        if first and route and route[0] == depot_id:
            inner = list(route)[1:]
            first = False
        else:
            inner = list(route)
            first = False
        if inner and inner[-1] == depot_id:
            inner = inner[:-1]
        for nid in inner:
            if nid == depot_id:
                continue
            if nid > num_loc + 1:
                # Phantom depot → emit a 0 (real depot return).
                out.append(0)
                continue
            # Uchoa customer id n (2..N+1) → CVRPEnv 1..N: subtract 1.
            out.append(int(nid) - 1)
        out.append(0)  # depot return at end of route
    if len(out) < num_loc:
        out.extend([0] * (num_loc - len(out)))
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Action -> routes (for the orchestrator's reverse path)
# ---------------------------------------------------------------------------


def action_to_routes(actions: Sequence[int], depot_id: int = 1) -> list[list[int]]:
    """Convert an RL4CO action (0=depot, 1..N=customers) to LKH route format
    (1-indexed, depot repeated at route boundaries).
    """
    routes: list[list[int]] = []
    cur: list[int] = [depot_id]
    for a in actions:
        if a == 0:
            cur.append(depot_id)
            routes.append(cur)
            cur = [depot_id]
        else:
            cur.append(int(a) + 1)  # 0-indexed -> 1-indexed
    if len(cur) > 1:
        cur.append(depot_id)
        routes.append(cur)
    # Drop any leading empty route (no customers before first depot return).
    return [r for r in routes if len(r) > 2 or r == [depot_id, depot_id]]


# ---------------------------------------------------------------------------
# LKH-3 parameter file builder
# ---------------------------------------------------------------------------


@dataclass
class LKHParameters:
    """Parameters for a single LKH-3 invocation."""

    problem_file: str
    output_tour_file: str | None = None
    initial_tour_file: str | None = None
    intermediate_tour_file: str | None = None  # LDE patch
    runs: int = 1
    max_trials: int = 1_000_000
    time_limit_s: float = 60.0
    total_time_limit_s: float | None = None
    seed: int = 1
    trace_level: int = 0
    # For CVRP variants where specifying VEHICLES makes LKH hang, we omit it.
    vehicles: int | None = None

    def to_par_string(self) -> str:
        lines: list[str] = []
        lines.append(f"PROBLEM_FILE = {self.problem_file}")
        if self.output_tour_file:
            lines.append(f"OUTPUT_TOUR_FILE = {self.output_tour_file}")
        if self.initial_tour_file:
            lines.append(f"INITIAL_TOUR_FILE = {self.initial_tour_file}")
        if self.intermediate_tour_file:
            lines.append(f"INTERMEDIATE_TOUR_FILE = {self.intermediate_tour_file}")
        lines.append(f"RUNS = {self.runs}")
        lines.append(f"MAX_TRIALS = {self.max_trials}")
        # LKH-3 treats TIME_LIMIT = 0 as "no trials, exit immediately"
        # and writes no OUTPUT_TOUR_FILE, so the subproblem orchestrator
        # would see ``routes=0`` and ``stitch_failed``. Floor at 1 second.
        lines.append(f"TIME_LIMIT = {max(1, int(round(self.time_limit_s)))}")
        if self.total_time_limit_s is not None:
            lines.append(
                f"TOTAL_TIME_LIMIT = {max(1, int(round(self.total_time_limit_s)))}"
            )
        if self.vehicles is not None:
            lines.append(f"VEHICLES = {self.vehicles}")
        lines.append(f"SEED = {self.seed}")
        lines.append(f"TRACE_LEVEL = {self.trace_level}")
        return "\n".join(lines) + "\n"

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_par_string())
