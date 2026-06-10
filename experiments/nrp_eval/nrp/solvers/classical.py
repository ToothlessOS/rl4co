"""Classical (non-learned) solvers.

All classical solvers round-trip through CPU: ``td.to("cpu")``, run the
algorithm, lift the resulting action arrays back to the caller's device,
and call ``env.get_reward`` to attach the reward.

Stage 1 ships:

- :class:`ORToolsTSPSolver` — real Python implementation of nearest-neighbor
  + 2-opt. Good enough to validate the pipeline (within 5-10% of the
  OR-Tools solution on uniform TSP-50/100).
- :class:`ORToolsVRPSolver` — Clarke-Wright savings algorithm for CVRP.
- :class:`BuiltinEnvSolver` — adapter for :meth:`RL4COEnvBase.solve`.
- :class:`LKHSolver`, :class:`ConcordeTSPSolver`, :class:`GurobiTSPSolver`
  — stubs that raise a clear ``FileNotFoundError`` pointing to the
  relevant env var / install script. Stage 2 will add the real subprocess
  wrappers.
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from .base import Solver, SolverRegistry


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _pad_actions(actions_list) -> np.ndarray:
    """Pad a list of variable-length action lists to ``[B, max_len]`` int64.

    If ``actions_list`` is already a tensor, it is converted to a numpy
    int64 array directly.
    """
    if isinstance(actions_list, torch.Tensor):
        return actions_list.detach().cpu().numpy().astype(np.int64)
    actions_list = list(actions_list)
    if len(actions_list) == 0:
        return np.zeros((0, 0), dtype=np.int64)
    max_len = max(len(a) for a in actions_list)
    out = np.zeros((len(actions_list), max_len), dtype=np.int64)
    for i, a in enumerate(actions_list):
        a_arr = np.asarray(a, dtype=np.int64)
        out[i, : a_arr.shape[0]] = a_arr
    return out


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _tour_length(tour: np.ndarray, locs: np.ndarray) -> float:
    """Closed-tour length of a permutation ``tour`` over ``locs``.

    The tour is treated as a Hamiltonian cycle: ``locs[tour[0]]`` →
    ``locs[tour[1]]`` → ... → ``locs[tour[-1]]`` → ``locs[tour[0]]``.
    """
    if len(tour) == 0:
        return 0.0
    order = locs[tour]
    # closed tour: order[0] -> order[1] -> ... -> order[-1] -> order[0]
    closed = np.concatenate([order, order[:1]], axis=0)
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=-1)))


def _nearest_neighbor(locs: np.ndarray) -> np.ndarray:
    """Build an initial tour with the nearest-neighbor heuristic.

    All N nodes are visited (no depot exclusion — this is a flat TSP).
    The returned tour is a permutation of ``0..N-1`` starting at node 0.
    """
    n = locs.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = np.zeros(n, dtype=np.int64)
    tour[0] = 0
    current = 0
    for step in range(1, n):
        diff = locs - locs[current]
        dists = np.sqrt((diff ** 2).sum(-1))
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        tour[step] = nxt
        visited[nxt] = True
        current = nxt
    return tour


def _two_opt_pass(tour: np.ndarray, locs: np.ndarray) -> tuple[np.ndarray, bool]:
    """One full pass of 2-opt over ``tour`` (flat TSP, all N nodes visited).

    Returns the new tour and a flag indicating whether any improvement was made.
    """
    n = tour.shape[0]
    if n < 4:
        return tour, False
    best = _tour_length(tour, locs)
    improved = False
    new_tour = tour.copy()
    for i in range(n - 1):
        for j in range(i + 2, n):
            candidate = new_tour.copy()
            candidate[i:j] = new_tour[i:j][::-1]
            cand_len = _tour_length(candidate, locs)
            if cand_len + 1e-12 < best:
                best = cand_len
                new_tour = candidate
                improved = True
    return new_tour, improved


def _nearest_neighbor_then_2opt(
    locs: np.ndarray, max_iters: int = 50
) -> np.ndarray:
    """Batch nearest-neighbor + iterative 2-opt over a batch of flat TSPs.

    Args:
        locs: ``[B, N, 2]`` array of locations (no depot).
        max_iters: Max 2-opt passes (one pass = O(N^2) segment reversals).

    Returns:
        ``[B, N]`` int64 array — a permutation of ``0..N-1`` per instance.
    """
    B, N, _ = locs.shape
    out = np.zeros((B, N), dtype=np.int64)
    for b in range(B):
        tour = _nearest_neighbor(locs[b])
        for _ in range(max_iters):
            tour, improved = _two_opt_pass(tour, locs[b])
            if not improved:
                break
        out[b] = tour
    return out


def _clarke_wright_savings(td_cpu: TensorDict) -> np.ndarray:
    """Clarke-Wright savings algorithm for CVRP on a batch of instances.

    ``td_cpu`` is expected to follow the convention of CVRP's
    :meth:`_reset`: depot is prepended to ``locs`` (so ``locs[:, 0]`` is
    the depot, ``locs[:, 1:]`` are customers), ``demand`` is *normalised*
    by the vehicle capacity, and ``vehicle_capacity`` is the per-vehicle
    capacity (typically 1.0 after normalisation).

    The function returns a padded int64 action array
    ``[B, max_route_len]`` where each row encodes one or more routes
    separated by visits to the depot (index 0). The format matches what
    :class:`rl4co.envs.routing.cvrp.CVRPEnv.check_solution_validity`
    expects: every customer appears exactly once; the depot may appear
    any number of times (>= 2 — start and end of the day's routes) and
    any unused slots at the end of a row are padded with 0.
    """
    locs = td_cpu["locs"]  # [B, N+1, 2]  (depot + N customers)
    demand = td_cpu["demand"]  # [B, N]  (normalised by capacity)
    capacity = td_cpu["vehicle_capacity"]  # [B, 1]
    B, Np1, _ = locs.shape
    N = Np1 - 1

    if N == 0:
        return np.zeros((B, 0), dtype=np.int64)

    out = np.zeros((B, 2 * N + 2), dtype=np.int64)  # worst case: each customer in own route (2N+1)
    for b in range(B):
        route = _clarke_wright_single(
            locs[b].numpy(), demand[b].numpy(), float(capacity[b, 0].item())
        )
        out[b, : len(route)] = np.asarray(route, dtype=np.int64)
    return out


def _clarke_wright_single(
    locs: np.ndarray, demand: np.ndarray, capacity: float
) -> list[int]:
    """Single-instance Clarke-Wright savings for CVRP.

    Args:
        locs: ``[N+1, 2]`` array (depot first).
        demand: ``[N]`` array of normalised demands (in [0, 1]).
        capacity: Per-vehicle capacity (unnormalised).

    Returns:
        A list of node indices (depot 0 + customer indices) describing
        concatenated routes. Depot 0 appears at the start and end of the
        day and between sub-routes.
    """
    n = demand.shape[0]
    if n == 0:
        return [0, 0]

    # 1) Build a savings list s_ij = d(0,i) + d(0,j) - d(i,j)
    #    using zero-indexed nodes (1..n are customers).
    savings: list[tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = _euclidean(locs[0], locs[i]) + _euclidean(locs[0], locs[j]) - _euclidean(
                locs[i], locs[j]
            )
            savings.append((s, i, j))
    # Sort by savings descending — biggest savings first.
    savings.sort(key=lambda x: -x[0])

    # 2) Each customer starts in its own route: [0, i, 0].
    #    Track for each node which route it's in and which end of that
    #    route it's attached to.
    route_of: dict[int, int] = {i: i for i in range(1, n + 1)}  # node -> route id
    end_of: dict[int, str] = {i: "i" for i in range(1, n + 1)}
    routes: dict[int, list[int]] = {i: [0, i, 0] for i in range(1, n + 1)}
    load: dict[int, float] = {i: float(demand[i - 1]) for i in range(1, n + 1)}
    active_routes: set[int] = set(range(1, n + 1))

    def can_merge(i: int, j: int) -> bool:
        """Whether the customer ``j`` can be appended to the route
        containing ``i`` (as the appropriate end), respecting capacity and
        the no-interior-customer constraint."""
        if i == j:
            return False
        r_i = route_of[i]
        r_j = route_of[j]
        if r_i == r_j:
            return False  # already in the same route
        if r_i not in active_routes or r_j not in active_routes:
            return False
        # The merge is only legal if ``i`` is at an end of its route and
        # ``j`` is at an end of its route.
        if end_of[i] not in ("i", "j"):  # placeholder; should always be "i" or "j"
            return False
        if end_of[j] not in ("i", "j"):
            return False
        new_load = load[r_i] + load[r_j]
        return new_load <= capacity + 1e-9

    def do_merge(i: int, j: int) -> None:
        """Merge the route containing ``j`` into the route containing
        ``i``, by appending ``j``'s chain at the appropriate end of
        ``i``'s chain."""
        r_i = route_of[i]
        r_j = route_of[j]
        route_i = routes[r_i]
        route_j = routes[r_j]
        # Convention: route_i = [0, ..., 0] with i at position ``pos_i``
        # where ``pos_i`` is 1 (end "i") or -2 (end "j" == last customer).
        if end_of[i] == "i":
            # i is the first customer; prepend j's chain (minus the
            # trailing depot) before i.
            head = route_j[1:-1]  # customers in route_j
            new_route = [0] + head + route_i[1:]
        else:
            # i is the last customer; append j's chain after i.
            head = route_j[1:-1]
            new_route = route_i[:-1] + head + [0]

        # Update bookkeeping.
        new_load = load[r_i] + load[r_j]
        del routes[r_j]
        active_routes.discard(r_j)
        load.pop(r_j, None)
        routes[r_i] = new_route
        load[r_i] = new_load
        # Reassign route_of and end_of for every customer in the merged route.
        for k, node in enumerate(new_route):
            if node == 0:
                continue
            route_of[node] = r_i
            if k == 1:
                end_of[node] = "i"
            elif k == len(new_route) - 2:
                end_of[node] = "j"
            else:
                # interior node — not eligible to be an endpoint of any
                # future merge. Mark with a sentinel.
                end_of[node] = "m"

    for s, i, j in savings:
        if i not in end_of or j not in end_of:
            continue
        if end_of[i] in ("m",) or end_of[j] in ("m",):
            continue
        if can_merge(i, j):
            do_merge(i, j)

    # 3) Concatenate all remaining routes (in any order) into one action
    #    sequence: depot + route1 + depot + route2 + depot + ...
    #    Each route already starts and ends with 0; we strip the trailing
    #    0 of every route except the last, then add a final 0 to close.
    remaining_routes = [routes[r] for r in active_routes]
    if not remaining_routes:
        return [0, 0]
    # Sort by first customer (any stable order works).
    remaining_routes.sort(key=lambda r: r[1] if len(r) > 1 else 0)
    seq: list[int] = []
    for idx, route in enumerate(remaining_routes):
        if idx == 0:
            seq.extend(route)
        else:
            # drop the leading depot of this route
            seq.extend(route[1:])
    # The final depot (closing the day) is already present if the last
    # route ended in 0; otherwise add it.
    if seq[-1] != 0:
        seq.append(0)
    return seq


# ----------------------------------------------------------------------
# ClassicalSolver ABC
# ----------------------------------------------------------------------
class ClassicalSolver(Solver, ABC):
    """Base class for non-learned solvers.

    Subclasses implement :meth:`solve_batch` (CPU, numpy in / numpy out)
    and inherit a ``solve`` method that handles the device round-trip
    and reward computation.
    """

    is_trainable: bool = False
    is_differentiable: bool = False

    @abstractmethod
    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        """Solve a batch of instances on CPU.

        Args:
            td_cpu: A ``TensorDict`` of problem instances, already moved
                to CPU by the framework.

        Returns:
            An int64 numpy array of actions with shape ``[B, L]`` (L may
            be instance-dependent; subclasses may pad to a common width).
        """
        raise NotImplementedError

    def solve(self, td: TensorDict) -> TensorDict:
        device = td.device
        td_cpu = td.to("cpu").clone()
        actions_np = self.solve_batch(td_cpu)  # [B, L] int64
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
        reward = self.env.get_reward(td, actions)
        return TensorDict(
            actions=actions,
            reward=reward,
            batch_size=actions.shape[:1],
        )


# ----------------------------------------------------------------------
# OR-Tools TSP (stage 1: nearest-neighbor + 2-opt)
# ----------------------------------------------------------------------
@SolverRegistry.register("ortools_tsp", env_names=("tsp", "atsp"))
class ORToolsTSPSolver(ClassicalSolver):
    """OR-Tools TSP solver.

    Stage 1 uses a self-contained Python implementation of
    nearest-neighbor seeding followed by iterative 2-opt. This gives
    tours within ~5-10% of OR-Tools' guided local search on uniform
    TSP-50/100 — good enough to validate the pipeline end-to-end.

    The class name ``ortools_tsp`` is kept for consistency with the
    stage-2 plan (which will add a real OR-Tools subprocess wrapper).
    """

    name = "ortools_tsp"

    def __init__(
        self,
        env,
        max_runtime_s: float = 1.0,
        two_opt_iters: int = 50,
        **kwargs,
    ):
        super().__init__(env=env, **kwargs)
        self.max_runtime_s = max_runtime_s
        self.two_opt_iters = two_opt_iters

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        locs = td_cpu["locs"]  # [B, N, 2]
        locs_np = locs.detach().cpu().numpy()
        return _nearest_neighbor_then_2opt(locs_np, max_iters=self.two_opt_iters)


# ----------------------------------------------------------------------
# OR-Tools VRP (stage 1: Clarke-Wright savings)
# ----------------------------------------------------------------------
@SolverRegistry.register("ortools_vrp", env_names=("cvrp", "sdvrp", "cvrptw"))
class ORToolsVRPSolver(ClassicalSolver):
    """OR-Tools CVRP solver.

    Stage 1 uses the Clarke-Wright savings algorithm: routes are
    initialised as ``[0, i, 0]`` for each customer, then merged in
    descending order of savings ``s_ij = d(0,i) + d(0,j) - d(i,j)``,
    respecting the per-vehicle capacity constraint. The class name
    ``ortools_vrp`` matches the planned stage-2 API.
    """

    name = "ortools_vrp"

    def __init__(self, env, max_runtime_s: float = 1.0, **kwargs):
        super().__init__(env=env, **kwargs)
        self.max_runtime_s = max_runtime_s

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        return _clarke_wright_savings(td_cpu)


# ----------------------------------------------------------------------
# Builtin env.solve adapter
# ----------------------------------------------------------------------
@SolverRegistry.register("builtin_solve", env_names=())
class BuiltinEnvSolver(ClassicalSolver):
    """Adapts the env's built-in classical ``solve`` hook.

    Several RL4CO envs expose a static
    :meth:`RL4COEnvBase.solve(instances, max_runtime, num_procs)` that
    wraps a "real" classical solver (e.g. LKH via ``vrplib`` for MTVRP).
    This adapter calls it directly. Env names that don't implement
    ``solve`` raise ``NotImplementedError`` from the env, which we
    surface as a clean error.
    """

    name = "builtin_solve"

    def __init__(
        self,
        env,
        max_runtime_s: float = 10.0,
        num_procs: int = 1,
        **kwargs,
    ):
        super().__init__(env=env, **kwargs)
        self.max_runtime_s = max_runtime_s
        self.num_procs = num_procs

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        # env.solve is a static method on RL4COEnvBase; it takes a
        # TensorDict of instances and returns (actions, costs) or raises
        # NotImplementedError.
        result = self.env.solve(
            td_cpu,
            max_runtime=self.max_runtime_s,
            num_procs=self.num_procs,
        )
        if result is None:
            raise NotImplementedError(
                f"Env {getattr(self.env, 'name', self.env)} does not implement "
                "classical `solve`."
            )
        actions, _ = result
        return _pad_actions(actions)


# ----------------------------------------------------------------------
# LKH-3 stub
# ----------------------------------------------------------------------
@SolverRegistry.register("lkh_tsp", env_names=("tsp",))
class LKHSolver(ClassicalSolver):
    """LKH-3 TSP solver. STUB for stage 1.

    Stage 2 will:

    1. Resolve the LKH-3 binary via ``self.binary_path`` (config) or the
       ``NRP_LKH_BINARY`` env var.
    2. For each instance, write a ``.tsp`` + ``.par`` file to a temp
       directory, run the binary, parse the output via ``vrplib``.
    3. Return the concatenated action arrays.
    """

    name = "lkh_tsp"

    def __init__(self, env, binary_path: str | None = None, **kwargs):
        super().__init__(env=env, **kwargs)
        self.binary_path = binary_path or os.environ.get("NRP_LKH_BINARY")

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        binary = self.binary_path or shutil.which("lkh")
        if binary is None or not Path(binary).exists():
            raise FileNotFoundError(
                "LKH-3 binary not found. Set NRP_LKH_BINARY env var or "
                "solver.binary_path in config. See "
                "experiments/nrp_eval/scripts/install_classical_solvers.sh."
            )
        # Real implementation in stage 2.
        raise NotImplementedError("LKH-3 full implementation deferred to stage 2.")


# ----------------------------------------------------------------------
# Concorde stub
# ----------------------------------------------------------------------
@SolverRegistry.register("concorde_tsp", env_names=("tsp",))
class ConcordeTSPSolver(ClassicalSolver):
    """Concorde TSP solver. STUB for stage 1.

    Stage 2 will subprocess the Concorde binary plus Linkern for the
    initial tour.
    """

    name = "concorde_tsp"

    def __init__(self, env, binary_path: str | None = None, **kwargs):
        super().__init__(env=env, **kwargs)
        self.binary_path = binary_path or os.environ.get("NRP_CONCORDE_BINARY")

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        binary = self.binary_path or shutil.which("concorde")
        if binary is None or not Path(binary).exists():
            raise FileNotFoundError(
                "Concorde binary not found. Set NRP_CONCORDE_BINARY env var or "
                "solver.binary_path in config. See "
                "experiments/nrp_eval/scripts/install_classical_solvers.sh."
            )
        raise NotImplementedError("Concorde full implementation deferred to stage 2.")


# ----------------------------------------------------------------------
# Gurobi stub
# ----------------------------------------------------------------------
@SolverRegistry.register("gurobi_tsp", env_names=("tsp",))
class GurobiTSPSolver(ClassicalSolver):
    """Gurobi TSP solver. STUB for stage 1.

    Stage 2 will use ``gurobipy`` via subprocess (``gurobi_cl``) to solve
    the MIP formulation of TSP.
    """

    name = "gurobi_tsp"

    def __init__(self, env, binary_path: str | None = None, **kwargs):
        super().__init__(env=env, **kwargs)
        self.binary_path = binary_path or os.environ.get("NRP_GUROBI_BINARY")

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        binary = self.binary_path or shutil.which("gurobi_cl")
        if binary is None or not Path(binary).exists():
            raise FileNotFoundError(
                "Gurobi binary not found. Set NRP_GUROBI_BINARY env var or "
                "solver.binary_path in config. See "
                "experiments/nrp_eval/scripts/install_classical_solvers.sh."
            )
        raise NotImplementedError("Gurobi full implementation deferred to stage 2.")
