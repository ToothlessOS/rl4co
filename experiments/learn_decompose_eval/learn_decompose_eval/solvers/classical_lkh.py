"""LKH-3-based CVRP solvers (raw and Barycenter-Clustering-decomposed).

Both are classical solvers that subclass ``nrp.solvers.classical.ClassicalSolver``
and are registered with ``nrp.solvers.SolverRegistry``.

Note: we import the nrp package lazily inside the class to keep this module
importable without the nrp_eval harness on ``sys.path`` (which is only added
at CLI launch time).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from .lkh_format import (
    LKHParameters,
    cvrp_td_to_lkh_problem,
    parse_lkh_tour,
    routes_to_action,
    write_lkh_problem,
)
from .orchestration import IntermediateTourWatcher, OrchestratorConfig, _tail

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_lkh_binary(binary_path: str | None) -> str:
    """Resolve the LKH-3 binary path from explicit arg, env var, or PATH."""
    if binary_path is None:
        binary_path = os.environ.get("LDE_LKH_BINARY") or os.environ.get(
            "NRP_LKH_BINARY"
        )
    if binary_path is None:
        binary_path = shutil.which("lkh") or shutil.which("LKH")
    if binary_path is None or not Path(binary_path).exists():
        raise FileNotFoundError(
            "LKH-3 binary not found. Set LDE_LKH_BINARY (or NRP_LKH_BINARY) "
            "to the absolute path of the LKH-3 executable, or pass "
            "binary_path to the solver."
        )
    return binary_path


def _solve_one_instance(
    lkh_binary: str,
    td_instance: TensorDict,
    *,
    name: str = "inst",
    time_limit_s: float,
    seed: int = 1,
    trace_level: int = 0,
    tmpdir: str | None = None,
    cleanup_tmp: bool = True,
) -> tuple[list[list[int]], float]:
    """Solve a single CVRP instance with one LKH-3 invocation."""
    if tmpdir is None:
        tmp = tempfile.mkdtemp(prefix=f"lde_raw_{name}_")
    else:
        tmp = tmpdir
        os.makedirs(tmp, exist_ok=True)
    problem_path = os.path.join(tmp, "instance.vrp")
    final_path = os.path.join(tmp, "final.tour")
    par_path = os.path.join(tmp, "instance.par")
    stderr_path = os.path.join(tmp, "lkh.stderr")
    stdout_path = os.path.join(tmp, "lkh.stdout")

    try:
        problem_str = cvrp_td_to_lkh_problem(td_instance, name=name)
        write_lkh_problem(problem_path, problem_str)
        LKHParameters(
            problem_file=problem_path,
            output_tour_file=final_path,
            runs=1,
            max_trials=10_000_000,
            time_limit_s=time_limit_s,
            seed=seed,
            trace_level=trace_level,
        ).write(par_path)
        log.info(
            "raw_solve_start(name=%s, time_limit_s=%.1f, tmpdir=%s)",
            name,
            time_limit_s,
            tmp,
        )

        t0 = time.monotonic()
        try:
            with open(stdout_path, "wb") as outf, open(stderr_path, "wb") as errf:
                rc = subprocess.call(
                    [lkh_binary, par_path],
                    timeout=time_limit_s + 10,
                    stdout=outf,
                    stderr=errf,
                )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            log.error(
                "raw_timeout(name=%s, time_limit_s=%.1f, elapsed=%.1f, stderr_tail=%r)",
                name,
                time_limit_s,
                elapsed,
                _tail(stderr_path, 2048),
            )
            return [], elapsed
        elapsed = time.monotonic() - t0
        if rc != 0:
            log.error(
                "raw_nonzero(name=%s, rc=%d, elapsed=%.1f, stderr_tail=%r)",
                name,
                rc,
                elapsed,
                _tail(stderr_path, 2048),
            )
            return [], elapsed
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            log.error(
                "raw_no_final_tour(name=%s, elapsed=%.1f, final_path=%s, stderr_tail=%r)",
                name,
                elapsed,
                final_path,
                _tail(stderr_path, 2048),
            )
            return [], elapsed
        routes = parse_lkh_tour(final_path)
        log.info(
            "raw_solve_done(name=%s, routes=%d, elapsed=%.1f)",
            name,
            len(routes),
            elapsed,
        )
        return routes, elapsed
    finally:
        if cleanup_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Raw LKH-3 solver
# ---------------------------------------------------------------------------


class RawLKH3CVRSolver:
    """Raw LKH-3 CVRP solver (no decomposition).

    Each instance is solved by a single LKH-3 subprocess with a time limit.
    """

    name = "raw_lkh_cvrp"
    is_trainable = False
    is_differentiable = False

    def __init__(
        self,
        env,
        binary_path: str | None = None,
        max_runtime_s: float = 60.0,
        num_workers: int = 1,
        seed: int = 1,
        trace_level: int = 0,
        **kwargs,
    ):
        self.env = env
        self.device = torch.device("cpu")
        self.kwargs = kwargs
        self.lkh_binary = _resolve_lkh_binary(binary_path)
        self.max_runtime_s = max_runtime_s
        self.num_workers = num_workers
        self.seed = seed
        self.trace_level = trace_level
        self._registered_name = "raw_lkh_cvrp"
        self._supported_envs = ("cvrp",)

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        batch = td_cpu.batch_size[0] if td_cpu.batch_size else 1
        results: list[np.ndarray] = []
        n_failed = 0
        # Parallel across instances
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = []
            for i in range(batch):
                td_i = td_cpu[i]
                futures.append(
                    ex.submit(
                        _solve_one_instance,
                        self.lkh_binary,
                        td_i,
                        name=f"raw_inst_{i}",
                        time_limit_s=self.max_runtime_s,
                        seed=self.seed,
                        trace_level=self.trace_level,
                    )
                )
            for i, fut in enumerate(futures):
                try:
                    routes, _ = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.exception("raw_instance_exception(idx=%d): %s", i, e)
                    routes = []
                # num_loc is the same for all instances in the batch
                num_loc = td_cpu["locs"].shape[-2]
                if td_cpu["locs"].shape[-2] == td_cpu["demand"].shape[-1] + 1:
                    num_loc = td_cpu["demand"].shape[-1]
                else:
                    num_loc = td_cpu["locs"].shape[-2] - 1
                if not routes:
                    n_failed += 1
                    # No solution found; emit a dummy action (all depot=0).
                    results.append(np.zeros(num_loc + 1, dtype=np.int64))
                else:
                    results.append(routes_to_action(routes, num_loc))
        # Pad to common length
        max_len = max(a.shape[0] for a in results)
        out = np.zeros((len(results), max_len), dtype=np.int64)
        for i, a in enumerate(results):
            out[i, : a.shape[0]] = a
        log.info(
            "raw_batch_done(batch=%d, failed=%d, ok=%d)",
            batch,
            n_failed,
            batch - n_failed,
        )
        return out

    def solve(self, td: TensorDict) -> TensorDict:
        device = td.device
        td_cpu = td.to("cpu").clone()
        actions_np = self.solve_batch(td_cpu)
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
        reward = self.env.get_reward(td, actions)
        return TensorDict(
            actions=actions,
            reward=reward,
            batch_size=actions.shape[:1],
        )


# ---------------------------------------------------------------------------
# Barycenter-clustering LKH-3 solver
# ---------------------------------------------------------------------------


class BarycentreLKH3CVRSolver:
    """LKH-3 + Barycenter-Clustering Decomposition CVRP solver.

    Each instance is solved by ``IntermediateTourWatcher`` which:
    - launches patched LKH-3 with INTERMEDIATE_TOUR_FILE,
    - polls for intermediate tours,
    - decomposes them via BCC, solves subproblems in parallel,
    - warm-restarts the master with the stitched solution,
    - until the total time budget is exhausted.
    """

    name = "bcc_lkh_cvrp"
    is_trainable = False
    is_differentiable = False

    def __init__(
        self,
        env,
        binary_path: str | None = None,
        max_total_s: float = 60.0,
        decompose_every_s: float = 5.0,
        num_workers: int = 4,
        target_max_subproblem_size: int = 200,
        random_state: int = 0,
        seed: int = 1,
        trace_level: int = 0,
        **kwargs,
    ):
        self.env = env
        self.device = torch.device("cpu")
        self.kwargs = kwargs
        self.lkh_binary = _resolve_lkh_binary(binary_path)
        self.max_total_s = max_total_s
        self.decompose_every_s = decompose_every_s
        self.num_workers = num_workers
        self.target_max_subproblem_size = target_max_subproblem_size
        self.random_state = random_state
        self.seed = seed
        self.trace_level = trace_level
        self._registered_name = "bcc_lkh_cvrp"
        self._supported_envs = ("cvrp",)

    def _watcher(self, tmpdir: str | None = None) -> IntermediateTourWatcher:
        cfg = OrchestratorConfig(
            lkh_binary=self.lkh_binary,
            decompose_every_s=self.decompose_every_s,
            max_total_s=self.max_total_s,
            num_workers=self.num_workers,
            target_max_subproblem_size=self.target_max_subproblem_size,
            random_state=self.random_state,
            seed=self.seed,
            tmpdir=tmpdir,
            trace_level=self.trace_level,
        )
        return IntermediateTourWatcher(cfg)

    def solve_batch(self, td_cpu: TensorDict) -> np.ndarray:
        batch = td_cpu.batch_size[0] if td_cpu.batch_size else 1
        num_loc = td_cpu["demand"].shape[-1]
        results: list[np.ndarray] = []
        n_failed = 0
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = []
            for i in range(batch):
                td_i = td_cpu[i]
                futures.append(ex.submit(self._solve_one, td_i, f"bcc_inst_{i}"))
            for i, fut in enumerate(futures):
                try:
                    routes = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.exception("bcc_instance_exception(idx=%d): %s", i, e)
                    routes = []
                if not routes:
                    n_failed += 1
                    results.append(np.zeros(num_loc + 1, dtype=np.int64))
                else:
                    results.append(routes_to_action(routes, num_loc))
        max_len = max(a.shape[0] for a in results)
        out = np.zeros((len(results), max_len), dtype=np.int64)
        for i, a in enumerate(results):
            out[i, : a.shape[0]] = a
        log.info(
            "bcc_batch_done(batch=%d, failed=%d, ok=%d)",
            batch,
            n_failed,
            batch - n_failed,
        )
        return out

    def _solve_one(self, td_i: TensorDict, name: str = "inst") -> list[list[int]]:
        watcher = self._watcher()
        routes, _ = watcher.solve(td_i, name=name)
        return routes

    def solve(self, td: TensorDict) -> TensorDict:
        device = td.device
        td_cpu = td.to("cpu").clone()
        actions_np = self.solve_batch(td_cpu)
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
        reward = self.env.get_reward(td, actions)
        return TensorDict(
            actions=actions,
            reward=reward,
            batch_size=actions.shape[:1],
        )


# ---------------------------------------------------------------------------
# Registration (deferred; only runs if nrp.solvers.base is importable)
# ---------------------------------------------------------------------------


def _register() -> None:
    try:
        from nrp.solvers.base import SolverRegistry
    except Exception:  # noqa: BLE001
        return
    SolverRegistry.registry.setdefault("raw_lkh_cvrp", RawLKH3CVRSolver)
    SolverRegistry.registry.setdefault("bcc_lkh_cvrp", BarycentreLKH3CVRSolver)
    SolverRegistry.env_support.setdefault("raw_lkh_cvrp", ("cvrp",))
    SolverRegistry.env_support.setdefault("bcc_lkh_cvrp", ("cvrp",))
    RawLKH3CVRSolver._registered_name = "raw_lkh_cvrp"
    RawLKH3CVRSolver._supported_envs = ("cvrp",)
    BarycentreLKH3CVRSolver._registered_name = "bcc_lkh_cvrp"
    BarycentreLKH3CVRSolver._supported_envs = ("cvrp",)


_register()
