"""Orchestrator: run patched LKH-3 with periodic BCC decomposition.

The orchestrator launches LKH-3 with ``INTERMEDIATE_TOUR_FILE`` pointing to
a file it polls. Every ``decompose_every_s`` seconds (or whenever the file
is updated), the orchestrator:

    1. Reads the current best tour from the intermediate file.
    2. Splits it into routes.
    3. Decomposes the routes via ``BarycentreClusteringDecomposer``.
    4. Runs each subproblem with LKH-3 in parallel
       (``ThreadPoolExecutor``) — each subproblem is a short LKH-3
       invocation warm-started with the sub-tour from the parent solution.
    5. Stitches the best sub-solutions into a new initial tour.
    6. Kills the running master LKH-3 (``SIGTERM``) and restarts it with
       the consolidated tour as ``INITIAL_TOUR_FILE``.

Continues until the total time budget ``max_total_s`` is exhausted.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from .decomposition import BarycentreClusteringDecomposer, Subproblem
from .lkh_format import (
    LKHParameters,
    action_to_routes,
    cvrp_td_to_lkh_problem,
    parse_lkh_tour,
    parse_tour_with_cost,
    write_cvrp_initial_tour,
    write_lkh_problem,
    write_subproblem_initial_tour,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tail(path: str, n: int = 2048) -> str:
    """Return the last ``n`` bytes of a file as a decoded string (best-effort)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n), os.SEEK_SET)
            data = f.read()
    except OSError as e:
        return f"<unreadable: {e}>"
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return repr(data)


def _run_lkh_blocking(
    lkh_binary: str,
    par_path: str,
    stderr_path: str | None = None,
    timeout_s: float | None = None,
) -> int | None:
    """Run LKH-3 blocking; return its returncode (or None on timeout).

    If ``stderr_path`` is given, LKH-3's stderr is redirected to that file
    (which can later be tailed for diagnostics).  Otherwise it is captured
    in memory and logged at DEBUG.
    """
    try:
        if stderr_path is not None:
            with open(stderr_path, "wb") as errf:
                rc = subprocess.call(
                    [lkh_binary, par_path],
                    timeout=timeout_s,
                    stdout=subprocess.DEVNULL,
                    stderr=errf,
                )
        else:
            proc = subprocess.run(
                [lkh_binary, par_path],
                timeout=timeout_s,
                check=False,
                capture_output=True,
            )
            rc = proc.returncode
            if rc != 0:
                log.debug("LKH-3 returned %d (stderr below)\n%s", rc, proc.stderr[:500])
        if rc != 0 and stderr_path is not None:
            log.error(
                "LKH-3 returned %d (stderr tail below)\n%s",
                rc,
                _tail(stderr_path),
            )
        return rc
    except subprocess.TimeoutExpired:
        log.error("LKH-3 exceeded timeout %ss", timeout_s)
        return None


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _infer_depot_id(routes: list[list[int]]) -> int:
    """Infer the depot id from a list of LKH routes (1-indexed).

    The depot appears at the start and end of each route. LKH-3's CVRP
    output replaces the real depot (id = DIMENSION) with phantom depots
    (one per vehicle, ids DIMENSION+1..DIMENSION+Salesmen-1) at route
    boundaries, so we pick the *largest* node id (which is the last
    phantom depot) as the depot marker.
    """
    if not routes:
        return 1
    max_id = max(max(r) for r in routes)
    return max_id


def _read_tour_safely(
    path: str,
    retries: int = 5,
    sleep_s: float = 0.05,
    num_loc: int | None = None,
) -> list[list[int]] | None:
    """Read a TSPLIB tour file with retries to handle non-atomic writes.

    ``num_loc`` is the number of customer nodes in the problem (used to
    recognise phantom depots in LKH-3 CVRP monster tours — see
    ``parse_lkh_tour``). Pass ``None`` to fall back to the legacy
    TSP-style depot-id counting.
    """
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                time.sleep(sleep_s)
                continue
            return parse_lkh_tour(path, num_loc=num_loc)
        except (OSError, ValueError) as e:
            last_err = e
            log.debug("Read failed (%s); retrying", e)
            time.sleep(sleep_s)
    log.warning(
        "_read_tour_safely: giving up on %s after %d retries: %s",
        path,
        retries,
        last_err,
    )
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    lkh_binary: str
    max_total_s: float = 60.0
    # Minimum seconds between master restart cycles.
    min_restart_interval_s: float = 10.0
    # Population size for BCC. 1 = single-tour (no diversification).
    # 4 = small in-process population (default; matches the paper's
    # intent on a single-tour LKH-3 master).
    population_size: int = 4
    # Cap on cumulative sub-problem solving time as a fraction of
    # max_total_s. Once exceeded, no more decompositions.
    max_subs_time_fraction: float = 0.3
    num_workers: int = 4
    target_max_subproblem_size: int = 200
    random_state: int = 0
    seed: int = 1
    tmpdir: str | None = None
    trace_level: int = 0
    # Deprecated. Kept for backwards compatibility with old Hydra
    # configs. The orchestrator logs a one-line warning if this is
    # set to a non-default value, then ignores it.
    decompose_every_s: float = 5.0


class IntermediateTourWatcher:
    """Run LKH-3 on a single instance with periodic BCC decomposition."""

    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        self.decomposer = BarycentreClusteringDecomposer(
            target_max_subproblem_size=cfg.target_max_subproblem_size,
            random_state=cfg.random_state,
        )
        if cfg.decompose_every_s != 5.0:
            log.warning(
                "OrchestratorConfig.decompose_every_s is deprecated and "
                "ignored; use min_restart_interval_s instead "
                "(got decompose_every_s=%s).",
                cfg.decompose_every_s,
            )

    def solve(
        self, td: TensorDict, name: str = "inst"
    ) -> tuple[list[list[int]], float]:
        """Solve a single instance with continuous-master + population.

        Lifecycle:
            1. Launch one LKH-3 master with TIME_LIMIT = max_total_s + 5.
            2. Poll ``intermediate.tour`` mtime at 100ms granularity.
            3. When an update arrives AND min_restart_interval_s has
               elapsed since the last restart: pick a random population
               slot (current or one of N-1 snapshots), decompose it,
               run subproblems, stitch, restart the master with the
               stitched result as the new initial tour.
            4. Exit on time budget, master self-exit, or never-updated.
        """
        cleanup_tmp = self.cfg.tmpdir is None
        if self.cfg.tmpdir is None:
            tmp = tempfile.mkdtemp(prefix=f"lde_{name}_")
        else:
            tmp = self.cfg.tmpdir
            os.makedirs(tmp, exist_ok=True)
        problem_path = os.path.join(tmp, "instance.vrp")
        intermediate_path = os.path.join(tmp, "intermediate.tour")
        final_path = os.path.join(tmp, "final.tour")
        initial_path = os.path.join(tmp, "initial.tour")
        master_stderr_path = os.path.join(tmp, "lkh.master.stderr")
        master_stdout_path = os.path.join(tmp, "lkh.master.stdout")
        population_dir = os.path.join(tmp, "population")
        os.makedirs(population_dir, exist_ok=True)

        # Write the TSPLIB problem file once.
        problem_str = cvrp_td_to_lkh_problem(td, name=name)
        write_lkh_problem(problem_path, problem_str)
        num_loc_hint = (
            td["locs"].shape[-2] - 1
            if "locs" in td.keys()
            and td["locs"].shape[-2] == td["demand"].shape[-1] + 1
            else td["demand"].shape[-1]
        )
        log.info(
            "solve_start(name=%s, num_loc=%d, max_total_s=%.1f, "
            "min_restart_interval_s=%.1f, population_size=%d, "
            "max_subs_time_fraction=%.2f, num_workers=%d, "
            "target_max_subproblem_size=%d, lkh=%s, tmpdir=%s)",
            name,
            num_loc_hint,
            self.cfg.max_total_s,
            self.cfg.min_restart_interval_s,
            self.cfg.population_size,
            self.cfg.max_subs_time_fraction,
            self.cfg.num_workers,
            self.cfg.target_max_subproblem_size,
            self.cfg.lkh_binary,
            tmp,
        )

        start = time.monotonic()
        best_routes: list[list[int]] = []
        iteration = 0
        n_decompositions = 0
        n_restarts = 0
        total_subs_s = 0.0
        subs_by_source: dict[str, int] = {}
        subs_budget_s = self.cfg.max_total_s * self.cfg.max_subs_time_fraction
        master_self_exit = False
        loop_exit_reason = "unknown"
        # Number of vehicles the master CVRP will use, read from the
        # master's first final.tour.  We need this to build a
        # phantom-depot-aware initial tour when we restart the
        # master with the stitched result.  Initialized to None
        # and populated when the master first exits (or after the
        # first final.tour is read).
        salesmen: int | None = None

        # Population: 1 "current" slot (intermediate_path) + N-1 snapshots.
        # Snapshots live as .tour files under population_dir/snap_<i>.tour.
        n_snapshots = max(0, self.cfg.population_size - 1)
        snapshot_paths = [
            os.path.join(population_dir, f"snap_{i}.tour") for i in range(n_snapshots)
        ]

        def _pick_population_slot() -> tuple[int, str]:
            """Return (slot_index, slot_kind).

            For Barycenter clustering we follow the original paper's
            ALNS implementation: decompose the *current/immediate*
            solution of the running LKH-3 master (slot 0). The
            ``population_size > 1`` machinery (snapshot slots 1..N-1)
            is preserved for future decomposition methods that do use
            a true population, but is unused in this baseline.
            """
            return 0, "current"

        def _read_slot(slot_idx: int) -> list[list[int]] | None:
            path = intermediate_path if slot_idx == 0 else snapshot_paths[slot_idx - 1]
            return _read_tour_safely(path, num_loc=num_loc_hint)

        def _write_slot(slot_idx: int, routes_1indexed: list[int]) -> None:
            path = intermediate_path if slot_idx == 0 else snapshot_paths[slot_idx - 1]
            _write_initial_tour(path, routes_1indexed, depot_id=1)

        try:
            # Launch master with TIME_LIMIT = max_total_s + 5s slack.
            # LKH-3 self-exits when its limit is hit, but we will also
            # SIGTERM it at the end of solve(). The 5s slack prevents a
            # race where LKH-3 exits 1s before our budget.
            self._launch_master(
                problem_path=problem_path,
                intermediate_path=intermediate_path,
                final_path=final_path,
                initial_path=None,
                stderr_path=master_stderr_path,
                stdout_path=master_stdout_path,
                time_limit_s=self.cfg.max_total_s + 5,
            )
            last_mtime = 0.0
            last_restart_time = start  # No restart in the first min_restart_interval_s

            while True:
                elapsed = time.monotonic() - start
                if elapsed >= self.cfg.max_total_s:
                    loop_exit_reason = "time_budget"
                    break

                # Master self-exit detection.
                if self._master_proc.poll() is not None:
                    rc = self._master_proc.returncode
                    log.info(
                        "master_self_exited(name=%s, rc=%d, elapsed=%.1fs)",
                        name,
                        rc,
                        elapsed,
                    )
                    # Capture the master's Salesmen from the
                    # final.tour dimension so the next warm-start
                    # can build a phantom-depot-aware initial tour.
                    if salesmen is None:
                        meta = parse_tour_with_cost(final_path)
                        if meta is not None:
                            _, _, master_dim = meta
                            salesmen = master_dim - num_loc_hint
                    master_self_exit = True
                    loop_exit_reason = "master_self_exited"
                    break

                cur_mtime = _mtime(intermediate_path)
                got_update = cur_mtime > last_mtime and cur_mtime > 0
                interval_ok = (
                    time.monotonic() - last_restart_time
                    >= self.cfg.min_restart_interval_s
                )

                if not got_update:
                    time.sleep(0.1)
                    continue
                last_mtime = cur_mtime

                if not interval_ok:
                    log.debug(
                        "skip_restart(name=%s, iter=%d, since_last=%.1fs < "
                        "min_interval=%.1fs)",
                        name,
                        iteration,
                        time.monotonic() - last_restart_time,
                        self.cfg.min_restart_interval_s,
                    )
                    continue

                # Sub-budget check.
                if total_subs_s >= subs_budget_s:
                    log.info(
                        "subs_budget_exhausted(name=%s, total_subs_s=%.1f, "
                        "budget=%.1f); keeping master running without "
                        "decomposition",
                        name,
                        total_subs_s,
                        subs_budget_s,
                    )
                    continue

                # Pick a population slot to decompose.
                slot_idx, slot_kind = _pick_population_slot()
                routes = _read_slot(slot_idx)
                if not routes:
                    log.warning(
                        "empty_slot(name=%s, slot=%s, iter=%d)",
                        name,
                        slot_kind,
                        iteration,
                    )
                    continue
                log.info(
                    "intermediate_updated(name=%s, iter=%d, mtime=%.3f, "
                    "elapsed=%.1fs, slot=%s)",
                    name,
                    iteration,
                    cur_mtime,
                    elapsed,
                    slot_kind,
                )
                best_routes = routes
                n_decompositions += 1
                subs_by_source[slot_kind] = subs_by_source.get(slot_kind, 0) + 1

                num_loc = (
                    td["locs"].shape[-2] - 1
                    if "locs" in td.keys()
                    and td["locs"].shape[-2] == td["demand"].shape[-1] + 1
                    else td["demand"].shape[-1]
                )
                try:
                    subproblems = self.decomposer.decompose(
                        td,
                        _routes_zero_indexed(routes, depot_id=1, num_loc=num_loc),
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("Decomposition failed: %s", e)
                    continue

                if not subproblems:
                    log.info(
                        "decompose_no_subproblems(name=%s, iter=%d, "
                        "n_routes=%d, slot=%s)",
                        name,
                        iteration,
                        len(routes),
                        slot_kind,
                    )
                    continue

                sub_t0 = time.monotonic()
                sub_results = self._solve_subproblems(subproblems, start, name=name)
                total_subs_s += time.monotonic() - sub_t0

                stitched = _stitch_routes(sub_results, td)
                if stitched is None:
                    log.warning(
                        "stitch_failed(name=%s, iter=%d, sub_results=%d, " "slot=%s)",
                        name,
                        iteration,
                        len(sub_results),
                        slot_kind,
                    )
                    continue
                # Overwrite the picked slot with the stitched result.
                _write_slot(slot_idx, stitched)

                # Decide whether to restart the master with the
                # stitched result as a warm-start INITIAL_TOUR_FILE.
                # We need the master to have already produced a
                # final.tour (so we know ``Salesmen``) and the
                # stitched tour must be non-empty.
                if salesmen is None or salesmen < 1:
                    # No Salesmen yet — fall back to a MinSalesmen
                    # heuristic so the warm-start can still happen.
                    if salesmen is None:
                        # Try parsing final.tour (in case the master
                        # exited cleanly between the previous iter
                        # and this one, even though the loop is
                        # still going).
                        meta = parse_tour_with_cost(final_path)
                        if meta is not None:
                            _, _, master_dim = meta
                            salesmen = master_dim - num_loc_hint
                if salesmen is None or salesmen < 1:
                    # Fallback: estimate MinSalesmen from total
                    # demand / capacity, defaulting to 1.
                    if "demand" in td.keys() and "capacity" in td.keys():
                        cap = float(td["capacity"].reshape(()).item())
                        total_demand = float(
                            (td["demand"] * cap).sum().reshape(()).item()
                        )
                        salesmen = max(
                            1,
                            int(total_demand // cap) + (1 if total_demand % cap else 0),
                        )
                    else:
                        salesmen = 1
                    log.info(
                        "master_salesmen(name=%s, salesmen=%d, source=heuristic)",
                        name,
                        salesmen,
                    )
                else:
                    log.info(
                        "master_salesmen(name=%s, salesmen=%d, source=final_tour)",
                        name,
                        salesmen,
                    )

                # Write a CVRP-format initial tour with phantom
                # depots matching the master's expected DIMENSION.
                # ``stitched[1:]`` drops the leading depot entry
                # (the real depot is emitted by write_cvrp_initial_tour
                # at the start of the body).  This is the LAST write
                # to ``initial_path`` so the warm-start file isn't
                # clobbered by the simpler flat-tour writer below.
                if len(stitched) > 1:
                    write_cvrp_initial_tour(
                        initial_path,
                        customer_seq_1indexed=stitched[1:],
                        num_loc=num_loc_hint,
                        salesmen=salesmen,
                        name=f"lde_{name}_iter{iteration}",
                    )
                log.info(
                    "stitched(name=%s, iter=%d, n_routes=%d, "
                    "total_customers=%d, slot=%s, salesmen_used=%d)",
                    name,
                    iteration,
                    sum(1 for x in stitched if x == 1),
                    len(stitched),
                    slot_kind,
                    salesmen,
                )

                # Kill the master and restart with the warm-start
                # initial tour.  LKH-3's CVRP ReadTour will accept
                # the file because the dimension matches
                # (num_loc + salesmen).
                self._stop_master()
                n_restarts += 1
                remaining = self.cfg.max_total_s - (time.monotonic() - start)
                self._launch_master(
                    problem_path=problem_path,
                    intermediate_path=intermediate_path,
                    final_path=final_path,
                    initial_path=initial_path,
                    stderr_path=master_stderr_path,
                    stdout_path=master_stdout_path,
                    time_limit_s=remaining + 5,
                )
                last_mtime = 0.0
                last_restart_time = time.monotonic()
                iteration += 1

            # Final read of the master output — must happen INSIDE the
            # try so the tmpdir (and final_path) are still alive for
            # diagnostics.
            final_routes = _read_tour_safely(final_path, num_loc=num_loc_hint)
            if final_routes:
                best_routes = final_routes
            else:
                log.warning(
                    "no_final_tour(name=%s, final_path=%s, " "master_stderr_tail=%r)",
                    name,
                    final_path,
                    _tail(master_stderr_path, 1024),
                )
        finally:
            # Always reap the master, regardless of how we exit.
            self._stop_master()
            if cleanup_tmp:
                shutil.rmtree(tmp, ignore_errors=True)

        total_s = time.monotonic() - start
        log.info(
            "solve_done(name=%s, loop_exit=%s, iterations=%d, "
            "n_decompositions=%d, n_restarts=%d, subs_by_source=%s, "
            "total_subs_s=%.1f, subs_budget_s=%.1f, "
            "master_self_exit=%s, best_routes=%d, total_s=%.2f)",
            name,
            loop_exit_reason,
            iteration,
            n_decompositions,
            n_restarts,
            subs_by_source,
            total_subs_s,
            subs_budget_s,
            master_self_exit,
            len(best_routes),
            total_s,
        )
        return best_routes, total_s

    # -----------------------------------------------------------------------

    def _launch_master(
        self,
        *,
        problem_path: str,
        intermediate_path: str,
        final_path: str,
        initial_path: str | None,
        time_limit_s: float,
        stderr_path: str | None = None,
        stdout_path: str | None = None,
    ) -> None:
        # Build the .par file.
        par_path = problem_path.replace(".vrp", ".par")
        par = LKHParameters(
            problem_file=problem_path,
            output_tour_file=final_path,
            intermediate_tour_file=intermediate_path,
            initial_tour_file=initial_path,
            runs=1,
            max_trials=10_000_000,
            time_limit_s=time_limit_s,
            seed=self.cfg.seed,
            trace_level=self.cfg.trace_level,
        )
        par.write(par_path)
        # Open stderr/stdout files now so the Popen can be told to write
        # to them.  We hold the FDs in self._master_stdout_fd / _master_stderr_fd
        # so they don't get GC'd before the subprocess closes them.
        stdout_fh = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL
        stderr_fh = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL
        # Run in background so we can poll.
        proc = subprocess.Popen(
            [self.cfg.lkh_binary, par_path],
            stdout=stdout_fh,
            stderr=stderr_fh,
            preexec_fn=os.setsid,
        )
        self._master_stdout_fh = stdout_fh
        self._master_stderr_fh = stderr_fh
        self._master_proc = proc
        log.info(
            "master_launch(pid=%d, time_limit_s=%.1f, initial_tour=%s, "
            "stderr=%s, stdout=%s)",
            proc.pid,
            time_limit_s,
            initial_path or "NONE",
            stderr_path or "DEVNULL",
            stdout_path or "DEVNULL",
        )

    def _stop_master(self) -> None:
        proc = getattr(self, "_master_proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        rc = proc.poll()
        log.info("master_stopped(pid=%d, rc=%s)", proc.pid, rc)
        for attr in ("_master_stdout_fh", "_master_stderr_fh"):
            fh = getattr(self, attr, None)
            if fh is not None and fh is not subprocess.DEVNULL:
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)

    def _solve_subproblems(
        self,
        subproblems: list[Subproblem],
        start_time: float,
        name: str = "inst",
    ) -> list[tuple[Subproblem, list[list[int]]]]:
        """Solve each subproblem with LKH-3 in parallel; return a list of
        (Subproblem, routes) pairs so the stitch can map subproblem-local
        customer ids back to master Uchoa ids."""
        # Decide a fair per-subproblem time slice.
        remaining = max(1.0, self.cfg.max_total_s - (time.monotonic() - start_time))
        # We need to leave time for the master restart + final read; budget ~50% to subs.
        per_sub_s = min(
            self.cfg.min_restart_interval_s,
            max(0.5, remaining * 0.5 / max(1, len(subproblems))),
        )
        # But cap at the budget for one LKH-3 sub-call.
        per_sub_s = min(per_sub_s, remaining)
        log.info(
            "subs_solving(name=%s, k=%d, per_sub_s=%.1f, num_workers=%d)",
            name,
            len(subproblems),
            per_sub_s,
            self.cfg.num_workers,
        )

        def _one(sp: Subproblem) -> tuple[Subproblem, list[list[int]]]:
            tmp = tempfile.mkdtemp(prefix=f"lde_sub_{sp.num_loc}_")
            try:
                sp_vrp = os.path.join(tmp, "sub.vrp")
                sp_par = os.path.join(tmp, "sub.par")
                sp_tour = os.path.join(tmp, "sub.tour")
                sp_initial = os.path.join(tmp, "sub_initial.tour")
                sp_err = os.path.join(tmp, "lkh.stderr")
                write_lkh_problem(
                    sp_vrp, cvrp_td_to_lkh_problem(sp.to_td(), name="sub")
                )
                # Warm-start: write the parent routes as a TSPLIB TOUR
                # and pass it to LKH-3 as INITIAL_TOUR_FILE. SALESMEN
                # is set to n_parent_routes so the problem's expanded
                # DIMENSION matches the tour's DIMENSION. Edge case:
                # if n_parent_routes <= 1 there's nothing for LKH-3
                # to do (single-route clusters are already solved),
                # so we skip the LKH-3 call and return the parent
                # routes directly.
                warm_start_routes: list[list[int]] | None = None
                if sp.n_parent_routes == 0:
                    # No parent routes to warm-start with. Fall back
                    # to the no-warm-start path: let LKH-3 figure
                    # out the routes from the demand/capacity math.
                    salesmen: int | None = None
                elif sp.n_parent_routes == 1:
                    # Trivial: one parent route, one vehicle. No
                    # LKH-3 search needed; return the parent route
                    # in Uchoa 1-indexed form (with depot markers
                    # so the stitcher can treat it like any other
                    # sub-tour).
                    only_route = sp.parent_routes[0]
                    warm_start_routes = [[1] + list(only_route) + [1]]
                    salesmen = 1
                else:
                    # Write the parent routes as a warm-start tour.
                    write_subproblem_initial_tour(
                        sp_initial,
                        sp.parent_routes,
                        num_loc=sp.num_loc,
                        name="sub_initial",
                    )
                    salesmen = sp.n_parent_routes

                # Short-circuit: if we have a warm-start with 1
                # vehicle, return the parent route without invoking
                # LKH-3.
                if warm_start_routes is not None and sp.n_parent_routes == 1:
                    log.info(
                        "sub_skip_lkh(name=%s, sub_size=%d, "
                        "n_parent_routes=1, reason=single_route)",
                        name,
                        sp.num_loc,
                    )
                    return sp, warm_start_routes

                initial_tour_file_arg = (
                    sp_initial if sp.n_parent_routes > 1 else None
                )
                LKHParameters(
                    problem_file=sp_vrp,
                    output_tour_file=sp_tour,
                    initial_tour_file=initial_tour_file_arg,
                    runs=1,
                    max_trials=10_000_000,
                    time_limit_s=per_sub_s,
                    seed=self.cfg.seed,
                    trace_level=self.cfg.trace_level,
                    salesmen=salesmen,
                ).write(sp_par)
                t0 = time.monotonic()
                rc = _run_lkh_blocking(
                    self.cfg.lkh_binary,
                    sp_par,
                    stderr_path=sp_err,
                    timeout_s=per_sub_s + 5,
                )
                wallclock = time.monotonic() - t0
                routes = _read_tour_safely(sp_tour, num_loc=sp.num_loc) or []
                if rc != 0:
                    log.error(
                        "sub_failed(name=%s, sub_size=%d, rc=%s, "
                        "n_parent_routes=%d, stderr_tail=%r)",
                        name,
                        sp.num_loc,
                        rc,
                        sp.n_parent_routes,
                        _tail(sp_err, 2048),
                    )
                # Sub_improved check: compare sub-tour cost against
                # the parent route cost. Both are in the same
                # LKH_SCALING_FACTOR units (scaled by 100_000) so a
                # direct comparison is valid. The subproblem's
                # distance matrix is identical to the parent's for
                # the customer pairs that appear in this cluster.
                sub_cost_scaled: int | None = None
                parent_cost_scaled: int | None = None
                sub_improved: bool | None = None
                meta = parse_tour_with_cost(sp_tour)
                if meta is not None:
                    _, sub_cost_scaled, _ = meta
                if sub_cost_scaled is not None and sp.parent_routes:
                    # Compute parent route cost in the same scaled
                    # units, using the subproblem's distance matrix.
                    # We extract distances from the TSPLIB file.
                    parent_cost_scaled = _compute_parent_route_cost_scaled(
                        sp_vrp, sp.parent_routes
                    )
                    if parent_cost_scaled is not None:
                        # Strict improvement (sub < parent). Equal
                        # counts as not-improved (sub didn't strictly
                        # beat parent).
                        sub_improved = sub_cost_scaled < parent_cost_scaled
                log.info(
                    "sub_done(name=%s, sub_size=%d, rc=%s, "
                    "n_parent_routes=%d, warm_started=%s, "
                    "sub_cost=%s, parent_cost=%s, "
                    "sub_improved=%s, routes=%d, wallclock=%.2fs)",
                    name,
                    sp.num_loc,
                    rc,
                    sp.n_parent_routes,
                    sp.n_parent_routes > 1,
                    sub_cost_scaled,
                    parent_cost_scaled,
                    sub_improved,
                    len(routes),
                    wallclock,
                )
                return sp, routes
            except subprocess.TimeoutExpired:
                log.error(
                    "sub_timeout(name=%s, sub_size=%d, per_sub_s=%.1f, "
                    "n_parent_routes=%d)",
                    name,
                    sp.num_loc,
                    per_sub_s,
                    sp.n_parent_routes,
                )
                return sp, []
            except Exception as e:  # noqa: BLE001
                log.exception(
                    "sub_raised(name=%s, sub_size=%d, n_parent_routes=%d): %s",
                    name,
                    sp.num_loc,
                    sp.n_parent_routes,
                    e,
                )
                return sp, []
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        try:
            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
                results = list(ex.map(_one, subproblems))
        except Exception as e:  # noqa: BLE001
            log.exception("subs_executor_failed(name=%s): %s", name, e)
            results = [(sp, []) for sp in subproblems]
        finally:
            # Kill the master so we can restart it with the
            # warm-start initial tour built from the stitched
            # result.  This re-enables the kill+restart cycle that
            # LKH-3's CVRP dimension expansion forced us to disable
            # in a previous iteration.
            self._stop_master()
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _routes_zero_indexed(
    routes: list[list[int]],
    depot_id: int = 1,
    num_loc: int | None = None,
) -> list[list[int]]:
    """Convert LKH 1-indexed routes to 0-indexed customer-only routes.

    Drops the depot (at id = depot_id) and phantom depots (ids > num_loc+1
    in the Uchoa CVRP convention). Subtracts 2 to convert from Uchoa
    (customer ids 2..N+1) to 0-indexed (0..N-1).
    """
    out: list[list[int]] = []
    for r in routes:
        cust = [
            n - 2 for n in r if n > depot_id and (num_loc is None or n <= num_loc + 1)
        ]
        if cust:
            out.append(cust)
    return out


def _compute_parent_route_cost_scaled(
    vrp_path: str,
    parent_routes_1indexed: list[list[int]],
) -> int | None:
    """Sum the edge costs of the parent routes using the distance matrix
    embedded in the subproblem's TSPLIB CVRP file.

    Returns the cost in the same scaled units LKH-3 uses internally
    (``LKH_SCALING_FACTOR = 100_000`` — see ``lkh_format.py``), so a
    direct comparison with the sub-tour cost (parsed from
    ``parse_tour_with_cost``) is valid.

    The parent routes are 1-indexed Uchoa customer ids in the
    subproblem's local id space (depot = 1, customers = 2..N+1).
    Each route's cost is the sum of edges from depot→c1, c1→c2, ...,
    ck→depot. We round to the nearest integer to match LKH-3's
    integer cost accounting.

    Returns ``None`` if the TSPLIB file cannot be parsed or the
    EDGE_WEIGHT_SECTION is missing/malformed.
    """
    try:
        with open(vrp_path, "r") as f:
            text = f.read()
    except OSError:
        return None

    # Find DIMENSION and EDGE_WEIGHT_SECTION.
    dimension: int | None = None
    section_idx = text.find("EDGE_WEIGHT_SECTION")
    if section_idx < 0:
        return None
    header = text[:section_idx]
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("DIMENSION"):
            try:
                dimension = int(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                return None
            break
    if dimension is None:
        return None

    # Parse the FULL_MATRIX rows.
    body = text[section_idx:].split("\n", 1)[1]
    rows: list[list[int]] = []
    current: list[int] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line in ("-1", "EOF", "DEPOT_SECTION", "DEMAND_SECTION"):
            if current:
                rows.append(current)
                current = []
            if line in ("DEPOT_SECTION", "DEMAND_SECTION", "EOF"):
                break
            continue
        parts = line.split()
        for p in parts:
            try:
                current.append(int(p))
            except ValueError:
                # Skip non-integer tokens (e.g., a stray section name).
                continue
            if current and len(current) == dimension:
                rows.append(current)
                current = []
    if current:
        rows.append(current)
    if len(rows) != dimension:
        return None

    # Sum edge costs. Each parent route: depot(1) → c1 → c2 → ... → ck → depot(1).
    # We treat the depot as node index 0 in the matrix (i.e. id 1).
    total = 0
    for route in parent_routes_1indexed:
        if not route:
            continue
        prev = 1  # depot
        for nid in route:
            # 1-indexed Uchoa → 0-indexed matrix row.
            i = prev - 1
            j = int(nid) - 1
            if not (0 <= i < dimension and 0 <= j < dimension):
                return None
            total += int(rows[i][j])
            prev = int(nid)
        # Close the route: last customer → depot.
        i = prev - 1
        j = 0  # depot
        if not (0 <= i < dimension and 0 <= j < dimension):
            return None
        total += int(rows[i][j])
    return total


def _stitch_routes(
    sub_results: list[tuple[Subproblem, list[list[int]]]],
    td: TensorDict,
) -> list[list[int]] | None:
    """Stitch a list of (Subproblem, routes) pairs into a single master
    initial tour suitable for LKH-3's ``INITIAL_TOUR_FILE``.

    Format:
        The output is a flat 1-indexed master Uchoa tour of length
        ``num_loc + 1`` (1 depot + ``num_loc`` customers).  LKH-3's
        ``ReadTour`` validates the tour file's ``DIMENSION`` matches
        the problem and rejects entries that are out of range, duplicate,
        or phantom-depot ids.  Therefore:

        * The output is a *permutation* of ``[1..num_loc+1]`` —
          depot (1) at the start, then all customers (Uchoa ids
          2..num_loc+1) in some order.
        * LKH-3 figures out the route structure from this single
          sequence and the capacity constraints — we do *not* emit
          depot markers between routes (those would create duplicate
          ``1`` entries and trip the duplicate-id check).
        * We do not emit phantom depot ids in the master tour because
          their id range (``num_loc+2..num_loc+Salesmen``) is not
          known up front (LKH-3 picks the number of vehicles during
          search).

    Per-subproblem id mapping:
        For each subproblem's route, we drop the subproblem depot
        (id=1) and any phantom depots (ids > subproblem DIMENSION),
        then map each surviving subproblem-local customer id
        ``i`` (in 2..sub_dim) to the master Uchoa id
        ``sub.customer_ids[i-2] + 2``.
    """
    del td  # currently unused; kept for future per-instance hooks
    if not sub_results:
        return None
    out: list[int] = [1]  # start with the master depot
    for sub, routes in sub_results:
        n_sub = len(sub.customer_ids)
        if n_sub == 0:
            continue
        sub_dim = n_sub + 1  # subproblem DIMENSION = depot + n_sub customers
        if not routes:
            continue
        for r in routes:
            for nid in r:
                if nid == 1:
                    # Subproblem depot = master depot; already emitted
                    # as the master tour's start.
                    continue
                if nid > sub_dim:
                    # Phantom depot — drop.
                    continue
                # Map subproblem-local customer id to master Uchoa.
                master_id = sub.customer_ids[nid - 2] + 2
                out.append(master_id)
    # Sanity: must have at least the depot, plus all customers.
    # Note: a full coverage check (every master customer appears
    # exactly once) is the caller's responsibility, since the
    # orchestrator also needs to detect coverage gaps. Here we just
    # return whatever we have; coverage is logged in solve() via the
    # num_loc/len(out) ratio.
    if len(out) <= 1:
        return None
    return out


def _write_initial_tour(path: str, master_tour: list[int], depot_id: int = 1) -> None:
    """Write a TSPLIB TOUR file from a 1-indexed node sequence."""
    n = len(master_tour)
    lines = [
        f"NAME : {os.path.basename(path)}",
        f"DIMENSION : {n}",
        "TYPE : TOUR",
        "TOUR_SECTION",
    ]
    for nid in master_tour:
        lines.append(str(int(nid)))
    lines.extend(["-1", "EOF"])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
