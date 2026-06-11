"""Unified evaluation harness for any Solver.

Builds on rl4co.tasks.eval.* machinery via the _SolverAsPolicy adapter so
ClassicalSolver (and HybridSolver in stage 2) plug into the same evaluation
code path as RLSolver.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rl4co.data.dataset import TensorDictDataset
from rl4co.tasks.eval import (
    AugmentationEval,
    GreedyEval,
    GreedyMultiStartAugmentEval,
    GreedyMultiStartEval,
    SamplingEval,
    get_automatic_batch_size,
)
from tensordict import TensorDict
from torch.utils.data import DataLoader

from nrp.solvers import Solver
from nrp.utils.metrics import gap_to_optimal, tour_length_summary
from nrp.utils.pkl import save_versioned

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for an evaluation run's outputs."""

    per_instance: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# ----- method dispatch (mirrors rl4co/tasks/eval.py:350) -----

def _build_eval_fn(method: str, env, samples: int, num_augment: int,
                   softmax_temp: float, num_starts: int | None = None):
    """Build the rl4co EvalBase instance for the given method."""
    if method == "greedy":
        return GreedyEval(env)
    if method == "sampling":
        return SamplingEval(env, samples=samples, softmax_temp=softmax_temp)
    if method == "multistart_greedy":
        if num_starts is None:
            num_starts = getattr(env.generator, "num_loc", 1)
        return GreedyMultiStartEval(env, num_starts=num_starts)
    if method == "augment_dihedral_8":
        return AugmentationEval(env, num_augment=num_augment, force_dihedral_8=True)
    if method == "augment":
        return AugmentationEval(env, num_augment=num_augment)
    if method == "multistart_greedy_augment_dihedral_8":
        if num_starts is None:
            num_starts = getattr(env.generator, "num_loc", 1)
        return GreedyMultiStartAugmentEval(
            env, num_starts=num_starts, num_augment=num_augment, force_dihedral_8=True
        )
    if method == "multistart_greedy_augment":
        if num_starts is None:
            num_starts = getattr(env.generator, "num_loc", 1)
        return GreedyMultiStartAugmentEval(
            env, num_starts=num_starts, num_augment=num_augment
        )
    raise ValueError(
        f"Unknown eval method '{method}'. "
        f"Available: greedy, sampling, multistart_greedy, augment, "
        f"augment_dihedral_8, multistart_greedy_augment, multistart_greedy_augment_dihedral_8"
    )


# ----- the adapter that lets any Solver plug into rl4co EvalBase classes -----

class _SolverAsPolicy(nn.Module):
    """Adapter: wraps a Solver as an nn.Module with the policy interface.

    rl4co.tasks.eval.* expects a policy (an nn.Module) whose forward returns
    a dict-like with `actions`. This class adapts any nrp Solver to that shape.
    """

    def __init__(self, solver: Solver, env):
        super().__init__()
        self.solver = solver
        self.env = env
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, td, decode_type=None, num_starts=0, **kwargs):
        # We do not consume decode_type/num_starts here; the Solver was
        # constructed with its own decode_type/num_starts. The stage-1
        # contract is: call solver.solve(td) and return the result.
        out = self.solver.solve(td)
        # _SolverAsPolicy must return a dict-like; out is a TensorDict.
        return out


# ----- the main entry point -----

def evaluate(
    solver: Solver,
    env,
    dataset: TensorDict | TensorDictDataset,
    method: str = "augment_dihedral_8",
    batch_size: int | None = None,
    max_batch_size: int = 4096,
    samples: int = 1280,
    num_augment: int = 8,
    softmax_temp: float = 1.0,
    optima: Sequence[float] | None = None,
    wandb_run: Any | None = None,
    save_dir: str | Path | None = None,
    run_id: str | None = None,
    warmup: bool = True,
) -> EvaluationResult:
    """Run a solver on a dataset using one of the standard eval methods.

    Args:
        solver: a `nrp.solvers.Solver` instance.
        env: the RL4CO env (used to construct the EvalBase class).
        dataset: a `TensorDict` or `TensorDictDataset` of instances.
        method: one of the standard eval methods.
        batch_size: override auto batch sizing.
        max_batch_size: cap for auto batch sizing.
        samples: for `sampling` method.
        num_augment: for `augment_*` methods.
        softmax_temp: for `sampling` method.
        optima: optional per-instance optimal tour lengths; enables gap_to_opt_pct.
        wandb_run: optional wandb run for logging.
        save_dir: directory to write the .pkl result.
        run_id: filename stem for the .pkl result.
        warmup: if True, run one batch first to warm caches (excluded from timing).

    Returns:
        EvaluationResult with per_instance rows, summary stats, and metadata.
    """
    if isinstance(dataset, TensorDict):
        ds = TensorDictDataset(dataset)
    else:
        ds = dataset

    eval_fn = _build_eval_fn(
        method, env, samples=samples, num_augment=num_augment,
        softmax_temp=softmax_temp,
    )

    if batch_size is None:
        try:
            batch_size = get_automatic_batch_size(eval_fn, max_batch_size=max_batch_size)
        except Exception:
            batch_size = max_batch_size

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=ds.collate_fn,
    )

    if warmup and len(loader) > 0:
        try:
            first_batch = next(iter(loader))
            solver.warmup(first_batch.to(solver.device))
        except Exception:
            pass

    from .timing import timer

    per_instance: list[dict] = []
    n_done = 0
    t0 = time.perf_counter()
    with timer(sync_cuda=True):
        for batch_idx, td in enumerate(loader):
            td = td.to(solver.device)
            try:
                td = env.reset(td)
            except Exception:
                pass
            B = td.batch_size[0] if hasattr(td, "batch_size") and td.batch_size else 1
            try:
                out = solver.solve(td)
                _ = out["actions"]
                reward = out["reward"]
                # Optional per-instance feasible mask. A solver may attach
                # ``out["feasible"]`` (bool tensor of shape [B]) to mark
                # specific instances as not having produced a real tour
                # (e.g. LKH-3 timed out before writing final.tour). When
                # present, those rows are recorded as failed but the
                # remaining instances in the batch keep their reward.
                if "feasible" in out.keys():
                    feasible_t = out["feasible"]
                    if hasattr(feasible_t, "detach"):
                        feasible_t = feasible_t.detach()
                    feasible_list = (
                        feasible_t.flatten().cpu().tolist()
                        if hasattr(feasible_t, "flatten")
                        else list(feasible_t)
                    )
                    if len(feasible_list) != B:
                        # Solver returned a mismatched mask; fall back to
                        # legacy all-feasible behaviour rather than crash.
                        log.warning(
                            "feasible_mask_length_mismatch(batch_idx=%d, "
                            "B=%d, mask_len=%d)",
                            batch_idx,
                            B,
                            len(feasible_list),
                        )
                        feasible_list = [True] * B
                else:
                    feasible_list = [True] * B
            except Exception as e:
                for i in range(B):
                    per_instance.append({
                        "batch_idx": batch_idx,
                        "instance_idx": n_done + i,
                        "tour_length": float("nan"),
                        "reward": float("nan"),
                        "feasible": False,
                        "error": str(e),
                    })
                n_done += B
                continue
            # Per-instance rows. Honour the per-instance feasible mask so a
            # single miss does not poison the rest of the batch.
            reward_list = reward.flatten().cpu().tolist() if reward.dim() > 0 else [float(reward.item())]
            for i in range(B):
                r = reward_list[i] if i < len(reward_list) else float("nan")
                ok = bool(feasible_list[i]) if i < len(feasible_list) else True
                if ok:
                    per_instance.append({
                        "batch_idx": batch_idx,
                        "instance_idx": n_done + i,
                        "reward": float(r),
                        "tour_length": float(-r),
                        "feasible": True,
                    })
                else:
                    per_instance.append({
                        "batch_idx": batch_idx,
                        "instance_idx": n_done + i,
                        "tour_length": float("nan"),
                        "reward": float("nan"),
                        "feasible": False,
                        "error": "solver_returned_no_tour",
                    })
            n_done += B
    elapsed = time.perf_counter() - t0

    tour_lengths = [r["tour_length"] for r in per_instance]
    summary = tour_length_summary(tour_lengths)
    summary["wallclock_total_s"] = elapsed
    summary["wallclock_per_instance_s"] = elapsed / max(1, len(per_instance))
    summary["method"] = method
    summary["solver_name"] = solver.name
    summary["env_name"] = env.name
    summary["num_loc"] = getattr(env.generator, "num_loc", None)
    summary["num_instances"] = len(per_instance)

    if optima is not None and len(optima) == len(per_instance):
        gaps = gap_to_optimal(
            [r["tour_length"] for r in per_instance],
            list(optima),
        )
        for r, g in zip(per_instance, gaps):
            r["gap_to_opt_pct"] = g
        valid_gaps = [g for r, g in zip(per_instance, gaps) if r.get("feasible")]
        if valid_gaps:
            summary["mean_gap_to_opt_pct"] = sum(valid_gaps) / len(valid_gaps)

    metadata = {
        "solver_name": solver.name,
        "solver_type": type(solver).__name__,
        "env_name": env.name,
        "num_loc": getattr(env.generator, "num_loc", None),
        "method": method,
        "batch_size": batch_size,
        "num_instances": len(per_instance),
        "elapsed_s": elapsed,
    }
    result = EvaluationResult(
        per_instance=per_instance,
        summary=summary,
        metadata=metadata,
    )

    if wandb_run is not None:
        try:
            import wandb

            cols = ["instance_idx", "tour_length", "feasible", "gap_to_opt_pct"]
            table = wandb.Table(columns=cols)
            for r in per_instance[:1000]:
                table.add_data(
                    r.get("instance_idx", 0),
                    r.get("tour_length", float("nan")),
                    r.get("feasible", False),
                    r.get("gap_to_opt_pct", float("nan")),
                )
            wandb_run.log({"per_instance": table, **summary, **metadata})
        except Exception as e:
            import warnings
            warnings.warn(f"W&B logging failed: {e}", stacklevel=2)

    if save_dir is not None and run_id is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = save_dir / f"{run_id}.pkl"
        save_versioned(result, pkl_path, schema="evaluation_result")

    return result
