"""alpha-nearness figure experiment.

For each randomly generated Euclidean TSP instance:
  1. Solve with farthest insertion (FI).
  2. Solve with LKH-2.
  3. Compute the (n, n) alpha-nearness matrix.
  4. Save a 1-row x 2-panel figure to {out_dir}/instance_{k:02d}.png:
     - left: FI tour with edges colored by alpha(i, j)
     - right: LKH tour drawn plain (no colormap)

Print summary statistics at the end (FI vs LKH length, mean alpha on
tour edges for FI and for LKH).

Usage::

    python scripts/run_alpha_nearness.py --num-instances 5 --num-nodes 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Allow running this script directly (``python scripts/run_alpha_nearness.py``).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize

from utils.alpha_nearness import compute_alpha_nearness
from utils.farthest_insertion import farthest_insertion_tsp
from utils.lkh_runner import DEFAULT_BINARY, solve_lkh_tsp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _tour_segments(perm: list[int]) -> np.ndarray:
    """Closed-cycle line segments for a tour permutation.

    Returns an ``(n, 2, 2)`` array where each entry is a ``[[x0, y0],
    [x1, y1]]`` segment. The route closes implicitly (segment ``n-1``
    returns to the first node).
    """
    cycle = np.asarray(perm, dtype=int)
    n = cycle.shape[0]
    starts = coords_lookup[cycle]                          # (n, 2)
    ends = coords_lookup[np.roll(cycle, -1)]               # (n, 2)
    return np.stack([starts, ends], axis=1)               # (n, 2, 2)


def _tour_edge_alphas(perm: list[int], alpha: np.ndarray) -> np.ndarray:
    """alpha value for each segment of the closed tour."""
    cycle = np.asarray(perm, dtype=int)
    n = cycle.shape[0]
    next_idx = np.roll(cycle, -1)
    return alpha[cycle, next_idx]


# Module-level lookup used by ``_tour_segments``. Set at the top of
# ``_plot_instance`` so we don't have to thread coords through both
# helpers.
coords_lookup: np.ndarray = np.empty((0, 2))


def _alpha_color_norm(
    tour_edge_alphas: np.ndarray, all_alphas: np.ndarray, percentile: float = 99.0
) -> tuple[Normalize | LogNorm, float]:
    """Pick a colormap normalization that makes tour-edge alpha visible.

    Tour-edge alphas are typically 5-10x smaller than the global alpha
    maximum (which is dominated by a few long, never-tour edges). Using
    the global max as ``vmax`` collapses the bulk of the colormap to a
    single dark color. To make differences visible we anchor ``vmax`` at
    the ``percentile``-th percentile of tour-edge alphas (capped at the
    global max to keep the colorbar physically meaningful). When the
    alpha values span more than an order of magnitude we additionally
    switch to a log scale.
    """
    # Anchor vmax to the tour-edge distribution so the bulk of edges
    # span the colormap. Cap at the global max so the colorbar is
    # physically meaningful even when the percentile is far below max.
    raw_vmax = float(np.percentile(tour_edge_alphas, percentile))
    global_max = float(all_alphas.max())
    if not np.isfinite(raw_vmax) or raw_vmax <= 0:
        vmax = 1.0
    else:
        vmax = max(raw_vmax, 1e-9)
    # If the data spans > 1 order of magnitude, use a log scale; the
    # tiny ``vmin`` floor lets us include alpha=0 (which logs to -inf
    # and is rendered fully transparent by matplotlib's LogNorm —
    # see the clamp below) at the bottom of the colormap.
    nonzero = tour_edge_alphas[tour_edge_alphas > 0]
    if nonzero.size >= 2:
        ratio = float(nonzero.max() / max(nonzero.min(), 1e-12))
    else:
        ratio = 1.0
    if ratio > 10.0:
        vmin = max(vmax * 1e-3, 1e-6)
        norm: Normalize | LogNorm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=0.0, vmax=vmax)
    return norm, vmax


def _alpha_for_plot(values: np.ndarray, norm: Normalize | LogNorm) -> np.ndarray:
    """Clamp alpha values so they map to a finite, opaque color.

    ``LogNorm(0)`` returns ``NaN``, which matplotlib renders as fully
    transparent RGBA — i.e. invisible. For alpha=0 edges (MST edges
    are guaranteed alpha=0), we clamp them to the colormap's ``vmin``
    so they get the darkest visible color instead of disappearing.
    """
    if isinstance(norm, LogNorm):
        vmin = float(norm.vmin)
        return np.maximum(values, vmin)
    return values


def _plot_instance(
    coords: np.ndarray,
    fi_perm: list[int],
    fi_len: float,
    lkh_perm: list[int] | None,
    lkh_len: float,
    alpha: np.ndarray,
    title_prefix: str,
) -> plt.Figure:
    """Build the 1x2 side-by-side panel figure for one instance."""
    global coords_lookup
    coords_lookup = coords
    fig, (ax_fi, ax_lkh) = plt.subplots(1, 2, figsize=(12, 5))

    fi_segs = _tour_segments(fi_perm)
    fi_alphas = _tour_edge_alphas(fi_perm, alpha)
    fi_alpha_mean = float(fi_alphas.mean())

    norm, _ = _alpha_color_norm(fi_alphas, alpha)
    fi_alphas_plot = _alpha_for_plot(fi_alphas, norm)

    lc_fi = LineCollection(
        fi_segs,
        cmap="viridis",
        norm=norm,
        array=fi_alphas_plot,
        linewidths=2.0,
        zorder=2,
    )
    ax_fi.add_collection(lc_fi)
    ax_fi.scatter(coords[:, 0], coords[:, 1], s=30, c="tab:red", zorder=4)
    for i, (x, y) in enumerate(coords):
        ax_fi.annotate(str(i), (x, y), fontsize=7, ha="center", va="bottom",
                       xytext=(0, 3), textcoords="offset points", zorder=5)
    ax_fi.set_aspect("equal")
    ax_fi.set_xlim(-0.05, 1.05)
    ax_fi.set_ylim(-0.05, 1.05)
    ax_fi.set_title(
        f"Farthest insertion  |  L = {fi_len:.3f}\n"
        f"mean alpha(tour) = {fi_alpha_mean:.4f}",
        fontsize=10,
    )
    cb = fig.colorbar(lc_fi, ax=ax_fi, fraction=0.046, pad=0.04)
    cb.set_label("alpha(i, j)", fontsize=9)

    # ---- LKH panel: plain black tour --------------------------------------
    if lkh_perm is not None:
        lkh_segs = _tour_segments(lkh_perm)
        lkh_alphas = _tour_edge_alphas(lkh_perm, alpha)
        lkh_alpha_mean = float(lkh_alphas.mean())
        lc_lkh = LineCollection(
            lkh_segs, color="black", linewidths=1.5, zorder=2
        )
        ax_lkh.add_collection(lc_lkh)
        ax_lkh.scatter(coords[:, 0], coords[:, 1], s=30, c="tab:red", zorder=4)
        for i, (x, y) in enumerate(coords):
            ax_lkh.annotate(str(i), (x, y), fontsize=7, ha="center",
                            va="bottom", xytext=(0, 3),
                            textcoords="offset points", zorder=5)
        ax_lkh.set_title(
            f"LKH-2  |  L = {lkh_len:.3f}\n"
            f"mean alpha(tour) = {lkh_alpha_mean:.4f}",
            fontsize=10,
        )
    else:
        ax_lkh.text(
            0.5, 0.5, "LKH-2 failed",
            ha="center", va="center", transform=ax_lkh.transAxes,
            fontsize=12, color="red",
        )
        ax_lkh.set_title("LKH-2 (failed)", fontsize=10)

    ax_lkh.set_aspect("equal")
    ax_lkh.set_xlim(-0.05, 1.05)
    ax_lkh.set_ylim(-0.05, 1.05)

    fig.suptitle(title_prefix, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="alpha-nearness figure experiment (FI vs LKH-2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-instances", type=int, default=5,
        help="number of random TSP instances",
    )
    p.add_argument(
        "--num-nodes", type=int, default=20,
        help="number of cities per instance",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="rl4co generator seed (and Python RNG seed for FI)",
    )
    p.add_argument(
        "--lkh-binary", type=str, default=DEFAULT_BINARY,
        help="absolute path to the LKH-2 executable",
    )
    p.add_argument(
        "--out-dir", type=str, default="figures",
        help="directory for output PNG files (created if missing)",
    )
    p.add_argument(
        "--max-trials", type=int, default=10_000,
        help="MAX_TRIALS in the .par file",
    )
    p.add_argument(
        "--lkh-time-limit", type=float, default=30.0,
        help="per-run TIME_LIMIT in seconds (floored at 1)",
    )
    p.add_argument(
        "--lkh-seed", type=int, default=1,
        help="SEED in the .par file",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()

    # rl4co is heavy — import lazily so the script still parses without it.
    import torch
    from rl4co.envs.routing.tsp.env import TSPEnv

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = TSPEnv(generator_params={"num_loc": args.num_nodes})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fi_lengths: list[float] = []
    lkh_lengths: list[float] = []
    fi_alpha_means: list[float] = []
    lkh_alpha_means: list[float] = []
    failures = 0

    for k in range(args.num_instances):
        td = env.reset(batch_size=[1])
        coords = td["locs"][0].cpu().numpy().astype(np.float64)
        n = coords.shape[0]

        fi_perm, fi_len = farthest_insertion_tsp(coords)
        fi_lengths.append(fi_len)

        lkh_perm, lkh_len = solve_lkh_tsp(
            coords,
            binary_path=args.lkh_binary,
            max_trials=args.max_trials,
            seed=args.lkh_seed,
            time_limit_s=args.lkh_time_limit,
        )
        if lkh_perm is None:
            failures += 1
            lkh_lengths.append(float("nan"))
        else:
            lkh_lengths.append(lkh_len)

        alpha = compute_alpha_nearness(coords, root=0)
        fi_alphas = _tour_edge_alphas(fi_perm, alpha)
        fi_alpha_means.append(float(fi_alphas.mean()))
        if lkh_perm is not None:
            lkh_alphas = _tour_edge_alphas(lkh_perm, alpha)
            lkh_alpha_means.append(float(lkh_alphas.mean()))
        else:
            lkh_alpha_means.append(float("nan"))

        fig = _plot_instance(
            coords, fi_perm, fi_len, lkh_perm,
            lkh_len if lkh_len != float("inf") else float("nan"),
            alpha,
            title_prefix=f"Instance {k:02d}  |  TSP-{n}",
        )
        out_path = out_dir / f"instance_{k:02d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        ratio_str = (
            f"FI/LKH = {fi_len / lkh_len:.3f}"
            if lkh_perm is not None and lkh_len > 0
            else "FI/LKH = n/a"
        )
        print(
            f"  [{k:02d}]  FI = {fi_len:.4f}   "
            f"LKH = {lkh_len if lkh_perm else 'fail':>10}   "
            f"{ratio_str}   "
            f"alpha_FI = {fi_alpha_means[-1]:.4f}   "
            f"alpha_LKH = {lkh_alpha_means[-1] if lkh_perm else float('nan'):.4f}"
        )

    # ---- summary ------------------------------------------------------------
    fi_arr = np.asarray(fi_lengths)
    lkh_arr = np.asarray(lkh_lengths)
    valid = ~np.isnan(lkh_arr) & (lkh_arr != float("inf"))
    print("\n=== Summary ===")
    print(
        f"  FI length:   mean = {fi_arr.mean():.4f}   std = {fi_arr.std():.4f}"
    )
    if valid.any():
        print(
            f"  LKH length:  mean = {lkh_arr[valid].mean():.4f}   "
            f"std = {lkh_arr[valid].std():.4f}   "
            f"({int(valid.sum())} successful, {failures} failed)"
        )
        ratios = fi_arr[valid] / lkh_arr[valid]
        print(
            f"  FI / LKH:    mean = {ratios.mean():.4f}   "
            f"std = {ratios.std():.4f}   "
            f"(expect 1.10-1.20 for TSP-20 on uniform [0, 1])"
        )
    fi_a = np.asarray(fi_alpha_means)
    lkh_a = np.asarray(lkh_alpha_means)
    print(
        f"  mean alpha on FI tour:    mean = {fi_a.mean():.4f}   "
        f"std = {fi_a.std():.4f}"
    )
    if valid.any():
        lkh_a_valid = lkh_a[valid]
        print(
            f"  mean alpha on LKH tour:   mean = {lkh_a_valid.mean():.4f}   "
            f"std = {lkh_a_valid.std():.4f}"
        )
        if lkh_a_valid.mean() < fi_a[valid].mean():
            print(
                "  -> LKH picks lower-alpha edges on average (metric "
                "correlates with quality)"
            )
        else:
            print(
                "  -> NOTE: LKH mean alpha is NOT lower than FI mean alpha "
                "on this batch; the metric may not discriminate on the "
                "given instances."
            )
    print(f"\nFigures written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())