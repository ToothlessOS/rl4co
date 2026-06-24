"""alpha-nearness + NeuroLKH edge-score figure experiment.

For each randomly generated Euclidean TSP instance:
  1. Solve with farthest insertion (FI).
  2. Solve with LKH-2.
  3. Compute the (n, n) alpha-nearness matrix.
  4. Predict the (n, n) NeuroLKH score matrix.
  5. Save a 2-row x 2-panel figure to {out_dir}/instance_{k:02d}.png:
     - row 0 (FI tour):  α-colored | NeuroLKH-colored
     - row 1 (LKH tour): α-colored | NeuroLKH-colored

Print summary statistics at the end (FI vs LKH length, mean α and mean
NeuroLKH score on tour edges for FI and for LKH).

Usage::

    python scripts/run_alpha_nearness.py --num-instances 5 --num-nodes 100
    python scripts/run_alpha_nearness.py --num-instances 5 --num-nodes 50 \
        --no-neurolkh              # stage-1 fallback (1x2 figure)
    python scripts/run_alpha_nearness.py --neurolkh-device cpu   # no CUDA
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


# Module-level lookup used by ``_tour_segments``. Set at the top of
# ``_plot_instance`` so we don't have to thread coords through both
# helpers.
coords_lookup: np.ndarray = np.empty((0, 2))


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


def _tour_edge_values(perm: list[int], matrix: np.ndarray) -> np.ndarray:
    """Per-segment values for a closed tour from an arbitrary (n, n) matrix."""
    cycle = np.asarray(perm, dtype=int)
    next_idx = np.roll(cycle, -1)
    return matrix[cycle, next_idx]


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


def _neurolkh_color_norm(
    tour_edge_scores: np.ndarray, percentile: float = 99.0
) -> tuple[Normalize, float]:
    """Pick a colormap normalization for NeuroLKH scores.

    Same percentile-99 anchoring trick as `_alpha_color_norm`, but no
    LogNorm: scores are probabilities in (0, 1] and their dynamic range
    on a tour is typically only one order of magnitude, so a linear
    Normalize works.
    """
    finite = tour_edge_scores[np.isfinite(tour_edge_scores)]
    if finite.size == 0:
        # No scored edges at all — fall back to a sensible default so
        # matplotlib doesn't choke.
        return Normalize(vmin=0.0, vmax=1.0), 1.0
    raw_vmax = float(np.percentile(finite, percentile))
    if not np.isfinite(raw_vmax) or raw_vmax <= 0:
        vmax = 1.0
    else:
        # Cap at 1.0 because softmax outputs can occasionally exceed 1
        # by a tiny epsilon due to numerical drift.
        vmax = min(max(raw_vmax, 1e-3), 1.0)
    return Normalize(vmin=0.0, vmax=vmax), vmax


def _neurolkh_for_plot(
    values: np.ndarray, vmin: float = 0.0
) -> tuple[np.ndarray, int]:
    """Replace NaN tour-edge scores with ``vmin`` for the colormap.

    Returns ``(values_filled, n_missing)``. NaN values arise for tour
    edges that neither endpoint named in its 20-NN graph. Filling with
    ``vmin`` (rather than the mean) makes unscored edges appear at the
    bottom of the colormap, visually distinguishing them from scored
    edges — and avoids the "everything is medium-yellow" effect of
    filling with the mean on a sequential cmap. The panel title
    reports the count of unscored edges.
    """
    missing_mask = ~np.isfinite(values)
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        return values, 0
    out = values.copy()
    out[missing_mask] = vmin
    return out, n_missing


def _scatter_nodes(ax, coords: np.ndarray) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=30, c="tab:red", zorder=4)
    for i, (x, y) in enumerate(coords):
        ax.annotate(
            str(i), (x, y), fontsize=7, ha="center", va="bottom",
            xytext=(0, 3), textcoords="offset points", zorder=5,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


def _draw_tour_panel(
    ax,
    perm: list[int],
    edge_values: np.ndarray,
    visible_mask: np.ndarray,
    norm,
    cmap: str,
    *,
    title: str,
    linewidth: float = 2.0,
) -> tuple[LineCollection | None, int]:
    """Draw a colored tour with only the segments where ``visible_mask`` is True.

    ``edge_values`` is the per-segment value array (length n) for the
    closed tour. ``visible_mask`` (length n) selects which segments to
    draw — used to filter the NeuroLKH panels to only "tour edges that
    are also in the top-K candidates" (per the user's request). The α
    panels pass ``np.ones(n, dtype=bool)`` to draw every edge.

    Edges that are masked out are not drawn at all — they appear as gaps
    in the tour, which is the visual signal that the model "disagrees"
    with that edge.

    Returns ``(line_collection, n_visible)``. ``line_collection`` is the
    LineCollection carrying the colormap (for colorbar attachment); it
    is ``None`` when no segments are visible.
    """
    segs = _tour_segments(perm)
    visible_mask = np.asarray(visible_mask, dtype=bool)
    n_visible = int(visible_mask.sum())

    if n_visible == 0:
        _scatter_nodes(ax, coords_lookup)
        ax.set_title(title, fontsize=10)
        return None, 0

    lc = LineCollection(
        segs[visible_mask],
        cmap=cmap, norm=norm,
        array=edge_values[visible_mask],
        linewidths=linewidth,
        linestyles="-",
        zorder=2,
    )
    ax.add_collection(lc)
    _scatter_nodes(ax, coords_lookup)
    ax.set_title(title, fontsize=10)
    return lc, n_visible


def _draw_neuro_panel(
    fig: plt.Figure,
    ax,
    perm: list[int],
    score_matrix: np.ndarray,
    topk_mask: np.ndarray,
    norm,
    *,
    tour_label: str,
    tour_len: float,
    topk: int,
    colorbar_label: str,
    panel_label: str,
) -> tuple[LineCollection | None, int]:
    """Draw a NeuroLKH-colored tour panel and attach a colorbar.

    Wraps :func:`_draw_tour_panel` with the standard title format
    used by both columns 1 and 2 of the stage-3 figure (and the single
    column 1 of the stage-2 figure). Returns ``(lc, n_visible)``.
    """
    tour_edge_vals = _tour_edge_values(perm, score_matrix)
    lc, n_topk = _draw_tour_panel(
        ax, perm, tour_edge_vals, topk_mask, norm, cmap="viridis",
        title=f"{tour_label} / {panel_label}  |  L = {tour_len:.3f}",
    )
    in_topk_vals = tour_edge_vals[topk_mask]
    finite_in_topk = in_topk_vals[np.isfinite(in_topk_vals)]
    n_tour = len(perm)
    mean_str = (
        f"{float(np.nanmean(finite_in_topk)):.4f}"
        if finite_in_topk.size
        else "nan"
    )
    ax.set_title(
        f"{tour_label} / {panel_label}  |  L = {tour_len:.3f}\n"
        f"{n_topk}/{n_tour} tour edges in top-{topk}  |  "
        f"mean NeuroLKH(in-topk) = {mean_str}",
        fontsize=9,
    )
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(colorbar_label, fontsize=9)
    return lc, n_topk


def _plot_instance(
    coords: np.ndarray,
    fi_perm: list[int],
    fi_len: float,
    lkh_perm: list[int] | None,
    lkh_len: float,
    alpha: np.ndarray,
    neurolkh: np.ndarray | None,
    topk: int,
    title_prefix: str,
    augmented_scores: list[np.ndarray | None] | None = None,
) -> plt.Figure:
    """Build the per-instance figure.

    Layout depends on whether ``augmented_scores`` is provided:

    - ``augmented_scores is None``: 2x2 panel (stage-2 fallback).
    - ``augmented_scores = [fi_aug, lkh_aug]``: 2x3 panel (stage-3).
      The third column shows NeuroLKH scores on the **tour-augmented**
      20-graph (per-row: FI-aug for the FI row, LKH-aug for the LKH
      row), filtered to tour ∩ top-K.

    Per-row coloring convention matches the existing implementation:
    column 0 colors all tour edges by α; columns 1 (and 2) only draw
    tour edges that are also in some node's top-K candidates — the
    rest appear as gaps, signalling where the model "disagrees" with
    that edge.

    Color scales are shared across rows within a column: the
    α-panels share one ``alpha_norm``, the NeuroLKH-20NN panels share
    one ``neuro_norm``, and (in stage-3 mode) the NeuroLKH-augmented
    panels share a third ``neuro_aug_norm``.
    """
    global coords_lookup
    coords_lookup = coords

    n_cols = 3 if augmented_scores is not None else 2
    figsize = (20, 11) if augmented_scores is not None else (14, 11)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.18, "hspace": 0.22},
    )
    ax_fi_a, ax_fi_n = axes[0, 0], axes[0, 1]
    ax_lkh_a, ax_lkh_n = axes[1, 0], axes[1, 1]
    if augmented_scores is not None:
        ax_fi_aug = axes[0, 2]
        ax_lkh_aug = axes[1, 2]
    else:
        ax_fi_aug = ax_lkh_aug = None  # type: ignore[assignment]

    n = len(fi_perm)
    all_visible = np.ones(n, dtype=bool)

    # ---- α color scale (shared across both rows) ---------------------------
    fi_alphas = _tour_edge_values(fi_perm, alpha)
    fi_alpha_mean = float(np.nanmean(fi_alphas))
    if lkh_perm is not None:
        lkh_alphas = _tour_edge_values(lkh_perm, alpha)
        lkh_alpha_mean = float(np.nanmean(lkh_alphas))
        combined_alphas = np.concatenate([fi_alphas, lkh_alphas])
    else:
        lkh_alphas = None
        lkh_alpha_mean = float("nan")
        combined_alphas = fi_alphas
    alpha_norm, _ = _alpha_color_norm(combined_alphas, alpha)
    fi_alphas_plot = _alpha_for_plot(fi_alphas, alpha_norm)
    lkh_alphas_plot = (
        _alpha_for_plot(lkh_alphas, alpha_norm) if lkh_alphas is not None else None
    )

    # ---- NeuroLKH top-K + color scale (shared across both rows) ------------
    from utils.neurolkh_runner import compute_topk_mask

    fi_neuro: np.ndarray | None = None
    lkh_neuro: np.ndarray | None = None
    fi_neuro_mean = lkh_neuro_mean = float("nan")
    fi_in_topk: np.ndarray | None = None
    lkh_in_topk: np.ndarray | None = None
    neuro_norm = None
    if neurolkh is not None:
        topk_mask = compute_topk_mask(neurolkh, k=topk)

        fi_neuro = _tour_edge_values(fi_perm, neurolkh)
        fi_neuro_mean = float(np.nanmean(fi_neuro))
        fi_in_topk = topk_mask[fi_perm, np.roll(fi_perm, -1)]

        if lkh_perm is not None:
            lkh_neuro = _tour_edge_values(lkh_perm, neurolkh)
            lkh_neuro_mean = float(np.nanmean(lkh_neuro))
            lkh_in_topk = topk_mask[lkh_perm, np.roll(lkh_perm, -1)]
        else:
            lkh_neuro = None
            lkh_neuro_mean = float("nan")
            lkh_in_topk = None

        # Compute the color norm from the union of *plotted* values
        # (top-K ∩ scored) so the colorbar matches what we see.
        plotted_vals = []
        if fi_in_topk is not None:
            v = fi_neuro[fi_in_topk & np.isfinite(fi_neuro)]
            if v.size:
                plotted_vals.append(v)
        if lkh_in_topk is not None:
            v = lkh_neuro[lkh_in_topk & np.isfinite(lkh_neuro)]
            if v.size:
                plotted_vals.append(v)
        if plotted_vals:
            neuro_norm, _ = _neurolkh_color_norm(np.concatenate(plotted_vals))
        else:
            neuro_norm = Normalize(vmin=0, vmax=1)

    # ---- Augmented-graph color scales (stage-3 only) ----------------------
    fi_aug_score = lkh_aug_score = None
    fi_aug_in_topk = lkh_aug_in_topk = None
    neuro_aug_norm = None
    if augmented_scores is not None:
        fi_aug_score, lkh_aug_score = (
            augmented_scores[0],
            augmented_scores[1] if len(augmented_scores) > 1 else None,
        )
        if fi_aug_score is not None:
            topk_fi_aug = compute_topk_mask(fi_aug_score, k=topk)
            fi_aug_in_topk = topk_fi_aug[fi_perm, np.roll(fi_perm, -1)]
        if lkh_aug_score is not None and lkh_perm is not None:
            topk_lkh_aug = compute_topk_mask(lkh_aug_score, k=topk)
            lkh_aug_in_topk = topk_lkh_aug[lkh_perm, np.roll(lkh_perm, -1)]

        plotted_aug = []
        if fi_aug_score is not None and fi_aug_in_topk is not None:
            v_fi_aug = _tour_edge_values(fi_perm, fi_aug_score)
            v = v_fi_aug[fi_aug_in_topk & np.isfinite(v_fi_aug)]
            if v.size:
                plotted_aug.append(v)
        if lkh_aug_score is not None and lkh_aug_in_topk is not None:
            v_lkh_aug = _tour_edge_values(lkh_perm, lkh_aug_score)
            v = v_lkh_aug[lkh_aug_in_topk & np.isfinite(v_lkh_aug)]
            if v.size:
                plotted_aug.append(v)
        if plotted_aug:
            neuro_aug_norm, _ = _neurolkh_color_norm(np.concatenate(plotted_aug))
        else:
            neuro_aug_norm = Normalize(vmin=0, vmax=1)

    # ---- Row 0 (FI tour) ---------------------------------------------------
    lc_fi_a, _ = _draw_tour_panel(
        ax_fi_a, fi_perm, fi_alphas_plot, all_visible, alpha_norm, cmap="viridis",
        title=f"FI / α-colored  |  L = {fi_len:.3f}\nmean α(tour) = {fi_alpha_mean:.4f}",
    )
    cb_fi_a = fig.colorbar(lc_fi_a, ax=ax_fi_a, fraction=0.046, pad=0.02)
    cb_fi_a.set_label("alpha(i, j)", fontsize=9)

    if neurolkh is not None and fi_in_topk is not None:
        _draw_neuro_panel(
            fig, ax_fi_n, fi_perm, neurolkh, fi_in_topk, neuro_norm,
            tour_label="FI", tour_len=fi_len, topk=topk,
            colorbar_label="NeuroLKH score (softmax over 20-NN)",
            panel_label=f"NeuroLKH top-{topk}",
        )
    else:
        # Stage-1 fallback: plain black FI tour (no colorbar).
        for coll in list(ax_fi_n.collections):
            coll.remove()
        ax_fi_n.add_collection(LineCollection(
            _tour_segments(fi_perm),
            colors="black", linewidths=1.5, linestyles="-", zorder=2,
        ))
        ax_fi_n.scatter(
            coords[:, 0], coords[:, 1], s=30, c="tab:red", zorder=4,
        )
        ax_fi_n.set_aspect("equal")
        ax_fi_n.set_xlim(-0.05, 1.05)
        ax_fi_n.set_ylim(-0.05, 1.05)
        ax_fi_n.set_title(
            f"FI / plain  |  L = {fi_len:.3f}", fontsize=10,
        )

    # ---- Stage-3: augmented panel for FI row ------------------------------
    if ax_fi_aug is not None:
        if fi_aug_score is not None and fi_aug_in_topk is not None:
            _draw_neuro_panel(
                fig, ax_fi_aug, fi_perm, fi_aug_score, fi_aug_in_topk,
                neuro_aug_norm,
                tour_label="FI", tour_len=fi_len, topk=topk,
                colorbar_label="NeuroLKH score (softmax over tour-aug 20)",
                panel_label=f"NeuroLKH aug top-{topk}",
            )
        else:
            ax_fi_aug.text(
                0.5, 0.5, "FI aug N/A",
                ha="center", va="center", transform=ax_fi_aug.transAxes,
                fontsize=12, color="red",
            )
            ax_fi_aug.set_aspect("equal")
            ax_fi_aug.set_xlim(-0.05, 1.05)
            ax_fi_aug.set_ylim(-0.05, 1.05)
            ax_fi_aug.set_title("FI / NeuroLKH aug (n/a)", fontsize=10)

    # ---- Row 1 (LKH tour) --------------------------------------------------
    if lkh_perm is not None:
        lc_lkh_a, _ = _draw_tour_panel(
            ax_lkh_a, lkh_perm, lkh_alphas_plot, all_visible, alpha_norm, cmap="viridis",
            title=(
                f"LKH-2 / α-colored  |  L = {lkh_len:.3f}\n"
                f"mean α(tour) = {lkh_alpha_mean:.4f}"
            ),
        )
        cb_lkh_a = fig.colorbar(lc_lkh_a, ax=ax_lkh_a, fraction=0.046, pad=0.02)
        cb_lkh_a.set_label("alpha(i, j)", fontsize=9)

        if neurolkh is not None and lkh_in_topk is not None:
            _draw_neuro_panel(
                fig, ax_lkh_n, lkh_perm, neurolkh, lkh_in_topk, neuro_norm,
                tour_label="LKH-2", tour_len=lkh_len, topk=topk,
                colorbar_label="NeuroLKH score (softmax over 20-NN)",
                panel_label=f"NeuroLKH top-{topk}",
            )
        else:
            for coll in list(ax_lkh_n.collections):
                coll.remove()
            ax_lkh_n.add_collection(LineCollection(
                _tour_segments(lkh_perm),
                colors="black", linewidths=1.5, linestyles="-", zorder=2,
            ))
            ax_lkh_n.scatter(
                coords[:, 0], coords[:, 1], s=30, c="tab:red", zorder=4,
            )
            ax_lkh_n.set_aspect("equal")
            ax_lkh_n.set_xlim(-0.05, 1.05)
            ax_lkh_n.set_ylim(-0.05, 1.05)
            ax_lkh_n.set_title(
                f"LKH-2 / plain  |  L = {lkh_len:.3f}", fontsize=10,
            )

        # Stage-3: augmented panel for LKH row
        if ax_lkh_aug is not None:
            if lkh_aug_score is not None and lkh_aug_in_topk is not None:
                _draw_neuro_panel(
                    fig, ax_lkh_aug, lkh_perm, lkh_aug_score, lkh_aug_in_topk,
                    neuro_aug_norm,
                    tour_label="LKH-2", tour_len=lkh_len, topk=topk,
                    colorbar_label="NeuroLKH score (softmax over tour-aug 20)",
                    panel_label=f"NeuroLKH aug top-{topk}",
                )
            else:
                ax_lkh_aug.text(
                    0.5, 0.5, "LKH aug N/A",
                    ha="center", va="center", transform=ax_lkh_aug.transAxes,
                    fontsize=12, color="red",
                )
                ax_lkh_aug.set_aspect("equal")
                ax_lkh_aug.set_xlim(-0.05, 1.05)
                ax_lkh_aug.set_ylim(-0.05, 1.05)
                ax_lkh_aug.set_title("LKH-2 / NeuroLKH aug (n/a)", fontsize=10)
    else:
        # LKH failed: row 1 falls back to plain messages.
        for ax in (ax_lkh_a, ax_lkh_n):
            ax.text(
                0.5, 0.5, "LKH-2 failed",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="red",
            )
            ax.set_title("LKH-2 (failed)", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
        if ax_lkh_aug is not None:
            ax_lkh_aug.text(
                0.5, 0.5, "LKH-2 failed",
                ha="center", va="center", transform=ax_lkh_aug.transAxes,
                fontsize=12, color="red",
            )
            ax_lkh_aug.set_title("LKH-2 aug (failed)", fontsize=10)
            ax_lkh_aug.set_aspect("equal")
            ax_lkh_aug.set_xlim(-0.05, 1.05)
            ax_lkh_aug.set_ylim(-0.05, 1.05)

    fig.suptitle(title_prefix, fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="alpha + NeuroLKH figure experiment (FI vs LKH-2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-instances", type=int, default=5,
        help="number of random TSP instances",
    )
    p.add_argument(
        "--num-nodes", type=int, default=100,
        help="cities per instance (NeuroLKH pretrained model expects n=100)",
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
    # ---- Stage-2 (NeuroLKH) --------------------------------------------------
    p.add_argument(
        "--no-neurolkh", action="store_true",
        help="skip NeuroLKH scoring (renders plain-black tour in the right column)",
    )
    p.add_argument(
        "--neurolkh-binary", type=str, default=None,
        help="absolute path to the NeuroLKH LKH-3 binary "
             "(auto-built at NeuroLKH/LKH if missing)",
    )
    p.add_argument(
        "--neurolkh-checkpoint", type=str, default=None,
        help="absolute path to the NeuroLKH pretrained checkpoint (.pt)",
    )
    p.add_argument(
        "--neurolkh-device", type=str, default=None,
        help="torch device for SGN inference ('cuda' or 'cpu'); "
             "defaults to cuda if available else cpu",
    )
    p.add_argument(
        "--neurolkh-topk", type=int, default=5,
        help="Per-node top-K candidates to visualize. Only tour edges that "
             "appear in some node's top-K are drawn in the NeuroLKH panels; "
             "the rest appear as gaps. Defaults to 5 (matches the upstream "
             "LKH candidate-set size).",
    )
    # ---- Stage-3 (tour-augmented NeuroLKH) --------------------------------
    p.add_argument(
        "--no-augment", action="store_true",
        help="disable the tour-augmented third column. Default is on "
             "whenever NeuroLKH is enabled. The pretrained model is "
             "shape-locked to n_edges=20; stage-3 swaps slot contents "
             "(forces the tour edges in), it does NOT add slots.",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()

    # rl4co is heavy — import lazily so the script still parses without it.
    import torch
    from rl4co.envs.routing.tsp.env import TSPEnv

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Optional NeuroLKH setup --------------------------------------------
    neurolkh_enabled = not args.no_neurolkh
    neurolkh_score_fn = None
    neurolkh_model = None
    neurolkh_device = None

    if neurolkh_enabled:
        from utils.neurolkh_runner import (
            build_neurolkh_binary_if_needed,
            generate_20nn_features,
            load_neurolkh_model,
            predict_edge_scores,
            predict_edge_scores_augmented,
            _DEFAULT_LKH_BINARY,
            _DEFAULT_CHECKPOINT,
        )

        if args.num_nodes < 20:
            print(
                f"[WARN] --num-nodes={args.num_nodes} is below the 20-NN graph "
                f"size NeuroLKH assumes; NeuroLKH will be skipped."
            )
            neurolkh_enabled = False
        else:
            lkh_binary = Path(args.neurolkh_binary) if args.neurolkh_binary else _DEFAULT_LKH_BINARY
            checkpoint = (
                Path(args.neurolkh_checkpoint)
                if args.neurolkh_checkpoint
                else _DEFAULT_CHECKPOINT
            )

            print(f"[setup] Building NeuroLKH LKH-3 binary at {lkh_binary} (if missing)…")
            build_neurolkh_binary_if_needed()

            if not checkpoint.exists():
                print(
                    f"[WARN] NeuroLKH checkpoint not found at {checkpoint}; "
                    f"NeuroLKH will be skipped."
                )
                neurolkh_enabled = False
            else:
                if args.neurolkh_device is not None:
                    neurolkh_device = args.neurolkh_device
                elif torch.cuda.is_available():
                    neurolkh_device = "cuda"
                else:
                    neurolkh_device = "cpu"
                print(f"[setup] Loading NeuroLKH model from {checkpoint} on {neurolkh_device}…")
                neurolkh_model = load_neurolkh_model(checkpoint, device=neurolkh_device)
                neurolkh_score_fn = lambda coords, ei, ef, iei: predict_edge_scores(  # noqa: E731
                    neurolkh_model, coords, ei, ef, iei, device=neurolkh_device,
                )

    # ---- rl4co env ---------------------------------------------------------
    env = TSPEnv(generator_params={"num_loc": args.num_nodes})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fi_lengths: list[float] = []
    lkh_lengths: list[float] = []
    fi_alpha_means: list[float] = []
    lkh_alpha_means: list[float] = []
    fi_neuro_means: list[float] = []
    lkh_neuro_means: list[float] = []
    # Stage-3: per-instance mean NeuroLKH on the *tour-augmented* 20-graph,
    # restricted to tour edges that fall in some node's top-K on the
    # augmented graph (i.e. what the col-3 panel actually draws).
    fi_aug_neuro_means: list[float] = []
    lkh_aug_neuro_means: list[float] = []
    # Stage-3: coverage lift = (# tour edges in top-K on augmented) -
    # (# tour edges in top-K on original 20-NN). Larger = more tour
    # edges now have a "model-confident" label.
    fi_coverage_lift: list[int] = []
    lkh_coverage_lift: list[int] = []
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
        fi_alphas = _tour_edge_values(fi_perm, alpha)
        fi_alpha_means.append(float(fi_alphas.mean()))
        if lkh_perm is not None:
            lkh_alphas = _tour_edge_values(lkh_perm, alpha)
            lkh_alpha_means.append(float(lkh_alphas.mean()))
        else:
            lkh_alpha_means.append(float("nan"))

        # ---- NeuroLKH scoring -------------------------------------------------
        neurolkh_score = None
        fi_aug_score: np.ndarray | None = None
        lkh_aug_score: np.ndarray | None = None
        if neurolkh_enabled and neurolkh_score_fn is not None:
            ei, ef, iei, _ = generate_20nn_features(
                coords,
                instance_name=f"instance_{k:02d}",
                lkh_binary=Path(args.neurolkh_binary) if args.neurolkh_binary else None,
            )
            neurolkh_score = neurolkh_score_fn(coords, ei, ef, iei)
            fi_neuro = _tour_edge_values(fi_perm, neurolkh_score)
            fi_neuro_means.append(float(np.nanmean(fi_neuro)))
            if lkh_perm is not None:
                lkh_neuro = _tour_edge_values(lkh_perm, neurolkh_score)
                lkh_neuro_means.append(float(np.nanmean(lkh_neuro)))
            else:
                lkh_neuro_means.append(float("nan"))

            # ---- Stage-3: tour-augmented graphs --------------------------------
            if not args.no_augment:
                if n == 20:
                    print(
                        f"[INFO] n={n}: tour-augmented graph equals the "
                        f"20-NN graph (complete graph); augmentation is a no-op."
                    )
                fi_aug_score = predict_edge_scores_augmented(
                    neurolkh_model, coords, fi_perm, ei, ef, iei,
                    device=neurolkh_device,
                )
                if lkh_perm is not None:
                    lkh_aug_score = predict_edge_scores_augmented(
                        neurolkh_model, coords, lkh_perm, ei, ef, iei,
                        device=neurolkh_device,
                    )

                # Compute coverage lift using the same topk as the figure.
                from utils.neurolkh_runner import compute_topk_mask

                nn_topk = compute_topk_mask(neurolkh_score, k=args.neurolkh_topk)
                fi_nn_in = int(
                    nn_topk[fi_perm, np.roll(fi_perm, -1)].sum()
                )
                fi_aug_v = _tour_edge_values(fi_perm, fi_aug_score)
                fi_aug_topk = compute_topk_mask(fi_aug_score, k=args.neurolkh_topk)
                fi_aug_in = int(
                    fi_aug_topk[fi_perm, np.roll(fi_perm, -1)].sum()
                )
                fi_aug_in_topk_vals = fi_aug_v[
                    fi_aug_topk[fi_perm, np.roll(fi_perm, -1)]
                    & np.isfinite(fi_aug_v)
                ]
                fi_aug_neuro_means.append(
                    float(np.nanmean(fi_aug_in_topk_vals))
                    if fi_aug_in_topk_vals.size
                    else float("nan")
                )
                fi_coverage_lift.append(fi_aug_in - fi_nn_in)

                if lkh_perm is not None:
                    lkh_nn_in = int(
                        nn_topk[lkh_perm, np.roll(lkh_perm, -1)].sum()
                    )
                    lkh_aug_v = _tour_edge_values(lkh_perm, lkh_aug_score)
                    lkh_aug_topk = compute_topk_mask(
                        lkh_aug_score, k=args.neurolkh_topk
                    )
                    lkh_aug_in = int(
                        lkh_aug_topk[lkh_perm, np.roll(lkh_perm, -1)].sum()
                    )
                    lkh_aug_in_topk_vals = lkh_aug_v[
                        lkh_aug_topk[lkh_perm, np.roll(lkh_perm, -1)]
                        & np.isfinite(lkh_aug_v)
                    ]
                    lkh_aug_neuro_means.append(
                        float(np.nanmean(lkh_aug_in_topk_vals))
                        if lkh_aug_in_topk_vals.size
                        else float("nan")
                    )
                    lkh_coverage_lift.append(lkh_aug_in - lkh_nn_in)
                else:
                    lkh_aug_neuro_means.append(float("nan"))
                    lkh_coverage_lift.append(0)
            else:
                fi_aug_neuro_means.append(float("nan"))
                lkh_aug_neuro_means.append(float("nan"))
                fi_coverage_lift.append(0)
                lkh_coverage_lift.append(0)
        else:
            fi_neuro_means.append(float("nan"))
            lkh_neuro_means.append(float("nan"))
            fi_aug_neuro_means.append(float("nan"))
            lkh_aug_neuro_means.append(float("nan"))
            fi_coverage_lift.append(0)
            lkh_coverage_lift.append(0)

        augmented_scores: list[np.ndarray | None] | None
        if neurolkh_enabled and not args.no_augment and fi_aug_score is not None:
            augmented_scores = [fi_aug_score, lkh_aug_score]
        else:
            augmented_scores = None

        fig = _plot_instance(
            coords, fi_perm, fi_len, lkh_perm,
            lkh_len if lkh_len != float("inf") else float("nan"),
            alpha,
            neurolkh_score,
            topk=args.neurolkh_topk,
            title_prefix=f"Instance {k:02d}  |  TSP-{n}",
            augmented_scores=augmented_scores,
        )
        out_path = out_dir / f"instance_{k:02d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        ratio_str = (
            f"FI/LKH = {fi_len / lkh_len:.3f}"
            if lkh_perm is not None and lkh_len > 0
            else "FI/LKH = n/a"
        )
        neuro_fi_str = (
            f"{fi_neuro_means[-1]:.4f}" if neurolkh_enabled else "skip"
        )
        neuro_lkh_str = (
            f"{lkh_neuro_means[-1]:.4f}"
            if neurolkh_enabled and lkh_perm is not None
            else ("skip" if not neurolkh_enabled else "nan")
        )
        if augmented_scores is not None and lkh_perm is not None:
            aug_fi_str = (
                f"{fi_aug_neuro_means[-1]:.4f}"
                if np.isfinite(fi_aug_neuro_means[-1])
                else "nan"
            )
            aug_lkh_str = (
                f"{lkh_aug_neuro_means[-1]:.4f}"
                if np.isfinite(lkh_aug_neuro_means[-1])
                else "nan"
            )
            cov_fi = fi_coverage_lift[-1]
            cov_lkh = lkh_coverage_lift[-1]
            aug_suffix = (
                f"  NeuroLKHaug_FI={aug_fi_str}  NeuroLKHaug_LKH={aug_lkh_str}  "
                f"Δcov_FI={cov_fi:+d}  Δcov_LKH={cov_lkh:+d}"
            )
        else:
            aug_suffix = ""
        print(
            f"  [{k:02d}]  FI = {fi_len:.4f}   "
            f"LKH = {lkh_len if lkh_perm else 'fail':>10}   "
            f"{ratio_str}   "
            f"α_FI={fi_alpha_means[-1]:.4f}  α_LKH={lkh_alpha_means[-1] if lkh_perm else float('nan'):.4f}   "
            f"NeuroLKH_FI={neuro_fi_str}  NeuroLKH_LKH={neuro_lkh_str}{aug_suffix}"
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
        f"  mean α on FI tour:    mean = {fi_a.mean():.4f}   "
        f"std = {fi_a.std():.4f}"
    )
    if valid.any():
        lkh_a_valid = lkh_a[valid]
        print(
            f"  mean α on LKH tour:   mean = {lkh_a_valid.mean():.4f}   "
            f"std = {lkh_a_valid.std():.4f}"
        )
        if lkh_a_valid.mean() < fi_a[valid].mean():
            print(
                "  -> LKH picks lower-α edges on average (metric "
                "correlates with quality)"
            )
        else:
            print(
                "  -> NOTE: LKH mean α is NOT lower than FI mean α "
                "on this batch; the metric may not discriminate on the "
                "given instances."
            )

    if neurolkh_enabled:
        fi_n = np.asarray(fi_neuro_means)
        lkh_n = np.asarray(lkh_neuro_means)
        valid_n = ~np.isnan(fi_n) & ~np.isnan(lkh_n)
        print(
            f"  mean NeuroLKH on FI tour:  mean = {fi_n[valid_n].mean():.4f}   "
            f"std = {fi_n[valid_n].std():.4f}"
        )
        if valid_n.any():
            lkh_n_valid = lkh_n[valid_n]
            print(
                f"  mean NeuroLKH on LKH tour: mean = {lkh_n_valid.mean():.4f}   "
                f"std = {lkh_n_valid.std():.4f}"
            )
            print(
                "  NOTE: NeuroLKH scores are softmax over each node's 20-NN "
                "(see notes/neurolkh-configs.md); higher = better, opposite "
                "direction from α. Mean FI vs LKH does NOT necessarily "
                "discriminate — the visualization highlights the per-edge "
                "model picks, not the global mean."
            )

    # Stage-3 summary
    if neurolkh_enabled and not args.no_augment:
        fi_aug = np.asarray(fi_aug_neuro_means, dtype=np.float64)
        lkh_aug = np.asarray(lkh_aug_neuro_means, dtype=np.float64)
        valid_aug = ~np.isnan(fi_aug) & ~np.isnan(lkh_aug)
        fi_cov = np.asarray(fi_coverage_lift, dtype=np.int64)
        lkh_cov = np.asarray(lkh_coverage_lift, dtype=np.int64)
        if valid_aug.any():
            print(
                f"  mean NeuroLKH (aug) on FI tour:  mean = {fi_aug[valid_aug].mean():.4f}   "
                f"std = {fi_aug[valid_aug].std():.4f}"
            )
            print(
                f"  mean NeuroLKH (aug) on LKH tour: mean = {lkh_aug[valid_aug].mean():.4f}   "
                f"std = {lkh_aug[valid_aug].std():.4f}"
            )
            print(
                f"  coverage lift (FI):  mean = {fi_cov[valid_aug].mean():+.1f}   "
                f"std = {fi_cov[valid_aug].std():.1f}   "
                f"min/max = {fi_cov[valid_aug].min()}/{fi_cov[valid_aug].max()}"
            )
            print(
                f"  coverage lift (LKH): mean = {lkh_cov[valid_aug].mean():+.1f}   "
                f"std = {lkh_cov[valid_aug].std():.1f}   "
                f"min/max = {lkh_cov[valid_aug].min()}/{lkh_cov[valid_aug].max()}"
            )
            print(
                "  NOTE: augmented-graph scores are softmax over the "
                "tour-augmented 20 (the tour edges are forced into the "
                "candidate set, displacing the farthest 20-NN slots). "
                "The interpretation is 'given these 20 candidates, which "
                "does the model prefer', not 'this edge is in OPT'."
            )

    print(f"\nFigures written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())