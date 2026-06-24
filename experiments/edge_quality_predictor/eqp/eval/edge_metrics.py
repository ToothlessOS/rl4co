"""Per-instance edge-quality metrics for the SGN edge classifier.

The classifier scores each of the 2 edges per node (in the input tour) for
its probability of being in the OPT tour. Metrics computed over one
instance:

- ``edge_accuracy``: ``argmax(P(in-OPT)) == y`` on non-padded slots.
- ``balanced_accuracy``: mean of per-class recall (handles class imbalance).
- ``precision@1``: fraction of the model's top-1 per node that is in OPT.
- ``precision@2``: fraction of the model's top-2 per node that are in OPT.
  (With k=2 slots per node, ``precision@2`` is over all slots.)
- ``top1_hit_rate``: fraction of OPT edges for which at least one endpoint's
  top-1 prediction lands on that edge.
- ``mean_p_in_opt``: mean predicted probability on slots actually labelled 1.
- ``pos_pred_rate``: fraction of slots predicted positive (P > 0.5).

All metrics are over the valid (non-padded) nodes only. PR-AUC is left as
a future extension (with only 2 slots/node it degenerates to a single
threshold choice).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def edge_quality_metrics(
    pred_probs: np.ndarray,
    y_edges: np.ndarray,
    edge_index: np.ndarray,
    pad_mask: np.ndarray,
    opt_tour: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute per-instance edge-quality metrics.

    Args:
        pred_probs: ``(max_n, 2)`` per-slot P(in-OPT) — exp(channel 1) of
            the model's log-probs, reshaped to per-node.
        y_edges: ``(max_n, 2)`` int64 in ``{0, 1, -100}``.
        edge_index: ``(max_n, 2)`` int64 0-indexed neighbour ids. Used to
            map top-1 slots back to their edge for OPT-hit computation.
        pad_mask: ``(max_n,)`` bool.
        opt_tour: ``(n,)`` int64 0-indexed OPT permutation, or ``None``
            (e.g. TSPlib). When ``None``, OPT-dependent metrics are
            returned as NaN.

    Returns:
        Dict of metric name → float. NaN entries indicate metrics that
        require OPT information.
    """
    valid_idx = np.where(pad_mask)[0]
    n_valid = valid_idx.size
    y_valid = y_edges[valid_idx]                                 # (n_valid, 2)
    p_valid = pred_probs[valid_idx]                              # (n_valid, 2)
    ei_valid = edge_index[valid_idx]                             # (n_valid, 2)

    out: dict[str, float] = {}

    # ---- Edge accuracy ----
    if n_valid > 0:
        preds = (p_valid >= 0.5).astype(np.int64)
        out["edge_accuracy"] = float((preds == y_valid).mean())
    else:
        out["edge_accuracy"] = float("nan")

    # ---- Balanced accuracy ----
    if n_valid > 0 and (y_valid == 0).any() and (y_valid == 1).any():
        neg_mask = y_valid == 0
        pos_mask = y_valid == 1
        neg_recall = float((preds[neg_mask] == 0).mean()) if neg_mask.any() else float("nan")
        pos_recall = float((preds[pos_mask] == 1).mean()) if pos_mask.any() else float("nan")
        out["balanced_accuracy"] = float(np.nanmean([neg_recall, pos_recall]))
    else:
        out["balanced_accuracy"] = float("nan")

    # ---- Precision@k ----
    if n_valid > 0:
        top1 = p_valid.argmax(axis=1)                           # (n_valid,)
        top1_y = np.take_along_axis(y_valid, top1[:, None], axis=1).squeeze(1)
        out["precision_at_1"] = float(top1_y.mean())
        out["precision_at_2"] = float(y_valid.mean())
    else:
        out["precision_at_1"] = float("nan")
        out["precision_at_2"] = float("nan")

    # ---- OPT-dependent metrics ----
    if opt_tour is not None and n_valid > 0:
        # Build OPT-edge set restricted to valid nodes.
        opt_edges: set[tuple[int, int]] = set()
        n = opt_tour.shape[0]
        for k in range(n):
            a, b = int(opt_tour[k]), int(opt_tour[(k + 1) % n])
            if valid_idx.size and (a in valid_idx) and (b in valid_idx):
                opt_edges.add((min(int(a), int(b)), max(int(a), int(b))))

        # For each valid node, the top-1 slot picks the edge (i, ei[i, top1]).
        # An OPT edge (a, b) is "hit" if either a's top-1 is (a, b) or b's
        # top-1 is (a, b). We check this by iterating OPT edges and looking
        # up the corresponding local top-1 picks.
        # Build a (i_local, j_local) lookup: for each local node i_local
        # and each of its 2 slots, the (i, j) edge it represents.
        top1_per_local: list[tuple[int, int]] = []
        for i_local in range(n_valid):
            slot = int(top1[i_local])
            i = int(valid_idx[i_local])
            j = int(ei_valid[i_local, slot])
            top1_per_local.append((i, j))
        # Reverse lookup: for any (i, j), what's i_local?
        node_to_local = {int(valid_idx[k]): k for k in range(n_valid)}

        n_hits = 0
        for (a, b) in opt_edges:
            a_hit = False
            b_hit = False
            if a in node_to_local:
                i_local_a = node_to_local[a]
                slot_a = int(top1[i_local_a])
                j_a = int(ei_valid[i_local_a, slot_a])
                if (min(a, j_a), max(a, j_a)) == (a, b):
                    a_hit = True
            if b in node_to_local:
                i_local_b = node_to_local[b]
                slot_b = int(top1[i_local_b])
                j_b = int(ei_valid[i_local_b, slot_b])
                if (min(b, j_b), max(b, j_b)) == (a, b):
                    b_hit = True
            if a_hit or b_hit:
                n_hits += 1
        out["top1_hit_rate"] = float(n_hits / max(len(opt_edges), 1))

        pos_slots = y_valid == 1
        if pos_slots.any():
            out["mean_p_in_opt"] = float(p_valid[pos_slots].mean())
        else:
            out["mean_p_in_opt"] = float("nan")
    else:
        out["top1_hit_rate"] = float("nan")
        out["mean_p_in_opt"] = float("nan")

    # ---- Calibration diagnostics ----
    if n_valid > 0:
        out["pos_pred_rate"] = float((p_valid >= 0.5).mean())
        out["neg_pred_rate"] = float((p_valid < 0.5).mean())
        out["mean_p"] = float(p_valid.mean())
    else:
        out["pos_pred_rate"] = float("nan")
        out["neg_pred_rate"] = float("nan")
        out["mean_p"] = float("nan")

    return out


def aggregate_metrics(per_instance: list[dict[str, float]]) -> dict[str, float]:
    """Mean (ignoring NaN) across a list of per-instance metric dicts.

    Skips keys starting with ``_`` (e.g. ``"_strategy"``) and keys whose
    values are not numeric.
    """
    if not per_instance:
        return {}
    keys = per_instance[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        if k.startswith("_"):
            continue
        vals = []
        for m in per_instance:
            v = m.get(k)
            if v is None or isinstance(v, str):
                continue
            try:
                if not np.isnan(float(v)):
                    vals.append(float(v))
            except (TypeError, ValueError):
                continue
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out