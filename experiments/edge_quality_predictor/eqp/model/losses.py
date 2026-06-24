"""Loss functions for the edge quality classifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_nll_loss(
    y_pred_edges: torch.Tensor,
    y_edges: torch.Tensor,
    pad_mask: torch.Tensor,
    n_edges_per_node: int,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """NLL on per-edge 2-class log-probs with padding/ignore masking.

    Args:
        y_pred_edges: ``(B, max_n*k, 2)`` log-probs (channel 0 = not-OPT,
            channel 1 = OPT). The class-weight tensor is built from
            ``pos_weight``; if ``pos_weight`` is ``None``, uniform weights.
        y_edges: ``(B, max_n*k)`` long in ``{0, 1, -100}``. ``-100`` is
            the NLL ignore index; padded slots must carry ``-100``.
        pad_mask: ``(B, max_n)`` bool. Slots where any of ``i``'s edges are
            padded get their loss zeroed (defensive — the dataset already
            sets ``y_edges=-100`` for padded slots).
        n_edges_per_node: ``k`` (2 for the tour-edge experiment).
        pos_weight: scalar applied to class 1 in the NLL weight tensor.
            ``None`` ⇒ uniform weights.

    Returns:
        Scalar mean over non-ignored slots.
    """
    weight = None
    if pos_weight is not None:
        weight = torch.tensor(
            [1.0, float(pos_weight)], dtype=y_pred_edges.dtype, device=y_pred_edges.device
        )
    # NLLLoss expects (B, C, ...) for input and (B, ...) for target.
    y_pred_perm = y_pred_edges.permute(0, 2, 1)                       # (B, 2, max_n*k)
    loss_flat = F.nll_loss(y_pred_perm, y_edges, weight=weight, reduction="none", ignore_index=-100)
    B, Nk = y_edges.shape
    max_n = B and (Nk // n_edges_per_node) or 0
    loss = loss_flat.view(B, max_n, n_edges_per_node)
    # Zero out padded nodes (defensive — y_edges=-100 should already cover this).
    if pad_mask is not None:
        node_mask = pad_mask.view(B, max_n, 1).expand_as(loss)
        loss = loss * node_mask.float()
        denom = node_mask.float().sum().clamp_min(1.0) * n_edges_per_node
    else:
        denom = (y_edges != -100).float().sum().clamp_min(1.0)
    return loss.sum() / denom