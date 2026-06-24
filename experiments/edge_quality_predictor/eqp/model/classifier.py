"""SGNEdgeClassifier — wraps SparseGCNModelV with a clean training interface.

The classifier reshapes the dataset's ``(B, max_n, 2)``-shaped edge tensors
into the flat ``(B, max_n*2)`` shape that ``SparseGCNModelV.forward``
expects, runs the model, and computes the masked NLL loss with class
weighting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .losses import masked_nll_loss
from .sgn import SparseGCNModelV


def _flat_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """``(B, n, k)`` int64 → ``(B, n*k)`` int64."""
    B, n, k = edge_index.shape
    return edge_index.reshape(B, n * k)


def _flat_edge_feat(edge_feat: torch.Tensor) -> torch.Tensor:
    """``(B, n, k)`` float32 → ``(B, n*k, 1)`` float32."""
    B, n, k = edge_feat.shape
    return edge_feat.reshape(B, n * k, 1)


def _flat_inverse(inv: torch.Tensor) -> torch.Tensor:
    """``(B, n, k)`` int64 → ``(B, n*k)`` int64.

    Padded slots have ``inv = -1`` (sentinel "no reverse edge"); the
    SparseGCNLayer's ``W_placeholder`` row handles these.
    """
    B, n, k = inv.shape
    return inv.reshape(B, n * k)


def _flat_y(y_edges: torch.Tensor) -> torch.Tensor:
    """``(B, n, k)`` int64 → ``(B, n*k)`` int64."""
    B, n, k = y_edges.shape
    return y_edges.reshape(B, n * k)


class SGNEdgeClassifier(nn.Module):
    """SparseGCNModelV + masked NLL loss with class weighting.

    Args:
        hidden_dim: SGN hidden width.
        n_gcn_layers: number of GCN layers.
        n_mlp_layers: number of MLP layers in the edge head.
        n_edges_per_node: 2 for tour edges (this experiment).
        pos_weight: scalar applied to class 1 in the NLL weight tensor.
        aggregation: how SGN aggregates incoming edge messages ("mean").
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_gcn_layers: int = 30,
        n_mlp_layers: int = 3,
        n_edges_per_node: int = 2,
        pos_weight: float = 9.0,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.n_edges_per_node = int(n_edges_per_node)
        self.backbone = SparseGCNModelV(
            hidden_dim=hidden_dim,
            n_gcn_layers=n_gcn_layers,
            n_mlp_layers=n_mlp_layers,
            n_edges_per_node=n_edges_per_node,
            aggregation=aggregation,
        )

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass on a collated batch.

        Args:
            batch: dict from ``collate_edge_batch`` with keys ``coords``,
                ``edge_index``, ``edge_feat``, ``inverse_edge_index``,
                ``y_edges``, ``pad_mask``, ``strategy_id``.

        Returns:
            y_pred_edges: ``(B, n*2, 2)`` log-probs.
            loss: scalar, masked NLL mean over non-padded slots.
        """
        coords = batch["coords"]                       # (B, max_n, 2)
        edge_index = _flat_edge_index(batch["edge_index"])            # (B, max_n*2)
        edge_feat = _flat_edge_feat(batch["edge_feat"])              # (B, max_n*2, 1)
        inverse = _flat_inverse(batch["inverse_edge_index"])         # (B, max_n*2)
        y_edges = _flat_y(batch["y_edges"])                          # (B, max_n*2)
        pad_mask = batch["pad_mask"]                                 # (B, max_n)

        edge_cw = torch.tensor(
            [1.0, self.pos_weight], dtype=coords.dtype, device=coords.device
        )

        y_pred_edges, _loss, _y_pred_nodes = self.backbone(
            x_nodes=coords,
            x_edges=edge_feat,
            edge_index=edge_index,
            inverse_edge_index=inverse,
            y_edges=y_edges,
            edge_cw=edge_cw,
            n_edges=self.n_edges_per_node,
        )

        loss = masked_nll_loss(y_pred_edges, y_edges, pad_mask, self.n_edges_per_node)
        return y_pred_edges, loss

    @torch.no_grad()
    def predict_edge_scores(self, batch: dict) -> np.ndarray:
        """Convert per-slot log-probs into a symmetric ``(B, max_n, max_n)`` score matrix.

        Score = exp(channel 1). Padded slots are filled with 0 (which will be
        turned into NaN outside the per-instance valid range by the caller).

        The returned matrix has zeros on the diagonal and 0 in cells outside
        the union of the per-node 2-sets; cells inside the union hold the
        per-slot probability.
        """
        device = self.device
        coords = batch["coords"].to(device)
        edge_index = _flat_edge_index(batch["edge_index"]).to(device)
        edge_feat = _flat_edge_feat(batch["edge_feat"]).to(device)
        inverse = _flat_inverse(batch["inverse_edge_index"]).to(device)
        pad_mask = batch["pad_mask"].to(device)                       # (B, max_n)

        y_pred_edges, _, _ = self.backbone(
            x_nodes=coords,
            x_edges=edge_feat,
            edge_index=edge_index,
            inverse_edge_index=inverse,
            y_edges=None,
            edge_cw=None,
            n_edges=self.n_edges_per_node,
        )
        # channel 1 = P(in-OPT); shape (B, max_n*2)
        p_in_opt = torch.exp(y_pred_edges[..., 1])
        B, max_n = pad_mask.shape
        per_node = p_in_opt.view(B, max_n, self.n_edges_per_node)
        # Fill score matrix from per-node slots.
        score = torch.zeros(B, max_n, max_n, device=device, dtype=p_in_opt.dtype)
        ei_flat = edge_index.view(B, max_n, self.n_edges_per_node)
        for b in range(B):
            for i in range(max_n):
                if not bool(pad_mask[b, i]):
                    continue
                for s in range(self.n_edges_per_node):
                    j = int(ei_flat[b, i, s].item())
                    if j < 0 or j >= max_n:
                        continue
                    if not bool(pad_mask[b, j]):
                        continue
                    val = float(per_node[b, i, s].item())
                    score[b, i, j] = max(score[b, i, j].item(), val)
        # Symmetrize and zero the diagonal.
        score = torch.maximum(score, score.transpose(1, 2))
        for b in range(B):
            score[b].fill_diagonal_(0.0)
        return score.cpu().numpy()