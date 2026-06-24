"""Forked SparseGCNModel with configurable ``n_edges_per_node``.

The original ``experiments/decompose-on-edges/NeuroLKH/net/sgcn_model.py``
hard-codes ``n_edges_per_node = 20`` because the pretrained checkpoint is
shape-locked to the 20-NN candidate graph. We need ``n_edges_per_node = 2``
for the tour-edge graph (each node has exactly 2 neighbours in a closed
cycle), so we fork the model here. We do **not** load the pretrained
weights — the new architecture cannot reuse them.

This fork is a near-verbatim copy of the upstream ``SparseGCNModel`` and
``SparseGCNLayer`` (also imported from upstream ``sgcn_layers``), with two
changes:

1. ``n_edges_per_node`` is a constructor parameter instead of a module-
   level constant.
2. The output MLP head is sized ``hidden_dim -> 2`` (binary classification
   over the per-slot 2 classes), hard-coded for TSP. Upstream also uses
   ``hidden_dim -> 2`` for PDP/CVRPTW but ``hidden_dim -> 1`` for TSP —
   here we want log-probs of 2 classes, so we always use 2.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layers (forked from experiments/decompose-on-edges/NeuroLKH/net/sgcn_layers.py)
# ---------------------------------------------------------------------------


class BatchNormNode(nn.Module):
    """Per-node BatchNorm1d with ``track_running_stats=False``.

    Forked from upstream ``BatchNormNode``. We keep the same shape contract:
    ``(B, n, hidden)`` input/output.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2).contiguous()
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()
        return x_bn


class NodeFeatures(nn.Module):
    """Forked from upstream ``NodeFeatures``. Aggregates edge messages into nodes."""

    def __init__(self, hidden_dim: int, aggregation: str = "mean"):
        super().__init__()
        self.aggregation = aggregation
        self.node_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.to_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.edge_embedding = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor,
                n_edges: int) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.size()
        Ux = self.node_embedding(x)
        Vx = self.to_embedding(x)
        Ve = self.edge_embedding(e).view(batch_size, num_nodes, n_edges, hidden_dim)
        Ve = F.softmax(Ve, dim=2).view(batch_size, num_nodes * n_edges, hidden_dim)

        # Gather node embeddings along the sparse edge_index.
        Vx = Vx[torch.arange(batch_size).view(-1, 1), edge_index]
        to = (Ve * Vx).view(batch_size, num_nodes, n_edges, hidden_dim).sum(2)
        return Ux + to


class EdgeFeatures(nn.Module):
    """Forked from upstream ``EdgeFeatures``. Computes the next-step edge embedding."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_from = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_to = nn.Linear(hidden_dim, hidden_dim, True)
        self.inverse_U = nn.Linear(hidden_dim, hidden_dim, True)
        self.W_placeholder = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_placeholder.data.uniform_(-1, 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor,
                inverse_edge_index: torch.Tensor, n_edges: int) -> torch.Tensor:
        batch_size, graph_size, hidden_dim = x.size()
        Ue = self.U(e)
        inverse_Ue = self.inverse_U(e)
        # Append a "no inverse" placeholder row for slots whose inverse is -1.
        inverse_Ue = torch.cat(
            (inverse_Ue, self.W_placeholder.view(1, 1, hidden_dim).repeat(batch_size, 1, 1)),
            dim=1,
        )
        inverse_node_embedding = inverse_Ue[
            torch.arange(batch_size).view(batch_size, 1), inverse_edge_index
        ]
        Vx_from = self.V_from(x)
        Vx_to = self.V_to(x)
        Vx = Vx_to[torch.arange(batch_size).view(-1, 1), edge_index]
        Vx = Vx.view(batch_size, -1, n_edges, hidden_dim) + Vx_from.view(batch_size, -1, 1, hidden_dim)
        Vx = Vx.view(batch_size, -1, hidden_dim)
        return Ue + Vx + inverse_node_embedding


class SparseGCNLayer(nn.Module):
    """Forked from upstream ``SparseGCNLayer``."""

    def __init__(self, hidden_dim: int, aggregation: str = "mean"):
        super().__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormNode(hidden_dim)

    def forward(self, x: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor,
                inverse_edge_index: torch.Tensor, n_edges: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_tmp = self.node_feat(x, e, edge_index, n_edges)
        x_tmp = self.bn_node(x_tmp)
        x = F.relu(x_tmp)
        x_new = x + x

        e_tmp = self.edge_feat(x_new, e, edge_index, inverse_edge_index, n_edges)
        e_tmp = self.bn_edge(e_tmp)
        e = F.relu(e_tmp)
        e_new = e + e
        return x_new, e_new


class MLP(nn.Module):
    """Forked from upstream ``MLP``. ``L`` total linear layers, last is the head."""

    def __init__(self, hidden_dim: int, output_dim: int, L: int = 2):
        super().__init__()
        self.L = L
        layers = []
        for _ in range(L - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SparseGCNModelV(nn.Module):
    """Sparse GCN with configurable per-node candidate count.

    Forward contract (matches upstream)::

        x_nodes:           (B, n, 2)        -- raw node features (coords)
        x_edges:           (B, n*n_edges, 1) -- edge features (e.g. distance)
        edge_index:        (B, n*n_edges)    -- 0-indexed neighbour ids
        inverse_edge_index: (B, n*n_edges)   -- flat positions of the reverse edge
        y_edges:           (B, n*n_edges) or None -- optional integer labels
        edge_cw:           (2,) class weights for NLLLoss; ignored when y_edges is None
        n_edges:           int (== self.n_edges_per_node)

    Returns ``(y_pred_edges, loss, y_pred_nodes)`` where ``y_pred_edges`` has
    shape ``(B, n*n_edges, 2)`` (log-probs of the two classes).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_gcn_layers: int = 30,
        n_mlp_layers: int = 3,
        n_edges_per_node: int = 2,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.node_dim = 2      # x, y for TSP
        self.edge_dim = 1
        self.hidden_dim = hidden_dim
        self.n_gcn_layers = n_gcn_layers
        self.n_mlp_layers = n_mlp_layers
        self.n_edges_per_node = n_edges_per_node
        self.aggregation = aggregation

        self.nodes_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)

        gcn_layers = []
        for _ in range(self.n_gcn_layers):
            gcn_layers.append(SparseGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Per-edge raw logit head. The 2-channel softmax over the k slots
        # is computed downstream in ``forward``; we collapse to 1 logit here
        # to keep the per-node softmax shape (B, n*k) → (B, n, k).
        self.mlp_edges = MLP(self.hidden_dim, 1, self.n_mlp_layers)
        self.mlp_nodes = MLP(self.hidden_dim, 1, self.n_mlp_layers)

    def forward(
        self,
        x_nodes: torch.Tensor,
        x_edges: torch.Tensor,
        edge_index: torch.Tensor,
        inverse_edge_index: torch.Tensor,
        y_edges: Optional[torch.Tensor] = None,
        edge_cw: Optional[torch.Tensor] = None,
        n_edges: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if n_edges is None:
            n_edges = self.n_edges_per_node
        batch_size, num_nodes, _ = x_nodes.size()
        x = self.nodes_embedding(x_nodes)
        e = self.edges_embedding(x_edges)

        loss_mask = edge_index.view(batch_size, num_nodes, n_edges).sum(-1) != 0

        for layer in self.gcn_layers:
            x, e = layer(x, e, edge_index, inverse_edge_index, n_edges)

        # Per-edge raw logit, then per-node softmax over the k slots so they
        # sum to 1 (matching upstream convention). ``y_pred_edges`` is the
        # 2-channel log-prob ``[log(1-p), log(p)]`` used as input to NLLLoss.
        y_pred_edges = self.mlp_edges(e).view(batch_size, num_nodes, n_edges)
        y_pred_edges = torch.exp(y_pred_edges)
        y_pred_edges = y_pred_edges / (y_pred_edges.sum(2).view(batch_size, num_nodes, 1) + 1e-5)
        y_pred_edges = y_pred_edges.view(batch_size, num_nodes * n_edges, 1)
        y_pred_edges = torch.cat([1 - y_pred_edges, y_pred_edges], dim=2)
        y_pred_edges = torch.log(y_pred_edges)

        loss: Optional[torch.Tensor] = None
        if y_edges is not None:
            y_pred_perm = y_pred_edges.permute(0, 2, 1)
            loss = F.nll_loss(y_pred_perm, y_edges, weight=edge_cw, reduction="none")
            loss = loss.view(batch_size, num_nodes, n_edges)[loss_mask]

        y_pred_nodes = 10 * torch.tanh(self.mlp_nodes(x))
        return y_pred_edges, loss, y_pred_nodes