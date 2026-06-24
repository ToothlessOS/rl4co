"""EdgeQualityModule — LightningModule wrapper around ``SGNEdgeClassifier``.

Logs per-step and per-epoch metrics. At validation/test time we also
bucket metrics by strategy id (nn / fi / kopt / random / opt) so we can
see how the model performs across the 4 strategies from the spec.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from torch.optim import AdamW

from ..eval.edge_metrics import aggregate_metrics, edge_quality_metrics
from ..model import SGNEdgeClassifier


_STRATEGY_NAMES = ["nn", "fi", "kopt", "random", "opt"]


class EdgeQualityModule(pl.LightningModule):
    """Lightning module for the SGN edge classifier.

    Args:
        model_cfg: dict with SGN hyperparameters (``hidden_dim``,
            ``n_gcn_layers``, ``n_mlp_layers``, ``n_edges_per_node``,
            ``pos_weight``).
        optim_cfg: dict with optimizer hyperparameters (``lr``,
            ``weight_decay``).
    """

    def __init__(self, model_cfg: dict[str, Any], optim_cfg: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.model = SGNEdgeClassifier(
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            n_gcn_layers=int(model_cfg.get("n_gcn_layers", 30)),
            n_mlp_layers=int(model_cfg.get("n_mlp_layers", 3)),
            n_edges_per_node=int(model_cfg.get("n_edges_per_node", 2)),
            pos_weight=float(model_cfg.get("pos_weight", 9.0)),
        )
        # Per-strategy accumulators for epoch-end aggregation.
        self._val_per_inst: list[dict] = []
        self._test_per_inst: list[dict] = []

    # ---- forward / steps ------------------------------------------------

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(batch)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        y_pred_edges, loss = self(batch)
        # Per-step log (Lightning auto-averages per epoch).
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "test"),
                 on_step=(stage == "train"), on_epoch=True, batch_size=batch["coords"].size(0))
        # Also compute accuracy + per-strategy metrics on val/test.
        if stage in ("val", "test"):
            self._accumulate_per_instance_metrics(y_pred_edges, batch, stage)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    # ---- per-strategy metrics -------------------------------------------

    def _accumulate_per_instance_metrics(
        self, y_pred_edges: torch.Tensor, batch: dict, stage: str
    ) -> None:
        pred_probs = torch.exp(y_pred_edges[..., 1])              # (B, max_n*2)
        B, max_n = batch["pad_mask"].shape
        n_edges_per_node = self.model.n_edges_per_node
        pred_probs_per_node = pred_probs.view(B, max_n, n_edges_per_node).detach().cpu().numpy()
        y_flat = batch["y_edges"].view(B, max_n, n_edges_per_node).cpu().numpy()
        ei_flat = batch["edge_index"].view(B, max_n, n_edges_per_node).cpu().numpy()
        pad = batch["pad_mask"].cpu().numpy()
        strat = batch["strategy_id"].cpu().numpy()

        opt_tours = batch.get("opt_tour", None)
        for b in range(B):
            opt = opt_tours[b].cpu().numpy() if opt_tours is not None else None
            m = edge_quality_metrics(
                pred_probs=pred_probs_per_node[b],
                y_edges=y_flat[b],
                edge_index=ei_flat[b],
                pad_mask=pad[b],
                opt_tour=opt,
            )
            m["_strategy"] = _STRATEGY_NAMES[int(strat[b])]
            if stage == "val":
                self._val_per_inst.append(m)
            else:
                self._test_per_inst.append(m)

    def _flush_metrics(self, stage: str) -> None:
        per_inst = (
            self._val_per_inst if stage == "val" else self._test_per_inst
        )
        if not per_inst:
            return
        agg = aggregate_metrics(per_inst)
        for k, v in agg.items():
            if k.startswith("_"):
                continue
            self.log(f"{stage}/{k}", v, prog_bar=False, on_epoch=True, batch_size=1)
        # Per-strategy bucket means.
        for sid, sname in enumerate(_STRATEGY_NAMES):
            bucket = [m for m in per_inst if m.get("_strategy") == sname]
            if bucket:
                agg_s = aggregate_metrics(bucket)
                for k, v in agg_s.items():
                    if k.startswith("_"):
                        continue
                    self.log(f"{stage}/strategy_{sname}/{k}", v,
                             prog_bar=False, on_epoch=True, batch_size=1)
        per_inst.clear()

    def on_validation_epoch_end(self) -> None:
        self._flush_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._flush_metrics("test")

    # ---- optimizer ------------------------------------------------------

    def configure_optimizers(self) -> torch.optim.Optimizer:
        lr = float(self.hparams.optim_cfg.get("lr", 1e-4))
        wd = float(self.hparams.optim_cfg.get("weight_decay", 0.0))
        return AdamW(self.parameters(), lr=lr, weight_decay=wd)