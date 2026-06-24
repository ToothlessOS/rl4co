"""LightningDataModule for the edge quality predictor.

Loads train/test .txt files (and optionally TSPlib), wraps them in
``EdgeQualityDataset`` instances, and returns DataLoaders.

Per-spec test sets are split into separate sub-datasets so per-strategy
metrics can be reported per-file at test time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ..data import (
    EdgeQualityDataset,
    TSPInstance,
    collate_edge_batch,
    load_test_file,
    load_train_file,
    load_tsplib_file,
)


class EdgeQualityDataModule(pl.LightningDataModule):
    """DataModule with separate train, val, and (multi) test sets.

    Args:
        data_cfg: dict with the following keys:
            train_path, val_path, test_paths (list), train_limit,
            val_limit, test_limit, pad_to_n, strategy_weights,
            kopt_n_moves, kopt_p_3opt, num_workers.
    """

    def __init__(self, data_cfg: dict):
        super().__init__()
        self.data_cfg = dict(data_cfg)
        self.train_ds: EdgeQualityDataset | None = None
        self.val_ds: EdgeQualityDataset | None = None
        self.test_ds_list: list[tuple[str, EdgeQualityDataset]] = []

    def _load(self, path: str, limit: int | None, kind: str) -> list[TSPInstance]:
        p = Path(path)
        if kind == "tsplib" or "TSPlib" in p.name:
            return load_tsplib_file(p, max_instances=limit)
        return load_test_file(p, max_instances=limit)

    def setup(self, stage: str | None = None) -> None:
        cfg = self.data_cfg
        pad_to_n = cfg.get("pad_to_n")
        strategy_weights = cfg.get("strategy_weights")
        kopt_n_moves = int(cfg.get("kopt_n_moves", 5))
        kopt_p_3opt = float(cfg.get("kopt_p_3opt", 0.3))

        if stage in (None, "fit"):
            train_path = cfg["train_path"]
            val_path = cfg.get("val_path")
            train_limit = cfg.get("train_limit")
            val_limit = cfg.get("val_limit")

            train_inst = load_train_file(train_path, max_instances=train_limit)
            self.train_ds = EdgeQualityDataset(
                train_inst,
                pad_to_n=pad_to_n,
                strategy_weights=strategy_weights,
                kopt_n_moves=kopt_n_moves,
                kopt_p_3opt=kopt_p_3opt,
                seed=42,
            )
            if val_path:
                val_inst = load_test_file(val_path, max_instances=val_limit)
                self.val_ds = EdgeQualityDataset(
                    val_inst,
                    pad_to_n=pad_to_n,
                    strategy_weights=strategy_weights,
                    kopt_n_moves=kopt_n_moves,
                    kopt_p_3opt=kopt_p_3opt,
                    seed=43,
                )

        if stage in (None, "test"):
            test_paths: Sequence[str] = cfg.get("test_paths", [])
            test_limit = cfg.get("test_limit")
            self.test_ds_list = []
            for tp in test_paths:
                p = Path(tp)
                kind = "tsplib" if "TSPlib" in p.name else "txt"
                inst = self._load(tp, test_limit, kind)
                ds = EdgeQualityDataset(
                    inst,
                    pad_to_n=pad_to_n,
                    strategy_weights=strategy_weights,
                    kopt_n_moves=kopt_n_moves,
                    kopt_p_3opt=kopt_p_3opt,
                    seed=44,
                )
                self.test_ds_list.append((tp, ds))

    # ---- dataloaders ----------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=int(self.data_cfg.get("batch_size", 32)),
            shuffle=True,
            num_workers=int(self.data_cfg.get("num_workers", 0)),
            collate_fn=collate_edge_batch,
            drop_last=True,
            persistent_workers=bool(self.data_cfg.get("num_workers", 0)),
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=int(self.data_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(self.data_cfg.get("num_workers", 0)),
            collate_fn=collate_edge_batch,
            persistent_workers=bool(self.data_cfg.get("num_workers", 0)),
        )

    def test_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=int(self.data_cfg.get("batch_size", 32)),
                shuffle=False,
                num_workers=0,
                collate_fn=collate_edge_batch,
            )
            for _, ds in self.test_ds_list
        ]