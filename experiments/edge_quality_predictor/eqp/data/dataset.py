"""EdgeQualityDataset — produces padded model inputs for the SGN.

For each ``TSPInstance`` we:

1. Sample a strategy id (per-instance, on the fly).
2. Run the strategy to get an ``input_tour`` (permutation).
3. Build the 2-slot graph from ``input_tour`` (each node's two neighbours
   in the closed cycle).
4. Derive the OPT-edge label set from the instance's ``opt_tour``.
5. Compute per-slot labels: ``1`` if ``(i, edge_index[i,k])`` is an OPT
   edge, ``0`` otherwise; ``-100`` for padded slots.
6. Pad everything to ``pad_to_n`` (the dataset's max n, capped by config).

The output dict is consumed by ``EdgeQualityDataModule``'s collate to stack
items into a single ``(B, max_n, ...)`` batch. Padding is encoded via
``pad_mask`` (True = real node) and via ``y_edges = -100`` (NLL ignore).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .intermediate_tours import (
    STRATEGY_REGISTRY,
    STRATEGY_WEIGHTS_DEFAULT,
    sample_strategy_id,
)
from .tsp_data import TSPInstance


def _pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    """``(n, n)`` Euclidean distance matrix; ``D[i, i] = 0``."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def _tour_to_edge_graph(
    input_tour: np.ndarray, D: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 2-slot graph for one instance from a closed tour.

    For each node ``i`` we record the predecessor and successor in the
    cycle (the two neighbours of ``i``). With ``n = len(input_tour)``::

        prev[i] = input_tour[(k-1) % n]   where input_tour[k] == i
        next[i] = input_tour[(k+1) % n]   where input_tour[k] == i
        edge_index[i]  = [prev[i], next[i]]
        edge_feat[i]   = [D[i, prev[i]], D[i, next[i]]]
        inverse[i, s]  = position of i in edge_index[j, :] * n + slot

    where ``j = edge_index[i, s]``. In a closed tour, ``inverse[i, s]``
    is the flat position of the reverse edge ``(j, i)`` in the
    ``n*2``-sparse edge_index matrix.

    Args:
        input_tour: ``(n,)`` int64 permutation.
        D: ``(n, n)`` Euclidean distance matrix.

    Returns:
        edge_index: ``(n, 2)`` int64 0-indexed neighbour ids.
        edge_feat:  ``(n, 2)`` float32 distances.
        inverse_edge_index: ``(n, 2)`` int64; flat positions of reverse
            edges in the ``n*2`` sparse matrix.
    """
    n = input_tour.shape[0]
    position_of = np.empty(n, dtype=np.int64)
    position_of[input_tour] = np.arange(n, dtype=np.int64)

    pred_pos = (position_of - 1) % n
    succ_pos = (position_of + 1) % n
    prev = input_tour[pred_pos]
    nxt = input_tour[succ_pos]

    edge_index = np.stack([prev, nxt], axis=1)              # (n, 2)
    edge_feat = np.stack(
        [D[np.arange(n), prev], D[np.arange(n), nxt]], axis=1
    ).astype(np.float32)                                    # (n, 2)

    # For each j, build a dict {neighbour: slot_in_j} so the inverse of
    # (i, j) can be looked up in O(1). Stored as a Python list of dicts
    # rather than a (n, n) matrix to keep memory flat.
    slot_of_j_list: list[dict[int, int]] = [dict() for _ in range(n)]
    for j in range(n):
        slot_of_j_list[j][int(prev[j])] = 0
        slot_of_j_list[j][int(nxt[j])] = 1

    inv = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        for s in range(2):
            j = int(edge_index[i, s])
            if j < 0 or j >= n:
                inv[i, s] = -1
                continue
            slot_in_j = slot_of_j_list[j].get(i, -1)
            if slot_in_j < 0:
                inv[i, s] = -1
            else:
                inv[i, s] = j * 2 + slot_in_j

    return edge_index, edge_feat, inv


def _opt_edge_set(opt_tour: np.ndarray) -> set[tuple[int, int]]:
    """Return the set of undirected OPT edges as ``(min, max)`` pairs."""
    n = opt_tour.shape[0]
    edges: set[tuple[int, int]] = set()
    for k in range(n):
        a, b = int(opt_tour[k]), int(opt_tour[(k + 1) % n])
        edges.add((min(a, b), max(a, b)))
    return edges


def _derive_labels(
    edge_index: np.ndarray, opt_edges: set[tuple[int, int]]
) -> np.ndarray:
    """Per-slot binary label: ``1`` if ``(i, j)`` is an OPT edge."""
    n, k = edge_index.shape
    y = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        for s in range(k):
            j = int(edge_index[i, s])
            if j < 0:
                y[i, s] = -100
                continue
            y[i, s] = 1 if (min(i, j), max(i, j)) in opt_edges else 0
    return y


@dataclass
class _PadSpec:
    pad_to_n: int
    n_edges_per_node: int = 2


class EdgeQualityDataset(Dataset):
    """Padded, on-the-fly dataset for edge-quality prediction.

    Each item produces a dict suitable for stacking into a ``(B, max_n, ...)``
    batch. Padding is to a fixed ``pad_to_n`` (the largest n in the loaded
    instances, optionally capped).

    Args:
        instances: list of ``TSPInstance`` (parsed from disk).
        pad_to_n: padding length. Padded slots have ``y_edges=-100``,
            ``edge_index=-1``, ``edge_feat=0``, ``pad_mask=False``.
        strategy_weights: list of 5 floats summing to 1; controls the mix
            of strategies per item. ``None`` uses ``STRATEGY_WEIGHTS_DEFAULT``.
        fixed_strategy: if not ``None``, force this strategy id for every
            item. Used at eval time to bucket per-strategy metrics.
        kopt_n_moves: number of k-opt moves for the ``kopt`` strategy.
        kopt_p_3opt: probability of a 3-opt block reversal per iteration.
        seed: base seed for the per-worker random generator.
    """

    def __init__(
        self,
        instances: Sequence[TSPInstance],
        pad_to_n: int | None = None,
        strategy_weights: list[float] | None = None,
        fixed_strategy: int | None = None,
        kopt_n_moves: int = 5,
        kopt_p_3opt: float = 0.3,
        seed: int = 0,
    ):
        self.instances = list(instances)
        if not self.instances:
            raise ValueError("EdgeQualityDataset requires >= 1 instance")
        max_n = max(inst.coords.shape[0] for inst in self.instances)
        self.pad_to_n = pad_to_n if pad_to_n is not None else max_n
        if self.pad_to_n < max_n:
            raise ValueError(
                f"pad_to_n={self.pad_to_n} < max instance n={max_n}"
            )
        self.strategy_weights = (
            list(strategy_weights) if strategy_weights is not None
            else STRATEGY_WEIGHTS_DEFAULT
        )
        self.fixed_strategy = fixed_strategy
        self.kopt_n_moves = kopt_n_moves
        self.kopt_p_3opt = kopt_p_3opt
        self.seed = seed
        # Set lazily per worker; reset in __getitem__ if worker_init_fn.
        self._worker_id = 0

    def __len__(self) -> int:
        return len(self.instances)

    # ``torch.utils.data`` calls this on each worker subprocess.
    def worker_init_fn(self, worker_id: int) -> None:
        self._worker_id = int(worker_id)

    def __getitem__(self, idx: int) -> dict:
        inst = self.instances[idx]
        coords = np.asarray(inst.coords, dtype=np.float64)
        n = coords.shape[0]
        # ``SeedSequence`` derives per-worker, per-epoch, per-item streams
        # without contention across DataLoader workers.
        ss = np.random.SeedSequence([self.seed, self._worker_id, idx])
        rng = np.random.default_rng(ss)

        opt_tour = inst.opt_tour
        if opt_tour is None:
            # TSPlib: no OPT tour available. We cannot compute labels here.
            # The trainer should skip label-dependent metrics for these.
            # For now, fall back to a self-loop "opt" with the OPT edge set
            # set to {} so all labels are 0; the model can still forward.
            opt_tour = np.arange(n, dtype=np.int64)
            opt_edges: set[tuple[int, int]] = set()
        else:
            opt_edges = _opt_edge_set(np.asarray(opt_tour, dtype=np.int64))

        if self.fixed_strategy is not None:
            sid = self.fixed_strategy
        else:
            sid = sample_strategy_id(rng, self.strategy_weights)
        strategy_name = list(STRATEGY_REGISTRY)[sid]
        fn = STRATEGY_REGISTRY[strategy_name]

        if strategy_name == "kopt":
            input_tour = fn(coords, opt_tour, n_moves=self.kopt_n_moves,
                            p_3opt=self.kopt_p_3opt, rng=rng)
        elif strategy_name in ("nn", "fi", "random"):
            input_tour = fn(coords, opt_tour, rng=rng)
        else:  # opt
            input_tour = fn(coords, opt_tour, rng=rng)

        D = _pairwise_euclidean(coords)
        edge_index, edge_feat, inv = _tour_to_edge_graph(input_tour, D)
        y_edges = _derive_labels(edge_index, opt_edges)

        pad_n = self.pad_to_n
        pad = pad_n - n
        if pad > 0:
            coords_padded = np.concatenate(
                [coords, np.zeros((pad, 2), dtype=np.float64)], axis=0
            )
            edge_index = np.concatenate(
                [edge_index, np.full((pad, 2), -1, dtype=np.int64)], axis=0
            )
            edge_feat = np.concatenate(
                [edge_feat, np.zeros((pad, 2), dtype=np.float32)], axis=0
            )
            inv = np.concatenate(
                [inv, np.full((pad, 2), -1, dtype=np.int64)], axis=0
            )
            y_edges = np.concatenate(
                [y_edges, np.full((pad, 2), -100, dtype=np.int64)], axis=0
            )
        else:
            coords_padded = coords
        pad_mask = np.zeros(pad_n, dtype=bool)
        pad_mask[:n] = True

        # Cast to torch.
        out = {
            "coords": torch.from_numpy(coords_padded.astype(np.float32)),
            "edge_index": torch.from_numpy(edge_index.astype(np.int64)),
            "edge_feat": torch.from_numpy(edge_feat),
            "inverse_edge_index": torch.from_numpy(inv.astype(np.int64)),
            "y_edges": torch.from_numpy(y_edges),
            "pad_mask": torch.from_numpy(pad_mask),
            "strategy_id": torch.tensor(sid, dtype=torch.int64),
            "instance_index": torch.tensor(idx, dtype=torch.int64),
            "n": torch.tensor(n, dtype=torch.int64),
        }
        if opt_tour is not None:
            out["opt_tour"] = torch.from_numpy(np.asarray(opt_tour, dtype=np.int64))
        return out


def collate_edge_batch(items: list[dict]) -> dict:
    """Stack a list of dataset items into a ``(B, pad_n, ...)`` batch."""
    keys = items[0].keys()
    batch: dict = {}
    for k in keys:
        vals = [it[k] for it in items]
        if k == "opt_tour":
            # Variable-length per instance; keep as a list of 1-D tensors.
            batch[k] = vals
        else:
            try:
                batch[k] = torch.stack(vals, dim=0)
            except Exception as e:
                raise RuntimeError(
                    f"collate failed at key {k}: shapes {[v.shape for v in vals]}"
                ) from e
    return batch