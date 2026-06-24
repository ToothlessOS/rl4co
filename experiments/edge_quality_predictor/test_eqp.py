"""Smoke tests + end-to-end pipeline validation for the edge quality predictor.

Run from ``experiments/edge_quality_predictor/`` with the parent ``.venv``
activated::

    python test_eqp.py

Each test prints ``OK`` (or ``FAIL`` with a traceback) and exits 1 on any
failure. The script is intentionally dependency-free beyond torch /
numpy — no pytest required.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA_TSP = ROOT / "data" / "TSP"


def _run(name: str, fn) -> bool:
    try:
        fn()
        print(f"  OK   {name}")
        return True
    except Exception:
        traceback.print_exc()
        print(f"  FAIL {name}")
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_train_line():
    from eqp.data import load_train_file

    inst = load_train_file(
        DATA_TSP / "training dataset" / "train_TSP100_n100w-002.txt",
        max_instances=1,
    )[0]
    assert inst.coords.shape == (100, 2)
    assert inst.opt_tour.shape == (100,)
    assert sorted(inst.opt_tour.tolist()) == list(range(100)), "tour must be a permutation"


def test_parse_tsplib():
    from eqp.data import load_tsplib_file

    inst = load_tsplib_file(
        DATA_TSP / "testing dataset" / "TSPlib_70instances.txt",
        max_instances=1,
    )[0]
    assert inst.coords.ndim == 2 and inst.coords.shape[1] == 2
    assert inst.coords.shape[0] >= 3
    assert inst.opt_tour is None
    assert "@" in inst.name, "name should encode instance@cost"


def test_intermediate_tours():
    from eqp.data import load_train_file
    from eqp.data.intermediate_tours import (
        STRATEGY_REGISTRY,
        kopt_perturb_tsp,
        opt_passthrough,
        random_edge_tour,
    )

    train = load_train_file(
        DATA_TSP / "training dataset" / "train_TSP100_n100w-002.txt",
        max_instances=2,
    )
    coords = train[0].coords
    opt = train[0].opt_tour

    # Every strategy returns a valid permutation.
    for name, fn in STRATEGY_REGISTRY.items():
        rng = np.random.default_rng(42)
        if name == "kopt":
            tour = fn(coords, opt, n_moves=3, p_3opt=0.3, rng=rng)
        elif name in ("nn", "fi", "random"):
            tour = fn(coords, opt, rng=rng)
        else:
            tour = fn(coords, opt, rng=rng)
        assert sorted(tour.tolist()) == list(range(100)), f"{name} not a permutation"

    # opt == opt
    opt_t = opt_passthrough(coords, opt)
    assert np.array_equal(opt_t, opt)

    # kopt changes some edges (2-opt moves swap endpoints). Most edges
    # in a kopt-perturbed tour should still match OPT though, since only
    # a few moves are applied. We verify "mostly the same" rather than
    # exact equality (kopt drops the (i-1, i) and (j, j+1) edges and
    # adds (i-1, j) and (i, j+1), so each move changes at most 4 edges).
    kopt_t = kopt_perturb_tsp(coords, opt, n_moves=3, p_3opt=0.0,
                              rng=np.random.default_rng(0))
    edges_opt = {(min(int(opt[k]), int(opt[(k + 1) % 100])),
                  max(int(opt[k]), int(opt[(k + 1) % 100]))) for k in range(100)}
    edges_kopt = {(min(int(kopt_t[k]), int(kopt_t[(k + 1) % 100])),
                   max(int(kopt_t[k]), int(kopt_t[(k + 1) % 100]))) for k in range(100)}
    overlap = len(edges_opt & edges_kopt)
    assert overlap >= 100 - 4 * 3, f"3 kopt moves should drop at most 12 edges, got {overlap}"

    # random is not the OPT
    random_t = random_edge_tour(coords, rng=np.random.default_rng(0))
    assert not np.array_equal(random_t, opt)


def test_edge_graph_build():
    from eqp.data.dataset import _pairwise_euclidean, _tour_to_edge_graph

    # Tiny 4-node example with known tour.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    # Cycle 0 -> 1 -> 2 -> 3 -> 0
    perm = np.array([0, 1, 2, 3], dtype=np.int64)
    D = _pairwise_euclidean(coords)
    edge_index, edge_feat, inv = _tour_to_edge_graph(perm, D)
    # Node 0's neighbours: prev=3, next=1.
    assert edge_index[0, 0] == 3 and edge_index[0, 1] == 1
    # Edge features are distances.
    assert np.isclose(edge_feat[0, 0], 1.0) and np.isclose(edge_feat[0, 1], 1.0)
    # Flat layout is (n * n_edges_per_node) = (4 * 2) = 8 entries.
    # inv[i, s] = j * n_edges_per_node + slot_in_j_for_i.
    # inv[0, 1]: j = edge_index[0, 1] = 1. Node 1's set = [0, 2], so node 0
    # appears at slot 0. inv = 1*2 + 0 = 2.
    assert inv[0, 1] == 1 * 2 + 0, f"got {inv[0, 1]}, expected 2"
    # inv[1, 0]: j = edge_index[1, 0] = 0. Node 0's set = [3, 1], so node 1
    # appears at slot 1. inv = 0*2 + 1 = 1.
    assert inv[1, 0] == 0 * 2 + 1, f"got {inv[1, 0]}, expected 1"
    # Every inverse should be a valid flat position.
    assert (inv >= 0).all()
    assert (inv < 4 * 2).all()


def test_label_derivation():
    from eqp.data.dataset import _derive_labels, _tour_to_edge_graph, _opt_edge_set, _pairwise_euclidean

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    opt = np.array([0, 1, 2, 3], dtype=np.int64)
    D = _pairwise_euclidean(coords)
    edge_index, _, _ = _tour_to_edge_graph(opt, D)
    opt_edges = _opt_edge_set(opt)
    y = _derive_labels(edge_index, opt_edges)
    # Each directed OPT edge contributes 1 positive per endpoint; total 2n=8.
    assert (y == 1).sum() == 2 * 4, f"expected 8 positives, got {(y == 1).sum()}"
    # All labels for OPT tour = all-1 since input_tour == opt_tour.
    assert (y == 0).sum() == 0


def test_sgn_v_forward():
    from eqp.model import SparseGCNModelV

    model = SparseGCNModelV(hidden_dim=16, n_gcn_layers=2, n_mlp_layers=2, n_edges_per_node=2)
    n = 10
    B = 2
    coords = torch.randn(B, n, 2)
    edge_feat = torch.rand(B, n * 2, 1)
    edge_index = torch.randint(0, n, (B, n * 2))
    inv = torch.randint(0, n * 2, (B, n * 2))
    y = torch.zeros(B, n * 2, dtype=torch.long)
    cw = torch.tensor([1.0, 1.0])
    yp, loss, yn = model(coords, edge_feat, edge_index, inv, y, cw, n_edges=2)
    assert yp.shape == (B, n * 2, 2)
    assert loss is not None and torch.isfinite(loss).all()
    assert yn.shape == (B, n, 1)


def test_classifier_forward():
    from eqp.data import EdgeQualityDataset, collate_edge_batch, load_train_file
    from eqp.model import SGNEdgeClassifier

    train = load_train_file(
        DATA_TSP / "training dataset" / "train_TSP100_n100w-002.txt",
        max_instances=4,
    )
    ds = EdgeQualityDataset(train, pad_to_n=100)
    batch = collate_edge_batch([ds[i] for i in range(4)])
    clf = SGNEdgeClassifier(hidden_dim=32, n_gcn_layers=5, n_mlp_layers=2,
                             n_edges_per_node=2, pos_weight=9.0)
    yp, loss = clf(batch)
    assert yp.shape == (4, 200, 2)
    assert torch.isfinite(loss).all()
    # Backward pass.
    loss.backward()
    # Predict scores.
    score = clf.predict_edge_scores(batch)
    assert score.shape == (4, 100, 100)
    # Diagonal is zero.
    for b in range(4):
        assert (score[b].diagonal() == 0).all()


def test_pad_mask_correctness():
    from eqp.data import EdgeQualityDataset, collate_edge_batch, load_train_file
    from eqp.model.losses import masked_nll_loss

    # Mixed-size: take 3 instances of TSP100 and verify loss only counts valid slots.
    train = load_train_file(
        DATA_TSP / "training dataset" / "train_TSP100_n100w-002.txt",
        max_instances=3,
    )
    ds = EdgeQualityDataset(train, pad_to_n=100, fixed_strategy=4)  # "opt" strategy
    batch = collate_edge_batch([ds[i] for i in range(3)])
    # y_edges is all-1 for the OPT strategy; pad slots = -100.
    valid_y = batch["y_edges"] != -100
    n_valid = int(valid_y.sum().item())
    n_padded = int((batch["y_edges"] == -100).sum().item())
    assert n_valid == 3 * 100 * 2
    assert n_padded == 0  # no padding needed since all instances are n=100

    # Now construct a synthetic mixed-size batch: instance 0 = n=80,
    # instance 1 = n=100, instance 2 = n=60 (use the same coords, then pad).
    coords = train[0].coords
    opt = train[0].opt_tour
    # Build proper sub-tours restricted to the first k node indices.
    from eqp.data import TSPInstance
    def _subtour(opt, k):
        # Keep only nodes in [0, k); map them to 0..k-1.
        mask = opt < k
        sub = opt[mask]
        # Re-index to 0..k-1.
        return sub.argsort().astype(np.int64)
    mixed = [
        TSPInstance(coords=coords[:80], opt_tour=_subtour(opt, 80)),
        TSPInstance(coords=coords[:100], opt_tour=opt),
        TSPInstance(coords=coords[:60], opt_tour=_subtour(opt, 60)),
    ]
    ds_mixed = EdgeQualityDataset(mixed, pad_to_n=100, fixed_strategy=4)
    batch_mixed = collate_edge_batch([ds_mixed[i] for i in range(3)])
    assert batch_mixed["pad_mask"][0].sum() == 80
    assert batch_mixed["pad_mask"][1].sum() == 100
    assert batch_mixed["pad_mask"][2].sum() == 60

    # Run a forward, check loss is finite and grad flows.
    yp = torch.log(torch.full((3, 200, 2), 0.5))
    yp.requires_grad_(True)
    # Flatten y_edges from (B, max_n, k) to (B, max_n*k) for masked_nll_loss.
    y_edges_flat = batch_mixed["y_edges"].reshape(3, -1)
    loss = masked_nll_loss(yp, y_edges_flat, batch_mixed["pad_mask"], n_edges_per_node=2)
    assert torch.isfinite(loss).all()
    loss.backward()


def test_metrics_basic():
    from eqp.eval import aggregate_metrics, edge_quality_metrics

    # Construct a synthetic case: 4 nodes, all-1 labels (OPT strategy).
    pred_probs = np.array([[0.9, 0.8], [0.7, 0.6], [0.95, 0.5], [0.4, 0.55]])
    y_edges = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
    edge_index = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])
    pad_mask = np.array([True, True, True, True])
    opt_tour = np.array([0, 1, 2, 3])
    m = edge_quality_metrics(pred_probs, y_edges, edge_index, pad_mask, opt_tour)
    assert 0.0 <= m["edge_accuracy"] <= 1.0
    assert 0.0 <= m["precision_at_1"] <= 1.0
    # Sanity: aggregate works.
    agg = aggregate_metrics([m, m])
    assert agg["edge_accuracy"] == m["edge_accuracy"]


def test_end_to_end_short():
    """Full Lightning fit on 16 train / 8 val, batch_size=4, max_epochs=1, small model."""
    import lightning.pytorch as pl
    from eqp.lightning import EdgeQualityDataModule, EdgeQualityModule

    pl.seed_everything(0, workers=True)

    data_cfg = {
        "train_path": str(DATA_TSP / "training dataset" / "train_TSP100_n100w-002.txt"),
        "val_path": str(DATA_TSP / "testing dataset" / "test_TSP100_n1w.txt"),
        "test_paths": [str(DATA_TSP / "testing dataset" / "test_TSP100_n1w.txt")],
        "train_limit": 16,
        "val_limit": 8,
        "test_limit": 8,
        "pad_to_n": 100,
        "strategy_weights": [0.20, 0.15, 0.40, 0.15, 0.10],
        "kopt_n_moves": 5,
        "kopt_p_3opt": 0.3,
        "num_workers": 0,
        "batch_size": 4,
    }
    model_cfg = {"hidden_dim": 32, "n_gcn_layers": 5, "n_mlp_layers": 2,
                 "n_edges_per_node": 2, "pos_weight": 9.0}
    optim_cfg = {"lr": 1e-4, "weight_decay": 0.0}

    dm = EdgeQualityDataModule(data_cfg)
    module = EdgeQualityModule(model_cfg, optim_cfg)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                          enable_checkpointing=False, precision="32-true")
    trainer.fit(module, datamodule=dm)
    assert "val/loss" in trainer.logged_metrics
    assert "val/edge_accuracy" in trainer.logged_metrics


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("parse_train_line",          test_parse_train_line),
        ("parse_tsplib",              test_parse_tsplib),
        ("intermediate_tours",        test_intermediate_tours),
        ("edge_graph_build",          test_edge_graph_build),
        ("label_derivation",          test_label_derivation),
        ("sgn_v_forward",             test_sgn_v_forward),
        ("classifier_forward",        test_classifier_forward),
        ("pad_mask_correctness",      test_pad_mask_correctness),
        ("metrics_basic",             test_metrics_basic),
        ("end_to_end_short",          test_end_to_end_short),
    ]
    print(f"Running {len(tests)} tests...\n")
    passed = sum(_run(name, fn) for name, fn in tests)
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())