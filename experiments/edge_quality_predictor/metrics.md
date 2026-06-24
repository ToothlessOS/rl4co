# Metrics Reference

Per-instance and per-strategy metrics logged by the edge quality
predictor's training, validation, and test pipelines.

> **Source of truth.** All metrics are computed in
> `eqp/eval/edge_metrics.py::edge_quality_metrics` and
> `aggregate_metrics`. The Lightning module logs them at
> `eqp/lightning/module.py:103` (`_flush_metrics`).

## Per-instance setup

For one TSP instance the model produces a probability for each of the
`2n` directed slot-edges (each node contributes two slots: predecessor
and successor in the input tour). For example, if node 7 in the input
tour has predecessor 12 and successor 3, the model emits:

```
P((7 → 12) ∈ OPT)   and   P((7 → 3) ∈ OPT)
```

Each slot is labelled 0 or 1 depending on whether the corresponding
undirected edge appears in the OPT tour. With `n = 100` we have 200
predictions per instance, of which roughly `2n / n² ≈ 2%` are positive
(OPT edges).

All metrics below are computed only over **non-padded slots** for a
given instance. Padded slots are filtered via the `pad_mask`.

## Metrics

### `edge_accuracy`

Fraction of all `(node, slot)` entries where `argmax(P(in-OPT)) ==
label`. With k=2 slots per node there are `2n` predictions per instance
and accuracy is `(correct) / (2n)`.

- **Best-case value:** 1.0
- **Untrained baseline:** ~0.50 (uniform 50/50 on a 1:9 imbalanced split)
- **Limitation:** Hides class imbalance — predicting all-zeros on a 1:9
  split gives ~90% accuracy but catches no OPT edges. Pair with
  `balanced_accuracy` to expose that failure mode.

### `balanced_accuracy`

Mean of per-class recall: `(recall_on_negatives + recall_on_positives) /
2`. For our setting: how often we get "this edge is NOT in OPT" right,
and how often we get "this edge IS in OPT" right, averaged.

- **Best-case value:** 1.0
- **Untrained baseline:** ~0.50 (each class recalled at ~50%)
- **Why it matters:** Pairs with `edge_accuracy` to expose whether the
  model is just predicting the majority class.

### `precision_at_1`

For each node, take the slot with the higher P(in-OPT) (the model's
"best guess" edge for that node). Then: of those `n` best guesses, what
fraction are actually in the OPT tour?

- **Best-case value:** 1.0
- **Untrained baseline:** matches the OPT edge fraction in the input
  tour. For random tours on TSP-100, very low (~0.01–0.05). For
  kopt-perturbed tours, ~0.85+ because kopt preserves most OPT edges.
- **Why it matters:** This is the actionable decomposition signal —
  "given a tour, which edges should I trust as OPT?".

### `precision_at_2`

Fraction of all `2n` slot-edges that are in OPT. With k=2 slots/node
this is essentially the fraction of input-tour edges that match OPT.

- **Best-case value:** 1.0
- **Why it's a fixed property of the (strategy, instance) pair, not the
  model:** With only 2 slots/node the model has to score ALL input edges,
  so precision@2 doesn't depend on the model at all — it's the tour's
  intrinsic overlap with OPT. Useful as a "how easy is this instance"
  diagnostic.

### `top1_hit_rate`

For each undirected OPT edge `{a, b}`, check whether the model's
**top-1** pick at node `a` OR node `b` lands on the OPT edge. Then
`top1_hit_rate = (OPT edges hit) / (OPT edges total)`.

- **Best-case value:** 1.0
- **Why it matters:** The strictest metric — it requires the model to
  pick the OPT edge *preferentially* at both endpoints, not just include
  it among the two candidates.

### `mean_p_in_opt`

Average predicted P(in-OPT) on slots that ARE actually in the OPT tour.

- **Best-case value:** 1.0
- **Untrained baseline:** ~0.50 (uniform prediction)
- **Why it matters:** Calibration on the positive class. After training
  this should move toward 1.0; before training the model is uncertain.

### `mean_p`

Average predicted P(in-OPT) over **all** slots (positive + negative).

- **Best-case value:** depends — ideally the mean of OPT and non-OPT
  probabilities weighted by their respective prevalences.
- **Why it matters:** Sanity check that the model isn't degenerately
  predicting all-0 or all-1.

### `pos_pred_rate`

Fraction of slots where the model's predicted class is positive
(P(in-OPT) ≥ 0.5).

- **Ideal value:** equals the OPT edge fraction in the input tour. For
  a random tour on TSP-100 the OPT fraction is `2n / n² ≈ 0.02`, so the
  ideal `pos_pred_rate` is ~0.02.
- **Untrained baseline:** ~0.50 (uniform).
- **Why it matters:** Together with `mean_p_in_opt`, tells you if the
  model has learned to be selective — predicting positive only when
  confident.

### `neg_pred_rate`

The complement: fraction of slots predicted negative
(P(in-OPT) < 0.5). Always `1 - pos_pred_rate`.

## Metrics intentionally omitted

A few common metrics we don't compute, and why:

- **`PR-AUC`** — With only 2 slots/node the per-instance ranking problem
  is trivial (just two scores). The 4 cells of the confusion matrix at
  threshold 0.5 already capture everything meaningful. Per-slot ranking
  makes PR-AUC degenerate.
- **`F1 / mAP`** — With per-instance n=100 and only ~2 OPT edges per
  node, the absolute counts are noisy. We report the *per-strategy*
  mean so individual instance outliers get averaged out.
- **`auroc`** — Same reasoning as PR-AUC.

## Per-strategy bucketing

Every metric above is also reported per strategy. The wandb key looks
like `val/strategy_kopt/precision_at_1`,
`test/strategy_random/edge_accuracy`, etc.

The `_flush_metrics` function in the Lightning module buckets
per-instance metrics by their `strategy_id` (sampled in
`EdgeQualityDataset.__getitem__` from `STRATEGY_WEIGHTS_DEFAULT`).

### Typical per-strategy patterns

Untrained model on TSP-100:

| Strategy | Typical `precision_at_2` | Why |
|---|---|---|
| `opt` | 1.00 (label is always 1) | input tour IS the OPT |
| `kopt` | 0.85–0.95 | only a few 2-opt moves changed edges |
| `nn` / `fi` | 0.50–0.70 | heuristics pick many OPT edges by chance |
| `random` | ~0.02 (≈ 2n/n²) | random tour edge ≈ OPT edge is rare |

After training you want `precision_at_1`, `top1_hit_rate`, and
`balanced_accuracy` to rise across **all five** strategies — that's the
proof the model is using the graph structure rather than memorizing
"the input tour usually = the OPT".

## Sanity invariants on an untrained model

When you run a fresh model on the validation set, you should see
approximately:

- `edge_accuracy ≈ 0.50` (random)
- `balanced_accuracy ≈ 0.50` (random)
- `precision_at_2 ≈ fraction of OPT edges among the 2 input edges`:
  0.50–0.71 across the 4 mixed strategies on a typical instance
- `pos_pred_rate ≈ 0.50` (uniform threshold)
- `mean_p_in_opt ≈ 0.50` (uniform)

If any of these are wildly off, the model is misconfigured (e.g. an
unintended bias toward class 1).

## How the metrics appear in wandb

The Lightning module logs to wandb with the following naming
convention:

```
{stage}/{metric}                            # aggregate over all instances
{stage}/strategy_{name}/{metric}            # per-strategy aggregate
```

where `stage ∈ {train, val, test}` and `name ∈ {nn, fi, kopt, random,
opt}`. So a typical wandb run will show metrics like:

- `val/loss`
- `val/edge_accuracy`
- `val/balanced_accuracy`
- `val/precision_at_1`
- `val/strategy_kopt/precision_at_1`
- `val/strategy_random/top1_hit_rate`
- `test/edge_accuracy`
- `test/strategy_opt/edge_accuracy`  ← should be ~1.0 (input tour is OPT)

The checkpoint callback (`configs/callbacks/default.yaml`) monitors
`val/edge_accuracy` (mode=`max`) and saves the top-1 model.