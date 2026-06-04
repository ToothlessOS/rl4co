"""Tests for nrp.utils.metrics."""
import json
import math

import pytest

from nrp.utils.metrics import PerInstanceWriter, gap_to_optimal, tour_length_summary


def test_gap_to_optimal_basic():
    assert gap_to_optimal([10, 20, 30], [10, 10, 10]) == [0.0, 100.0, 200.0]


def test_gap_to_optimal_length_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        gap_to_optimal([1.0, 2.0], [1.0])


def test_tour_length_summary_basic():
    s = tour_length_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 5.0
    assert s["n"] == 5
    assert s["feasible_ratio"] == 1.0


def test_tour_length_summary_with_nan():
    s = tour_length_summary([1.0, float("nan"), 3.0])
    assert s["n"] == 3
    assert s["feasible_ratio"] == 2 / 3
    assert math.isfinite(s["mean"])


def test_tour_length_summary_empty():
    s = tour_length_summary([])
    assert s["n"] == 0
    assert s["feasible_ratio"] == 0.0
    assert math.isnan(s["mean"])


def test_per_instance_writer_roundtrip(tmp_path):
    path = tmp_path / "rows.jsonl"
    with PerInstanceWriter(path) as w:
        w.write({"i": 0, "tour": 1.5})
        w.write({"i": 1, "tour": 2.5})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"i": 0, "tour": 1.5}, {"i": 1, "tour": 2.5}]
