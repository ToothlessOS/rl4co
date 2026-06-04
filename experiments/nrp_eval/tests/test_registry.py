"""Tests for the SolverRegistry."""
from __future__ import annotations

import pytest

from nrp.solvers import Solver, SolverRegistry
from nrp.solvers.base import Solver as _Solver  # ensure importable


def test_registry_register_and_build():
    @SolverRegistry.register("dummy_test_solver", env_names=("tsp",))
    class DummySolver(Solver):
        def solve(self, td):
            return td

    # Both import paths yield the same class object.
    assert _Solver is Solver
    assert "dummy_test_solver" in SolverRegistry.available()
    assert SolverRegistry.supports("dummy_test_solver", "tsp")
    assert not SolverRegistry.supports("dummy_test_solver", "cvrp")
    assert SolverRegistry.available(env_name="tsp")  # at least one
    # Cleanup so we don't pollute the global registry for other tests.
    SolverRegistry.registry.pop("dummy_test_solver", None)
    SolverRegistry.env_support.pop("dummy_test_solver", None)


def test_registry_build_unknown_raises():
    with pytest.raises(ValueError, match="Unknown solver"):
        SolverRegistry.build("definitely_not_a_real_solver", env=None)
