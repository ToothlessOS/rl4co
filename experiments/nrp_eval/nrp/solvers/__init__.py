"""Solver abstractions for the NRP evaluation pipeline."""
from .base import Solver, SolverRegistry
from .classical import (
    BuiltinEnvSolver,
    ClassicalSolver,
    ConcordeTSPSolver,
    GurobiTSPSolver,
    LKHSolver,
    ORToolsTSPSolver,
    ORToolsVRPSolver,
)
from .rl import RLSolver

__all__ = [
    "Solver",
    "SolverRegistry",
    "RLSolver",
    "ClassicalSolver",
    "ORToolsTSPSolver",
    "ORToolsVRPSolver",
    "BuiltinEnvSolver",
    "LKHSolver",
    "ConcordeTSPSolver",
    "GurobiTSPSolver",
]
