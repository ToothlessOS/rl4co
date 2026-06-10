"""learn_decompose_eval — LKH-3 ± BCC CVRP evaluation pipeline."""

# Importing the solvers package triggers the side-effect of registering
# the LKH-3 solvers with `nrp.solvers.SolverRegistry` (see classical_lkh.py).
from . import solvers  # noqa: F401

__version__ = "0.1.0"
