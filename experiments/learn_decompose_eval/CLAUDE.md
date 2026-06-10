# About this experiment

This experiment aim to check whether learning to decompose methods for solving vehicle routing problem works better that hand-crafted heuristics for decomposition

We will use the `barycenter clustering` decomposition method included in the `cvrp-decomposition` repo with `LKH-3` and evaluate the performance on the generated CVRP dataset (uniform sampling, coordinates normalized to [0, 1] and demand of each node normalized by vehicle capacity) (which is provided through RL4CO).

The target is to compare raw LKH-3 for CVRP problems and LKH-3 augmented with `barycenter clustering` decomposition.