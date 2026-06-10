"""Barycenter-Clustering Decomposition (BCC) for CVRP.

Python port of the C++ implementation at
``cvrp-decomposition/src/hgs-TV/Decomposition/RouteSequence/BarycentreClusteringDecomposition.cpp``.

Algorithm:
    1. Take a master CVRP solution (a list of routes; each route is a list of
       customer ids).
    2. Compute the barycentre (centroid) of each non-empty route.
    3. k-means cluster the route barycentres into ``k = ceil(nbClients /
       targetMaxSpCustomers)`` groups.
    4. For each cluster, build a subproblem containing the union of customers
       in the routes assigned to that cluster.
    5. (LDE divergence from C++) Enforce ``sum(demand) <= vehicle_capacity``
       per cluster via a greedy walk-and-split. The original HGS sub-GA
       respected capacity implicitly; LKH-3 needs a feasible sub-CVRP.

Reference: Santini, Alberto, et al. "Decomposition strategies for vehicle
routing problems." _Computers & Operations Research_ (2023).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.cluster import KMeans
from tensordict import TensorDict

log = logging.getLogger(__name__)


@dataclass
class Subproblem:
    """A single CVRP subproblem.

    Attributes:
        customer_ids: 0-indexed customer ids (in the original instance space)
        xy: ``[n, 2]`` numpy array of customer coordinates
        demand: ``[n]`` numpy array of customer demands (integer, unnormalized)
        capacity: vehicle capacity
        depot_xy: ``[2]`` numpy array of the depot coordinates
    """

    customer_ids: list[int]
    xy: np.ndarray
    demand: np.ndarray
    capacity: int
    depot_xy: np.ndarray

    @property
    def num_loc(self) -> int:
        return len(self.customer_ids)

    def to_td(self) -> TensorDict:
        """Convert to a single-instance TensorDict compatible with CVRPEnv."""
        return TensorDict(
            {
                "depot": torch.as_tensor(self.depot_xy, dtype=torch.float32),
                "locs": torch.as_tensor(self.xy, dtype=torch.float32),
                "demand": torch.as_tensor(
                    self.demand / self.capacity, dtype=torch.float32
                ),
                "capacity": torch.as_tensor([self.capacity], dtype=torch.float32),
            },
            batch_size=[],
        )


class BarycentreClusteringDecomposer:
    """Decompose a CVRP master solution into k subproblems via k-means on
    route barycentres.

    Args:
        target_max_subproblem_size: target maximum number of customers per
            subproblem (the literature default is 200, matching
            ``targetMaxSpCustomers`` in the C++ code).
        kmeans_max_iter: maximum k-means iterations (default 100, matching
            ``KMeans.h:max_iter=100``).
        kmeans_tol: convergence tolerance in coordinate shift (default 1e-2,
            matching ``KMeans.h:different()``).
        enforce_capacity: if True, greedily split clusters whose total demand
            exceeds ``vehicle_capacity``. This is a deliberate divergence
            from the C++ port (see module docstring).
        random_state: random seed for k-means++ init.
    """

    def __init__(
        self,
        target_max_subproblem_size: int = 200,
        kmeans_max_iter: int = 100,
        kmeans_tol: float = 1e-2,
        enforce_capacity: bool = True,
        random_state: int = 0,
    ):
        self.target_max_subproblem_size = target_max_subproblem_size
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_tol = kmeans_tol
        self.enforce_capacity = enforce_capacity
        self.random_state = random_state

    def decompose(
        self,
        td: TensorDict,
        routes: list[list[int]],
    ) -> list[Subproblem]:
        """Decompose a CVRP instance + master routes into a list of subproblems.

        Args:
            td: a single-instance CVRP TensorDict with ``depot``, ``locs``,
                ``demand``, ``capacity``.
            routes: list of routes. Each route is a list of 0-indexed customer
                ids (depot is implicit at start and end).

        Returns:
            A list of ``Subproblem`` objects. Sum of customers across
            subproblems equals ``num_loc``; each subproblem is feasible (if
            ``enforce_capacity`` is True).
        """
        if td.batch_size != torch.Size([]) and len(td.batch_size) > 0:
            if len(td.batch_size) > 1:
                raise ValueError(
                    f"Expected a single-instance TensorDict, got {td.batch_size}"
                )
            td = td[0]

        # Normalize locs to [N+1, 2] with depot at index 0.
        if "locs" in td.keys() and "depot" in td.keys():
            locs = td["locs"]
            if locs.shape[-2] == td["demand"].shape[-1] + 1:
                all_locs = locs  # depot prepended
                depot_xy = locs[0]
                customer_xy = locs[1:]
            else:
                depot_xy = td["depot"].reshape(2)
                all_locs = torch.cat([depot_xy.unsqueeze(0), locs], dim=-2)
                customer_xy = locs
        else:
            raise KeyError("TensorDict must have 'locs' and 'depot' keys")

        # Convert normalised demand to integer demand.
        capacity = int(td["capacity"].reshape(()).item())
        demand_norm = td["demand"].reshape(-1)
        demand_int = (demand_norm * capacity).round().long().numpy()  # [N]

        n = customer_xy.shape[0]
        if n == 0:
            return []

        # Number of clusters.
        k = max(1, int(math.ceil(n / self.target_max_subproblem_size)))
        n_routes_input = len(routes)
        n_non_empty = sum(1 for r in routes if r)
        log.info(
            "decompose_start(n=%d, n_routes=%d, n_non_empty=%d, "
            "capacity=%d, k_initial=%d, target_max_subproblem_size=%d)",
            n,
            n_routes_input,
            n_non_empty,
            capacity,
            k,
            self.target_max_subproblem_size,
        )
        # Cap k by the number of non-empty routes; k-means requires
        # n_samples >= n_clusters. Empty routes will be distributed to
        # existing clusters below.
        # (We delay the actual capping until we know how many non-empty
        # routes exist — see below.)

        # Step 1: barycentres of non-empty routes.
        barycentres: list[np.ndarray] = []
        non_empty_idx: list[int] = []
        for i, route in enumerate(routes):
            if not route:
                continue
            ids = torch.as_tensor(route, dtype=torch.long)
            xy = customer_xy[ids].numpy()  # [r, 2]
            barycentres.append(xy.mean(axis=0))
            non_empty_idx.append(i)

        # Cap k by the number of non-empty routes (k-means needs n_samples >= k).
        if barycentres:
            k_capped = min(k, len(barycentres))
            if k_capped != k:
                log.info(
                    "k_capped(k_before=%d, k_after=%d, n_non_empty_routes=%d)",
                    k,
                    k_capped,
                    len(barycentres),
                )
            k = k_capped

        subproblems: list[Subproblem] = []

        if k == 1 or not barycentres:
            # Trivial case: all customers in one subproblem.
            all_ids = sorted({c for r in routes for c in r})
            if not all_ids:
                return []
            subproblems = [
                Subproblem(
                    customer_ids=all_ids,
                    xy=customer_xy[torch.as_tensor(all_ids, dtype=torch.long)].numpy(),
                    demand=demand_int[all_ids],
                    capacity=capacity,
                    depot_xy=depot_xy.numpy(),
                )
            ]
        else:
            # Step 2: k-means on barycentres.
            centres_arr = np.stack(barycentres, axis=0)  # [R', 2]
            # sklearn KMeans with k-means++ init, single rest, fixed seed.
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=1,
                max_iter=self.kmeans_max_iter,
                tol=self.kmeans_tol,
                random_state=self.random_state,
            )
            labels = km.fit_predict(centres_arr)  # [R']

            # Step 3: empty routes (if any) get distributed round-robin to clusters.
            empty_idx = [i for i, r in enumerate(routes) if not r]
            cluster_to_routes: list[list[int]] = [[] for _ in range(k)]
            for r_idx, c_idx in zip(non_empty_idx, labels):
                cluster_to_routes[int(c_idx)].append(r_idx)
            cursor = 0
            for r_idx in empty_idx:
                cluster_to_routes[cursor % k].append(r_idx)
                cursor += 1

            # Step 4: build subproblems.
            for cluster_routes in cluster_to_routes:
                if not cluster_routes:
                    continue
                cust_ids = sorted({c for r in cluster_routes for c in routes[r]})
                if not cust_ids:
                    continue
                sp = Subproblem(
                    customer_ids=cust_ids,
                    xy=customer_xy[torch.as_tensor(cust_ids, dtype=torch.long)].numpy(),
                    demand=demand_int[cust_ids],
                    capacity=capacity,
                    depot_xy=depot_xy.numpy(),
                )
                subproblems.append(sp)

        # Step 5 (LDE divergence): enforce capacity via greedy walk-and-split.
        if self.enforce_capacity:
            subproblems = self._split_by_capacity(subproblems, capacity)

        # Drop any empty subproblems that may have been produced.
        subproblems = [sp for sp in subproblems if sp.num_loc > 0]
        if subproblems:
            sizes = [sp.num_loc for sp in subproblems]
            demand_pct = [
                round(float(sp.demand.sum()) / max(1, sp.capacity), 2)
                for sp in subproblems
            ]
        else:
            sizes = []
            demand_pct = []
        log.info(
            "decompose_done(k=%d, sizes=%s, demand_pct=%s, "
            "input_routes=%d, dropped_empty=%d)",
            len(subproblems),
            sizes,
            demand_pct,
            n_routes_input,
            n_routes_input - sum(1 for r in routes if r),
        )
        return subproblems

    @staticmethod
    def _split_by_capacity(
        subproblems: list[Subproblem], capacity: int
    ) -> list[Subproblem]:
        """Greedy walk-and-split: for each subproblem whose total demand
        exceeds the capacity, move customers (in the order they appear) into
        new subproblems until the remainder fits.

        Customers are moved in *demand-descending* order so we don't strand
        high-demand customers in tiny leftover clusters.
        """
        result: list[Subproblem] = []
        for sp in subproblems:
            order = sorted(
                range(len(sp.customer_ids)),
                key=lambda i: -int(sp.demand[i]),
            )
            remaining_ids = list(order)
            current_ids: list[int] = []
            current_demand = 0
            while remaining_ids:
                cid = remaining_ids.pop(0)
                d = int(sp.demand[cid])
                if current_demand + d <= capacity:
                    current_ids.append(cid)
                    current_demand += d
                else:
                    if current_ids:
                        result.append(
                            Subproblem(
                                customer_ids=[sp.customer_ids[i] for i in current_ids],
                                xy=sp.xy[current_ids],
                                demand=sp.demand[current_ids],
                                capacity=capacity,
                                depot_xy=sp.depot_xy,
                            )
                        )
                    current_ids = [cid]
                    current_demand = d
            if current_ids:
                result.append(
                    Subproblem(
                        customer_ids=[sp.customer_ids[i] for i in current_ids],
                        xy=sp.xy[current_ids],
                        demand=sp.demand[current_ids],
                        capacity=capacity,
                        depot_xy=sp.depot_xy,
                    )
                )
        return result
