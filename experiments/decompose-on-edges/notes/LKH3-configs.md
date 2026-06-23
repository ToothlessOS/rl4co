In the original LKH-paper, the authors talked about $\alpha$-nearness that better reflects the chances of a given link being a member of an optimal tour, which is based on sensitivity analysis using minimum spanning 1-trees (something we had used as a lower bound in previous experiments).

A 1-tree for a grpah $G = (N, E)$ is a spanning tree on the node set n \ {1} combined with two edges from E incident to node 1, where node 1 is chosen arbitarily.

1. an optimal tour is a minimum 1-tree where every node has degree 2.

2. If a minimum 1-tree is a tour, then the tour is optimal.

The minimum 1-tree normally contains between 70% and 80% of the edges of a minimum 1-tree, a desirable property => Edges that belong, or "nearly belong" to a minimum 1-tree, stand a good chance of also belonging to an optimal tour. Conversely, edges that are "far from" belonging to a minimum 1-tree have a low probability of also belonging to an optimal tour. The $\alpha$-nearness of an edge (i, j) is defined as the differences in the lengths of a minimum 1-tree containing (i, j) and the minimum 1-tree, which can be computed in $O(n^2)$ time.

Since $\alpha$-nearness can be computed efficiently, I am considering using it to determine which parts of the path to combined into a tunnel - ideally, the tunnels should contain a series of edges that are closed to each other, while we allow the edges with high $$\alpha$-nearness$ to be changed.
