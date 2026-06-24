# Edge Quality Predictor

Take an intermediate TSP solution instance as input and predict the probability of each edge being in the optimal solution - a metric that can be used for decomposition.

# Data

Expert solutions are available in `data/TSP/`; Instances are stored in txt files, with each line containing one instance, in the format of coordinates + optimal tour sequence. The create a wrapper / dataloader, you can refer to the implementation in `ref/env/`

Realistic intermediate TSP solution instances can be obtained in the following two ways;

1. Use quick heuristics like nearnest neightbour or farthest insertion to generate intermediate solutions

2. Apply a few steps of random 2-opt and 3-opt to the optimal ground truth solutions, creating solutions that are slightly worse

3. Randomly connect the edges to generate random solutions

4. Directly use ground truth solutions

The final dataset should be a mix of the four types of data, with more instances of (1) and (2).

One key concern is if we can generate these instances on the fly - make sure you look into that.

# Model and input / output

We will use the `sgcn` model from NeuroLKH in `experiments/decompose-on-edges/NeuroLKH/net`, however, the input and output need to be rewired:

1. We only need to predict the edge scores in this case, and each edge should be equipped with an 0 / 1 feature to indicate whether it is part of the optimal solution or not

2. The rest of the input / output will stick to the NeuroLKH for now

3. The loss will be the cross entropy loss between the output and the actuall edges in expert solutions

Please generate a review of the section before coding, as it needs to be carefully considered. I am also a bit concerned about handling varying sizes of input instances.

# Implementation conventions

The model, the env / dataloader components must be cleanly organized. Use `hydra` for managing the configurations. If there are available features with `rl4co` APIs, make sure you use them.
