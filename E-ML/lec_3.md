# Pruning & Sparsity (Part 1)

- MLPerf -> benchmark test suite for optimizing NNs (pruning)

- Two ways of pruning:
  - Depth pruning
  - Width pruning

## Neural Network Pruning

- neurons -> neurons, synapses -> connections between neurons
- Optimize the objective loss function $L$ with respective to the number of non-zero elements(neurons) $N$
- A few things need to be considered:

  - Pruning Granularity: in what pattern should we prune the network?
  - Pruning Criterion: what neurons / synapses should be pruned?
  - Pruning Ratio: what should be the sparsity for each layer after pruned? (aka how many of the neurons will be left intact)
  - Fine-tune the pruned NN: How to train and regain accuracy once pruned?

- Pruning -> Pruning + Fine-tuning -> Iterative (Pruning + Fine-tuning)
