# Pruning & Sparsity (Part 1)

- MLPerf $\rightarrow$ benchmark test suite for optimizing NNs (pruning)

- Two ways of pruning:
  - Depth pruning
  - Width pruning

## Neural Network Pruning

- neurons $\rightarrow$ neurons, synapses $\rightarrow$ connections between neurons
- Optimize the objective loss function $L$ with respective to the number of non-zero elements(neurons) $N$
- A few things need to be considered:

  - Pruning Granularity: in what pattern should we prune the network?
  - Pruning Criterion: what neurons / synapses should be pruned?
  - Pruning Ratio: what should be the sparsity for each layer after pruned? (aka how many of the neurons will be left intact)
  - Fine-tune the pruned NN: How to train and regain accuracy once pruned?

- Pruning $\rightarrow$ Pruning + Fine-tuning $\rightarrow$ Iterative (Pruning + Fine-tuning)

### Pruning Criterion

#### Pruning Synapses

- We want to remove the **less important** ones

- Magnitude-based pruning $\rightarrow$ a heuristic approach that prune weights with smaller absolute values

  - For **element-wise** pruning, $Importance = |W|$

    - Example: [-2, 3, -5, 1] $\rightarrow$ absolute values: [2, 3, 5, 1] $\rightarrow$ after prune: [0, 3, -5, 0]

  - For **row-wise** pruning, $Importance = \sum_{i\in S}|w_{i}|$,

- Scaling-based pruning $\rightarrow$ a scaling factor is associated with each filter in convolutional layers

  - Scaling factor is multiplied to the output of channel and is trainable parameters $\rightarrow$ smaller scaling factor $\rightarrow$ better

- Second-Order-based pruning
  - Minimize error on loss function introduced by pruning synapses
  - Read [Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf), Hessian Matrix

#### Pruning Neurons

- remove less useful neurons

- Percentage-Of-Zero-based pruning

  - ReLU activations will generate zeroes in the output activation
  - Average Percentage of Zero activations(APoZ) can be exploited to measure the importance of neurons
  - Small APoZ -> more important
  - Example: 2 batch, 3 channels, APoZ = $\frac{number \space of \space 0s}{total \space inputs}$

- Regression-based pruning

  - Minimize reconstruction error of the corresponding layer's outputs
  - Read [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/pdf/1707.06168)

- Over-parameterization helps with optimization because model may be released from a local minimum and find a better one

- [Slides](https://www.dropbox.com/scl/fi/6qspcmk8qayy7mft737gh/Lec03-Pruning-I.pdf?rlkey=9jpifc92be0sitiknpbhn9ggf&e=1&st=lml94lam&dl=0)
