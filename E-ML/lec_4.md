# Pruning & Sparsity (Part 2)

## Pruning Ratio

- non-uniform pruning is better than uniform shrinking(aka every layer prunes the same sparsity ratio)

- how to know how much to prune with each layer? $\rightarrow$ sensitivity analysis

  - Some layers are prone to changes $\rightarrow$ accuracy may drop drastically
  - Some layers are more redundant

- Process:

  - Assume each layer is independent
  - Pick a layer $L_{i}$
    - Prune the layer by ratio $r \in {0.1, 0.2, ...., 0.9}$
    - Observe the accuracy degrade for each ratio
  - Repeat for each layer
  - Pick a degradation threshold $T$ that would meet the overall pruning ratio

- Not works great as it assumes independence between layers

### Automatic Pruning

- Given the pruning ratio, automatically prune each layer

- Pruning as a RL problem $\rightarrow$ AMC: [AutoML for Model Compression](https://arxiv.org/abs/1802.03494)

#### AMC

#### NetAdapt

- NetAdapt $\rightarrow$ rule-based iterative method to find a per-layer pruning ratio to meet a global resource constraint(latency, energy, ....)

  - For each iteration, we want to reduce the latency for a certain amount $\Delta R$
    - For each layer, prune so that the latency reduction meets $\Delta R$(based on a pre-built lookup table)
    - Short-term Fine-tune model(10k iterations), then measure accuracy after fine-tuning
  - Pick the layer with highest accuracy even after being pruned
  - Repeat until satisfies global constraint
  - Long-term Fine-tuning to recover accuracy

- Iterative nature gives us a series of models with different sparsity and accuracy

## Find-tuning/Training

- larger pruning ratio $\rightarrow$ larger accuracy decrease
- fine-tune helps with recover model's accuracy
  - Learning rate for fine-tuning usually 1/100 or 1/10 of the original learning rate
- Iterative pruning gradually increase possible sparsity while keeps accuracy stable

- Regularization $\rightarrow$ added to penalize non-zero parameters + encourage small parameters
  - L1 regularization: $L' = L(x;W) + \lambda|W|$
  - L2 regularization: $L' = L(x;W) + \lambda||W^{2}||$
  - Example: Magnitude-based fine-grained pruning $\rightarrow$ L2, while Network Slimming applies L1 to channel scaling factors

## Accelerated Sparse Network

### Efficient Inference Engine(EIE)

- First DNN Accelerator for Sparse, Compressed Model
  - Sparse Weight
  - Sparse Activation
  - Weight Sharing
