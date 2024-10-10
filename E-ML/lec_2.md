## Basic of Neural Network

- synapses - weights - parameters -> same thing
- neurons -> features -> activations -> same thing

### NN Layers

- focus on the dimension of each tensor -> important when calculate FLOP, FLOPs (Floating-point Operation)
- Fully-connected (Linear) Layer
  - Output neuron connected to all input neurons
- Convoluted Layer
  - Output neuron connected to input neurons in the receptive field
  - Receptive field? Depicts how large the area that a neuron can capture information
  - Receptive field size = L \* (k - 1) + 1 with L -> number of layers, k is the kernel size
  - **Padding** helps to keep the feature map at same size as input
    - Zero Padding
    - Reflection Padding
    - Replication Padding
    - Constant Padding
- Problem with original convoluted layers? Need many layers to understand larger part of image -> downsample(move the sliding window faster)

- Grouped Convolution Layer
- Depthwise Convolution Layer
- Pooling Layer
  - Make feature map smaller -> dense information
  - Max Pooling or Average Pooling
- Normalization Layer
  - Make the optimization faster
  - Batch/Layer/Instance/Group Norm -> each serve a different purpose and use a different set of inputs
  - Fine-tuning usually focus on this layer because small # of parameters

### Activation Functions

- Non-linear functions: Sigmoid, ReLU, ReLU6, Leaky ReLU, Swish, Hard Swish

### Transformers

- Query-Key-Value design
  - Query: text prompt in Youtube search bar
  - Key: title or description of the videos
  - Value: corresponding videos

### Metrics to evalute efficiency of a model

- Latency: delay for a specific task
  - Latency = max($T_{computation}, T_{memory}$)
  - Computation -> cheap, data movement + memory access -> expensive
- Throughput: measure the rate at which data is processed
- Parameters:
- Model Size =# Parameters \* Bit Width
- Number of Activations -> bottleneck in both training and inference
