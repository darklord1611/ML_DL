## Neural Network
- sequence of inputs and outputs, outputs of a neuron will be the input of another
- inputs -> some computation -> outputs
- artificial neural network(ANN) or deep learning builds multiple neurons at same time
- now? not looking at biological motivation but rather engineering principles
- why now? more data + linear/logistic regression's performance not keeping up with amount of data
- neural networks can scale with data, take advantage of GPUs
- denote activation $a = f(x)$ with $f(x)$ is a sigmoid function
    - a neuron take input x, do some computation, output a(a probability)
- neural network? -> a bunch of neurons wiring together
- a group of neurons take similar inputs, generate few ouputs -> a layer
- neurons use inputs to compute new values -> activation values
- input layer aka a feature vector $\vec x$ -> fed to multiple hidden layers, generate a new input vector -> reapeat, ... -> output layer
- purpose of layer -> compute more appropriate features to feed to the next layer -> output
- challenge? how many layers? how many neurons for each layer? -> ultimately concern about architecture of NN
- NN with multiple layers? multilayer perceptron

