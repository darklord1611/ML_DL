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

### NN layer

- each neuron in a layer has parameters $w_{1}$, $b_{1}$
- output of a neuron is some activation value $a_{1}$ of a logistic unit
- [i] denotes the $i_{th}$ layer -> $w_{1}^{[i]}$, $b_{1}^{[i]}$
- output of a layer is activation vector $\vec a^{[i]}$ used as input for next layer, alongside with that layer's weights and biases
  - $a_{j}^{[l]} = g(\vec w_{j}^{[l]} \cdot \vec a^{[l - 1]} + b_{j}^{[l]})$
  - layer $l$, neuron unit $j$, $g()$ is activation function(output activation values)
  - input $\vec x$ now is $\vec a^{[0]}$

## Forward propagation

- start calculating from input vector $\vec x$ -> layer by layer -> forward propagation
- next layer will have fewer neurons compared to current layer
- TensorFlow using matrices to handle data

```
x = np.array([[200, 0.17]])
layer_1 = Dense(units=3, activation="sigmoid")
a1 = layer_1(x)
layer_2 = Dense(units=1, activation="sigmoid")
a2 = layer_1(a1)
```

```
layer_1 = Dense(units=3, activation="sigmoid")
layer_2 = Dense(units=1, activation="sigmoid")

model = Sequential([layer_1, layer_2])

input_x = np.array([[...], [...]])
input_y = np.array([...])

model.compile(...) # more on this later
model.fit(input_x, input_y)

model.predict(new_x)
```

- AI consists of: artificial narrow intelligence(ANI) vs artificial general intelligence(AGI)

### NN Training

- model specification -> defining loss, cost function -> minimize the cost function
- map to training NN? specify how to compute outputs given $\vec x$, $\vec w$, b

```
# Specify loss function for classification
model.compile(loss = BinaryCrossentropy())

# Specify loss function for classification
model.compile(loss = MeanSquareError())

```

### Activation Functions

- Rectified Linear Unit(ReLU) is an activation function $g(z) = max(0, z)$ allow for much more diverse activation values
- Linear activation function $g(z) = z$, basically not using any activation function as well
- Choosing activation function for output layer? -> Depends on type of problem
  - binary classification -> sigmoid
  - regression(allowed negative) -> linear
  - regression(non-negative) -> ReLU
- How about hidden layers? -> mostly ReLU
  - Why? less computation, faster gradient descent
- Why use AF at all?
  - Suppose we use linear AF, output will be just another linear regression

### Multiclass

- softmax regression -> general version of logistic regression (more than 2 possible discrete outputs)
  - $z_{i} = \vec w_{i} \cdot \vec x + b_{i}$
  - $a_{i} = \frac{e^{z_{i}}}{e^{z_{1}} + e^{z_{2}} + e^{z_{3}} + .... + e^{z_{n}}} = P(y = i \mid \vec x)$ with i is $i^{th}$ class, n is number of classes
  - loss function: $L = -\log(a_{i})$ if $y = i$

```
model.compile(loss = SparseCategoricalCrossentropy())
```
