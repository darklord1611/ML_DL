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

- numerical rounding errors with logistic regression:
  - Original loss: $L = -y\log(a\_{1}) + (1 - y)\log(1 - a)$
  - More accurate loss: $L = -y\log(\frac{1}{1 + e^{-z}}) + (1 - y)\log(1 - \frac{1}{1 + e^{-z}})$
  - add from_logrits=True to model.compile()
  - results now are $z_{1}, .... z_{n}$ instead of probabilities -> apply softmax() or sigmoid() accordingly
- multi-label classification -> an example has multiple output labels
- one NN for each of the label? not efficient -> one NN with 3 outputs -> sigmoid at output layer

### Additional Concepts

- Adaptive Moment estimation(ADAM) optimize gradient descent by increase/decrease learning rate $\alpha$ accordingly
  - increase if parameters going the same direction after update
  - decrease if parameters oscillating after update
- different types of layers:
  - Dense: each neuron output is a function of all activation outputs of the previous layer
  - Convolutional: each neuron only uses part of previous layer's outputs -> faster computation, less training data

## Temporary

### Bias - Variance

- $J_{train}$ is high -> indicator of high bias(underfit)
- $J_{train}$ is low, $J_{cv}$ is high -> high variance(overfit)
- $J_{train}$ is low, $J_{cv}$ is low -> ideal situation
- degree of the polynomial function increase -> $J_{train}$ is likely to decrease
- choose the regularization parameter $\lambda$ -> trial and error, compare different $J_{cv}$ -> pick the lowest
  - $J_{train}$ will increase as $\lambda$ increase, $J_{cv}$ will be a parabol
- how to evaluate? compare to a human level of performance, competing algorithms performance, guessing
- adding more data won't help with high bias, may be with high variance
- how to visualize? learning curve
- overfit -> pay too much attention to fit the training set not to generalize with new examples
- how to improve?
  - high bias: try adding more features, try adding more polynomial features, try decreasing $\lambda$
  - high variance: getting more training examples, try smaller sets of features, try increasing $\lambda$

### Bias-Variance with NN

- bias-variance tradeoff:
  - simple model -> high bias
  - complex model -> high variance
- NN large enough? almost always fit training set well
- $J_{train}$ is high -> high bias -> use bigger network aka more hidden layers or more units per layer
- $J_{cv}$ is high -> high variance -> get more data -> evaluate $J_{train}$ -> repeat
- larger NN almost always better with appropriate regularization terms but computationally expensive

```
# Regularized layer
layer = Dense(units=25, activation="relu", kernel_regularizer=L2(0.01))
```

### ML development cycle

- overall architecture(model, data, etc...)
- train the model
- diagnostics(bias, variance, error analysis)
