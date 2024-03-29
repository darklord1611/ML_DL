## Decision Trees

- decision nodes -> internal nodes that will make a decision based on value of a particular feature
- leaves -> binary number
- how to choose which feature to split at each node? maximize purity(number of correct examples per desired class)
- when do we stop splitting?

  - a node is 100% of that class
  - exceed the maximum depth(can be passed as a parameter)
  - when purity increases just a negligible amount
  - number of examples of a node below a certain threshold

- how to measure purity? -> entropy
- binary classification: $p_{0}$, $p_{1} = 1 - p_{0}$
  - $H(p_{0}) = -p_{1}\log(p1) - p_{0}\log(p_{0}) = -(1 - p_{1})\log(1 - p_{1}) - p_{0}\log(p_{0})$
- let $p_{1}$ denote is a cat, we split on an arbitrary feature, got $p_{1} = \frac{4}{5}$ on the left sub-branch and $p\_{1} = \frac{1}{5}$ on the right sub-branch
  - how to choose the best one? reduction entropy
  - information gain = $H(p_{1}^{root}) - (\frac{left \space examples}{total \space examples}H(p_{1}^{left}) + \frac{right \space examples}{total \space examples}H(p_{1}^{right}))$ = $H(p_{1}^{root}) - (w^{left}H(p_{1}^{left}) + w^{right}H(p_{1}^{right}))$

### Decision Tree Learning

- start with all examples at the root
- calculate information gain for all features, pick the one with the highest reduction entropy(information gain)
- split dataset according to selected features, create left and right branches of the tree recursively
- keep splitting until one of these conditions is met:
  - a node is 100% class -> entropy = 0
  - splitting will result in a tree exceeding the maximum depth
  - information gain is less than a threshold
  - number of examples in a node are less than a threshold
- maximum depth is just like highest polynomial degree in or number of layers in a NN
- what if a feature takes on a discrete range of values -> use one-hot encoding -> transfer into yes/no questions -> binary features
- what about continuous features? trial and error, split based on a specific value -> calculate information gain -> pick the best

### Regression Tree

- ok if we split the same feature on the left and right branches
