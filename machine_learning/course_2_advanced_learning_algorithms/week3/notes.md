## ML Advices

- how to evaluate a model? split data into training set vs test set
- regression:

  - compute test error: $J_{test}(\vec w,b) = \frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(f_{\vec w,b}(\vec x_{test}^{i}) - y_{test}^{i})^{2}$
  - compute training error: $J_{train}(\vec w,b) = \frac{1}{2m_{train}}\sum_{i=1}^{m_{train}}(f_{\vec w,b}(\vec x_{train}^{i}) - y_{train}^{i})^{2}$

- classification:

  - $J_{test}(\vec w,b)$ is the fraction of test examples that have been misclassified
  - $J_{train}(\vec w,b)$ is the fraction of train examples that have been misclassified

- once parameters are fit to the training set, $J_{train}(\vec w,b)$ -> not a good indicator -> use $J_{test}(\vec w,b)$
- model selection -> linear or polynomial(if this then what d = ?) -> might be flawed as we used an extra parameter d on the test set
- another way to split data: training - cross validation - test
  - calculate error: same as linear regression above
  - choose the polynomial function has lowest CV error -> feed to the test set -> output the generalization error

### Error Analysis

- try to figure out the pattern of misclassified data -> prioritized according to size of the portion -> resolve by collecting more related data or add more features
- disadvantage? generally good at something a human can do(identify spam emails but not predict which ads users will click on)
- data augmentation? create new similar training examples by modifying existing examples -> how? by distortion or transformation
  - example? speech recognition + noisy background -> new examples
- data synthesis? create brand new examples from scratch
- not much data + really hard to get more data? -> **transfer learning**

### Transfer learning

- train NN for an unrelated task -> get parameters values -> modify and apply to current task -> transfer learning
- how exactly?
  - Option 1: keep all parameters of hidden layers fixed, only update the output layer -> works well with really small training set
  - Option 2: relatively large training set? -> hidden layers initialized with trained parameters values, update all layers as usual
- 1st step is training on a large dataset, then tune the parameters further in a small dataset -> supervised pre-training
- 2nd step is modifying the weights to fit to the current task -> fine-tuning
- why this work?
  - first layer detects edges -> corners -> complex curves or shapes
- full cycle of ML development:

  - scoping
  - data collection
  - train model
  - evaluate -> not sufficient? -> fine-tuning techniques -> repeat data collection
  - deploy/monitor

- accuracy not always a good indicator -> use precision/recall instead
- confusion matrix consists of true(false) positive(negative)
  - precision: $\frac{true \space postives}{true \space positives \space + \space false \space positives}$
  - recall: $\frac{true \space postives}{true \space positives \space + \space false \space negatives}$
- high threshold -> high precision, low recall and vice versa
- F1 score aka harmonic mean combine both precision and recall: $F = \frac{2PR}{P \space + \space R}$
