## Classification

- binary classification: y can only be one of two values, 0 -> false(absence), 1 -> true(presence)
- intuition: use linear regression, fit a straight line, declare a threshold(decision boundary) -> separate into two groups

### Logistic Regression

- sigmoid(logistic) function:
  - output between 0 and 1
  - $g(z) = \frac{1}{1 + e^{-z}}$, $0 <= g(z) <= 1$
  - $z = \vec w \cdot \vec x + b$
- logistic regression: $f\_{\vec w,b}(\vec x) = g(\vec w \cdot \vec x + b) = \frac{1}{1 + e^{-(\vec w \cdot \vec x + b)}}$
- interpret as the probability of specific class is 1
- $f\_{\vec w, b}(x) = P(y = 1 |\vec x; \vec w, b)$
- define a threshold like 0.5, $\hat y >= 0.5$ when $z >= 0$
- **decision boundary** $z = \vec w \cdot \vec x + b = 0$ is a line
- what about non-linear boundaries?
  - Example: $f_{\vec w, b}(x) = g(z) = g(w_{1}x_{1}^2 + w_{2}x_{2}^2 + b)$

### Cost function for Logistic Regression

- recall SEC(square error cost) function for linear regression: $J(\vec w, b) = \frac{1}{m}\sum_{i=1}^{m} \frac{1}{2} (f_{\vec w, b}(\vec x^{i}) - y^{i})^{2}$
- with linear regression, $J$ is a convex function aka function with one global minima
- with logistic regression, $J$ is a non-convex -> multiple local minima -> not efficient for gradient descent
- logistic loss function: $L(f_{\vec w, b}(\vec x^{i}), y^{i})$
  - $L$ = $-\log(f_{\vec w,b}(\vec x^{i}))$ if $y^{i} = 1$
  - $L$ = $-\log(1 - f_{\vec w,b}(\vec x^{i}))$ if $y^{i} = 0$
- what are the intuition? try to sketch the graph of $y = -\log(x)$ and $y = \log(x)$, $0 <= x <= 1$, $y = 0 \mid y = 1$ and compare value of loss function with different x
- conclusion:

  - loss is lowest when prediction $f_{\vec w,b}(\vec x^{i})$ is close to true label $y^{i}$
  - loss is highest when prediction $f_{\vec w,b}(\vec x^{i})$ is far away from true label $y^{i}$
  - loss function is convex function -> can reach a global minimum

- **cost function**: $J(\vec w, b) = \frac{1}{m}\sum_{i=1}^{m}L(f_{\vec w, b}(\vec x^{i}), y^{i})$
- simplified version of loss function:
  - $L(f\_{\vec w, b}(\vec x^{i}), y^{i}) = -y^{i}\log(f_{\vec w,b}(\vec x^{i})) - (1 - y^{i})\log(1 - f_{\vec w,b}(\vec x^{i}))$
- simplified version of cost function:
  - $J(\vec w, b) = -\frac{1}{m}\sum_{i=1}^{m}(y^{i}\log(f_{\vec w,b}(\vec x^{i})) + (1 - y^{i})\log(1 - f_{\vec w,b}(\vec x^{i})))$
- intuition? derived from statistics, more specifically, maximum likelihood estimation -> find parameters of a logistic model

- gradient descent update rules:
  - $w_{j} = w_{j} -\frac{\alpha}{m}\sum_{i=1}{m}(f_{\vec w,b}(\vec x^{i}) - y^{i})x_{j}^{i}$
  -
  - $b = b -\frac{\alpha}{m}\sum_{i=1}{m}(f_{\vec w,b}(\vec x^{i}) - y^{i})$
  - With $f_{\vec w,b} = \frac{1}{1 + e^{-(\vec w \cdot \vec x + b)}}$
- same concepts: learning curve, vectorized implementation, feature scaling

### Overfitting

- not fitting the training data well? -> underfitting(high bias) (too few features)
  - bias? happen when the model fail to capture a pattern in the data, or has a strong but wrong preconception
- fitting the data set well? -> "normal-fitting" -> good generalization aka make predictions based on new inputs
- fitting the data set extremely well? -> overfitting(high variance) -> not generalize well(too many features)
- how to address?
  - get more training examples
  - selected fewer features -> problem? useful features may be left out
  - regularization -> reduce size of some parameters

### Regularized regression

- suppose linear regression with 4 parameters $w_{1}, w_{2}, w_{3}, w_{4}$
- modify cost function to be: $f_{\vec w,b}(x) := f_{\vec w,b}(x) + 1000w_{3}^{2} + 1000w_{4}^{2}$
  - high cost if $w_{3}$ and $w_{4}$ is large
  - solution? force $w_{3}$ and $w_{4}$ to be small(cancel out effect of that particular feature) -> regularized $w_{3} and w_{4}$
- how to know which features are important? include all then penalize by add to cost function:
  - $J(\vec w,b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\vec w,b}(\vec x^{i} - y^{i}))^{2} + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$
  - $n$ is the number of features, $\lambda$ is the regularization parameter ($\lambda$ > 0),
  - $\frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$ is the regularization term
  - why divide by $2m$ -> to scale better with additional training data, also easier to choose $\lambda$
- the new cost function try to minimize SEC to fit data, also minimize regularization term to reduce overfitting
- how? by choosing $\lambda$ to balance between two goals

  - what happen if we choose $\lambda = 0$? $\lambda = 10^{10}$ or any large number? Over or under fitting?

- regularized linear regression update rules:
- - $w_{j} = w_{j} - \alpha[\frac{1}{m}\sum_{i=1}^{m}(f_{\vec w,b}(\vec x^{i} - y^{i}))x_{j}^{i} + \frac{\lambda}{m}w_{j}]$
  - $b = b - \alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec w,b}(\vec x^{i} - y^{i})$
- intuition?

  - update rule for $w_{j}$ can be rewritten: $w*{j} = w*{j}(1 - \alpha\frac{\lambda}{m}) - \alpha[\frac{1}{m}\sum_{i=1}^{m}(f_{\vec w,b}(\vec x^{i} - y^{i}))x_{j}^{i}]$
  - note that $\alpha$ is really small -> $1 - \alpha\frac{\lambda}{m}$ close to 0.9998 -> after each iteration we shrink $w_{j}$ just a little bit, hence reduce

- regularized logistic regression:
  - cost function: $J(\vec w,b) := J(\vec w,b) + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$
  - update rules are the same as linear regression
