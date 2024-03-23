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
