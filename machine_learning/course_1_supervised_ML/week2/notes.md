### Multiple Linear Regression

- multiple features
  - $x_{j}$ = $j^{th}$ feature
  - n = number of features
  - $\vec x^{i}$ = features of $i^{th}$ training example, example: $\vec v$ = [1 2 3 4]
  - $x_{j}^{i}$ = value of feature $j$ in $i^{th}$ training example
- model with multiple features:
  - $f_{w,b}(x)$ = $w_{1}x_{1}$ + ..... + $w_{n}x_{n}$ + b
  - $\vec w$ = [$w_{1}$ ... $w_{n}$], b is a number, $\vec x$ = [$x_{1}$ ... $x_{n}$]
  - model can be rewritten: $f_{\vec w, b}(\vec x)$ = $\vec w \cdot \vec x$ + b -> multiple linear regression

### Vectorization

- calculate the sum of products using iterative way? inefficient -> using vectorization
- why vectorization? shorter code, faster, **numpy** utilize computer hardware(GPU) to execute parallel computation
  - f = np.dot(w,x) + b

### Gradient Descent for multiple LR

- update rule:
  - $w_{1}$ = $w_{1} - \alpha \frac{1}{m}(f_{\vec w, b}(\vec x^{i}) - y^{i})x_{1}^{i}$
  -
  - $b$ = $b - \alpha \frac{1}{m}(f_{\vec w, b}(\vec x^{i}) - y^{i})$
- calculate w, b without iterations? use normal equation(only use for LR)
  - this algorithm doesn't generalize to other learning algorithms
  - slow when number of features is large(> 10000)

### Feature scaling

- large difference between weights of features -> slower gradient descent -> how to solve? scale features in an appropriate way
- how to scale?
  - diving by the maximum -> scale back to 0 <= x <= 1
  - mean normalization(normally between -1 and 1) -> $x_{1}$ = $\frac{x_{1} - \mu_{1}}{max - min}$
  - Z-score normalization -> $x_{1}$ = $\frac{x_{1} - \mu_{1}}{\sigma_{1}}$, learn more about standard deviation
- aim for -1 <= x <= 1 for each feature $x_{i}$