## Machine Learning

- What is ML? a field of study that makes computer learn without being explicitly programmed
- ML is a sub-field of AI
- application? self-driving cars, payment fraud, healthcare, speech recognition, .....
- ultimate purpose? AGI(Artificial General Intelligence) perfoms at a human-like level
  - How? through learning algorithms
- ML algorithms
  - Supervised learning: widely used in real-life applications
  - Unsupervised learning
  - Recommender systems
  - Reinforcement learning
- Course also focus on practical advice on how to apply, when to apply different algorithms

---

### Supervised Learning

- map input x -> output label y
- learning algorithms include a set of examples + with the **right answer**
  - input: email, output: spam? -> spam filtering
  - input: English, output: Spanish -> machine translation
  - input: advertisements, user info, output: click? -> online advertising
- after training, take completely new input x -> produce y
- given a set of data points, figure out the appropriate line, curve, ... to fit most of them
- types of algorithms:
  - regression: predicting a real number(given the daily market data, predict the Bitcoin price)
  - classification: predicting a discrete number(class/category) (given the weight and height of a football player, determine his/her position)
- boundary lines: lines that separate alike groups of data points

---

### Unsupervised Learning

- find interesting pattern, structure in unlabeled data sets without preprocessed labels
- types of UL:
  - Clustering:
    - Example: Google News use clustering algorithms to group related news stories
    - group related things into clusters
  - Anomaly detection:
    - find unusual data points
  - Example: payment fraud, error detection, ...
    - Dimensionality reduction:
  - compress data using fewer numbers(features) but retain almost all info if possible, avoid overfitting
- hyper-parameters: values remain unchange before configuring and throughout training
  - Example: train-test split ratio, number of hidden layers in a NN, learning rate(gradient descent)
- parameters: values will change throughout training
  - Example: weights of features in linear or logistic regression models, centroids in clustering problems

---

### Linear Regression

- basically find a straight line that fit most of the data points
- regression models predict a number as the output, linear regression is a type of regression models
- classification models predict categories
- Useful terminology:
  - data used to train the model -> training set
  - input variable(feature) -> x
  - output variable(target) -> y
  - an example -> (x,y)
  - total number of examples -> m
  - $i^{(th)}$ training example -> ($x^i$, $y^i$)
- process:
  - feed the training set to learning alogrithms -> a function $f$(or a model)
  - $f$ will take the new input x -> produce a prediction(estimated) y
  - stick with $f_{w,b}$(x) = wx + b (w is weight, b is bias)
- cost function is next

---

### Cost function

- cost function measures how well your model fits the training data
- w, b are parameters aka variables(coefficients/weights) that can be adjusted during training to improve the model
- $\hat{y}$ = $f_{w,b}$($x^i$) = w$x^i$ + b is the predicted output
- $\sum_{i=1}^{m} (\hat{y}^{i} - y^{i})^{2}$ is the error
- what if training size get bigger? then the above expression grows as well -> compute the average square error instead
- cost function $J_{w,b}$ = $\frac{1}{2m}$ $\sum_{i=1}^{m}(\hat{y}^{i} - y^{i})^{2}$ (square error)
- ideally we want to minimize $J_{w,b}$
- $f_{w,b}$ is a function of x(with fixed w and given b == 0) while $J_{w,b}$ is a function of w(given b == 0)
- contour plots show 3D surface on a 2D plane

---

### Gradient descent

- minimize $J_{w,b}$ -> keep changing w, b to reduce until at or near minimum
- 3D plots will have hills aka maximum values, valleys aka minimum values
- gradient descent -> how to go from a hill(or any starting point) to a valley as efficiently as possible? -> spin around 360 degree, find the steepest direction -> go there -> repeat until convergence
- gradient descent not neccessarily direct to the global minimum
- formula: $w = w - \alpha\frac{\partial}{\partial w}$${J(w,b)}$, $b = b - \alpha\frac{\partial}{\partial b}$${J(w,b)}$
  - $\alpha$ is the learning rate(how big is a step? how agressive is the gradient descent?)
  - $\frac{\partial}{\partial w}$${J(w,b)}$ is the direction to which the model takes a step
  - update w and b simultaneously
- when do we stop? when w and b converge aka stop or change just a little compared to previous values
- intuition:
  - consider cost function $J$ with one variable w, derivative of $J$ is the tangent line
  - consider points from each side of the parabol, what does the update look like?
- what if $\alpha$ too small? too large? (divergent, overshooting, slow)
- able to reach local minimum with fixed $\alpha$ (as derivative becomes smaller)
- apply to the square error function:
  - $\frac{\partial}{\partial w}$${J(w,b)}$ = $\frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(x^{i}) - y^{i})x^{i}$
  -
  - $\frac{\partial}{\partial b}$${J(w,b)}$ = $\frac{1}{m}\sum_{i=1}^{m} (f_{w,b}(x^{i}) - y^{i})$
- square error cost function is a convex function aka function with one single, unique global minimum
- batch gradient descent: at each step of GD, look at all the training examples
