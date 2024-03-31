## Unsupervised Learning

- only input x, not target y -> find something interesting about the data
- $\mu_{i}$ denote $i^{th}$ cluster centroid
- k-means algorithm:

```
randomly intilize K cluster centroids
repeat until converge:
    assign each point to the closest centroid
    recompute the centroids by reassign to the average of all points in that particular centroid
```

- $c^{i}$ represents index of clusters(1, 2, ..., K) to which example $x^{i}$ is assigned currently
- $\mu_{c^{i}}$ is the cluster centroid to which example $x^{i}$ has been assigned
- cost(distortion) function: $J(c^{1}, ...., c^{m}, \mu_{1}, ...., \mu_{k}) = \frac{1}{m}\sum_{i=1}^{m}||(x^{i} - \mu_{c^{i}})||^{2}$
- K < m
- random intialization:
  - repeat arbitrary times like 100
  - pick K examples to be intial centroids
  - run k-means algorithm until convergence
  - compute cost function $J$
  - select the set of clusters that give lowest cost
- how to choose K?
  - elbow method -> just try bunch of values for K, compute cost -> select the value with remarkable difference(not the value minimize cost because k can be incredibly large)

### Anomaly Detection

- spot unusual data point
- density estimation: train dataset -> calculate probability of certain data point
  - $p(x_{test}) < \epsilon$ -> could be an anomaly
- example: fraud detection

  - $x^{i}$ = features of user $i$'s activities
  - model $p(x)$ from the data
  - check if $p(x_{test}) < \epsilon$
  - perform additional checks

- Gaussian distribution with mean $\mu$, variance(or standard deviation) $\sigma^{2}$

  - $p(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x - \mu)^{2}}{2\sigma^{2}}}$
  -
  - $\mu = \frac{1}{m}\sum_{i=1}^{m}x^{i}$
  -
  - $\sigma^{2} = \frac{1}{m}\sum_{i=1}^{m}(x^{i} - \mu)^{2}$
  - above is the maximum likelihood for $\mu, \sigma$

- what if we have n features? -> $p(\vec x) = p(x_{1}, \mu_{1}, \sigma_{1}^{2}) * p(x_{2}, ...) * ..... * p(x_{n}, ...)$
- real-number evaluation: number to evalute the learning algorithm and related changes to parameters
- when to use anomaly detection?
  - small number of positive examples
  - large number of negative examples
  - many different types of anomalies, future anomalies may not look similar to previous ones -> think of an aircraft engine's anomalies
  - supervised learning make assumptions that future anomalies will look similar to previously trained examples
- how to choose features?
  - for non-Gaussian features -> normalize to Gaussian distribution by multiply or something else
- problems?
  - for some cases, $p(x)$ is comparable both to normal and anomalous examples

### Recommender System

- notations:

  - $r(i,j)$ = 1 if user $j$ has rated movie $i$
  - $y^{(i,j)} is the rating given by user $i$ on movie $j$
  - $w^{j}, b^{j}$ parameters for user $j$
  - $x^{i}$ features vector for movie $i$
  - $m^{j}$ number of movies rated by user $j$
  - for user $j$ and movie $i$, predict rating: $w^{j} \cdot x^{i} + b$
  - cost function: $L = \frac{1}{2}\sum_{i:r(i,j)=1}(w^{j} \cdot x^{i} + b - y^{(i,j)})^{2}$
  - add regularization term, sum of all cost for all users

- how to come up with features?
- collaborative filtering -> only possible if we have inputs from different sources about the same thing -> cost function now have 3 parameters w, b, x -> linear regression
- binary labels? -> logistic regression
- problems? users not rated any movies are likely to be predicted to rate only 0 stars -> use mean normalization
  - subtract the mean of each movie's ratings(rows) and add back once figure out parameters

```
w = tf.Variables(3.0)
x = 1.0
y = 1.0
alpha = 0.1

iterations = 30

for iter in range(iterations):
    with tf.GradientTape() as tape:
        fwb = w * x
        costJ = (fwb - y) ** 2
    [dJdw] = tape.gradient(costJ, [w])

    w.assign_add(-alpha * dJdw)
```

- how to interpret features to find related items?
  - find item with similar feature vector
- limitations of collaborative filtering:

  - cold start problem
  - unable to use side information

- collaborative filtering: based on ratings of users who gave similar ratings as you

### Content-based filtering

- content-based filtering: based on features of user and items to find a good match
  - user features: age, gender, country, ...
  - movie features: year, genre, country, ....
  - goal is compute a vector represent user's preferences vs a vector represent a movie's features
- compute user network and movie network -> combine both and define a cost function

- how to find recommendation from a large set of items? retrieval & ranking
  - retrieval: generate a list of plausible candidate items
  - ranking: take precomputed list and rank them using trained model
