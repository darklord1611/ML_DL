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
