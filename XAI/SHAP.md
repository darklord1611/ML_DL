# Shapley values

- method for assigning payouts to player depending on their contribution to the total payout in a **game**
  - **game** -> prediction task for a single instance,
  - **gain** -> difference between the actual prediction and the average predictions fo all instances
  - **players** ->
- Suppose we have 3 binary features A, B, C, a **coalition** is a combination of feature value, for example: (A, B, C) = (0, 1, 0) or (A, B, C) = (1, 1, 0)
- Shapley values -> average marginal contribution of a feature value across all possible coalitions
- Suppose we need to determine Shapley values of C = 1, for each combination of (A, B) -> calculate the difference between predictions with and without C = 1 -> take average -> Shapley values
- 4 properties: Efficiency, Symmetry, Dummy and Additivity

- An intuitive way to understand the Shapley value is the following illustration: The feature values enter a room in random order. All feature values in the room participate in the game (= contribute to the prediction). The Shapley value of a feature value is the average change in the prediction that the coalition already in the room receives when the feature value joins them.

- Advantages:
  - **gain** is guaranteed to be fairly distributed among feature values in constrast with LIME
  - solid theory, contrastive explanations
- Disadvantages:
  - computationally expensive (exponential runtime)
  - can be misinterpreted -> Shapley value of a feature is not the difference of predicted value after removing the feature. It is the contribution to the difference between actual and mean prediction, given a set of feature values.
  - always use full set of features
  - inclusion of unrealistic data instances when features are correlated
