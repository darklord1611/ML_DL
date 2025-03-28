# Word Vectors & Language Models

## Gradient Descent

- SGD is faster than original GD, introduce a little noise when updating weights using SGD may benefit the neural network learning.
  
## Word Sense

- A word may have different senses(specific meaning in a given context) -> How to represent? A word vector for each sense? -> Impractical.
- Different senses of words are baked into one single word vector (like in word2vec), represented as a weighted average sum.
- Individual sense vectors are reconstructible using sparse coding theory