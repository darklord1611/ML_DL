# Word Vectors

- How do we represent the meaning of a word?
  - Use WordNet - a thesaurus of synonym sets and hypernyms(describes "is-a" relationship) -> what's wrong? not enough contexts, impossible to keep up-to-date, human labor
  - Statistical models represent words as one-hot vectors

## Word2Vec

- General Idea:
  - We have a large corpus of text -> long list of words
  - Every word in the fixed vocabulary is represented by a single vector
  - Go through each position $t$ in the text, which has a center word $c$ and context words $o$