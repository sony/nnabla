# Word2Vec Examples

---

## Overview

The example here written in python is a script that demonstrates word embedding leanings on Penn Treebank dataset.
Penn Treebank dataset is cached from the download page by running the example script.

---

## word_to_vec.py

This example demonstrates the training of the word embedding of word2vec on Penn Treebank dataset.
The implementation here referred from the paper "Distributed Representations of Words and Phrases and their Compositionality"
https://arxiv.org/abs/1310.4546
The Embedding Neural Network takes pairs of word indices as inputs, and outputs the embedding feature vectors.

You can start learning by the following command.

```
python word_to_vec.py [-c cuda.cudnn]
```

After the learning completes successfully, the output shows some of similar word search demonstrations 
that the network predicts the similar words of "Monday".

The learning results will be saved in "tmp.montors.w2v".
In this folder you could find model file "model.h5" and log files "*.txt".
