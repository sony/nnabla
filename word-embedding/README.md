# Word Embedding Examples

---

## Overview

The example here written in python is a script that demonstrates word embedding leanings on Penn Treebank dataset.
Penn Treebank dataset is cached from the download page by running the example script.

---

## word_embedding.py

This example demonstrates the training of the word embedding on Penn Treebank dataset.
The Embedding Neural Network takes pairs of word indices as inputs, and outputs the embedding feature vectors.

You can start learning by the following command.

```shell
python word_embedding.py
```

By default, the script will be executed with GPU.
If you prefer to run with CPU, try

```shell
python train.py -c cpu
```

After the learning completes successfully, the output shows some of similar word search demonstrations 
that the network predicts the similar words of "Monday".

The learning results will be saved in "tmp.montors.w2v".
In this folder you could find model file "model.h5" and log files "*.txt".
