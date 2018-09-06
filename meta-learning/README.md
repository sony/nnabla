# One-shot and few-shot learning examples

---

## Overview

Thie example script implements one-shot and few-shot learning of hand-written characters on Omniglot dataset.
It learns to classify a query image of character to its correct class from support classes, upon seeing only a few sample images from each support classe.

---

## Start training

Prepare omniglot dataset by

```
python omniglot_data.py
```

This script downloads the dataset from `...` into `~/nnabla_data`.
Also it generates compressed dataset files `*.npy` in `./omniglot/data`.
Extracting the dataset can take a while (around 1, 2 minutes).

Once the dataset is ready, start training, such as metric based meta learning, by

```
python metric_based_meta_learning.py
```

The output of the training will be saved in `tmp.results` directory.
You can see here the "Training loss curve", "Validation error curve" and "Test error result".
Also you can see a t-SNE 2d-visualized distribution of test samples.

By default, the script will be executed with GPU. If you prefer to run with CPU,

```
python metric_based_meta_learning.py -c cpu
```

## Metric based meta learning

We classified some of the meta-learning method into "metric-based meta-learning".
Siamese networks for oneshot:https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
Matching networks: https://arxiv.org/abs/1606.04080
Prototypical networks: https://arxiv.org/abs/1703.05175

The script `metric_based_meta_learning.py` can demonstrate the matching networks and the prototypical networks.
The default setting is a prototypical network with euclid distance of 20-way, 1-shot and 5-query setting.
We have many options to change parameters including network structures.
The following is an example of setting hyperparameters with corresponding options.

```
python metric_based_meta_learning.py -nw 20 -ns 1 -nq 5 -nwt 20 -nst 1 -nqt 5
```

Example of options are as follows.
-nw :	Number of ways in meta-test, typically 5 or 20
-ns :	Number of shots per class in meta-test, typically 1 or 5
-nq :	Number of queries per class in meta-test, typically 5
-nwt:	Number of ways in meta-training, typically 60, or same as meta-test
-nst:	Number of shots per class in meta-training, typically same as meta-test
-nqt:	Number of queries per class in meta-test, typically same as meta-test
-d  :   Similarity metric, you can select "cosine" or "euclid".
-n  ;   Network tpye, you can select "matching" and "prototypical".

### Prototypical networks
The default setting of this script is a prototypical network with euclid distance.
The embedding architecture follows the typical network with 4 convolutions written in papers.
To avoid all zero output from the embedding network, we omitted the last relu activation.
You can refer the paper in the following site.
https://arxiv.org/abs/1703.05175
Following the recommendation in this paper, we adoptted 60-way episodes for training instead of 1 or 5-way.

### Matching networks
You can also select matching networks by setting -n option to matching.
However, since we are interested in the aspect of the metric learning,
we implemented only the softmax attention part which works as soft nearest neighber search.
You can refer the paper in the following site.
https://arxiv.org/abs/1606.04080
We omitted the full context embedding in this paper, which uses the context by using a LSTM module.
