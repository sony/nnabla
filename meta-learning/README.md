# Few-shot learning examples

---

## Overview

This example contains several meta-learning methods for few-shot learning of hand-written characters on Omniglot dataset.
Currently, the following methods are implemented.
- Metric based meta-learning
    - Prototypical networks: https://arxiv.org/abs/1703.05175
    - Matching networks: https://arxiv.org/abs/1606.04080
- Model Agnostic Meta-Learning (MAML): http://proceedings.mlr.press/v70/finn17a/finn17a.pdf

---

## Start training

Prepare omniglot dataset by

```
python omniglot_data.py
```

This script downloads the dataset into `~/nnabla_data`.
Also it generates compressed dataset files `*.npy` in `./omniglot/data`.
Extracting the dataset can take a while (around 1, 2 minutes).

Once the dataset is ready, you can start training.
When you want to conduct MAML, you can start it by

```
python train_maml.py
```

By default, the script will be executed with GPU.
If you prefer to run with CPU,

```
python train_maml.py -c cpu
```

In case of Prototypical networks or Matching networks, you can use

```
python metric_based_meta_learning.py
```


## Model Agnostic Meta Learning

In MAML, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task.
To achieve this, the algorithm contains a nested loop:
- inner loop: Given an episode, the current initial model is temporarily trained with it to obtain a meta-gradient
- outer loop: Given a meta-batch (batch of episodes), the current initial model is updated by the accumulated meta-gradient

Important options are as follows:

| Options | |
|:---|:---|
| --num_classes | A number of classes in meta-learning. When it is set to N, this setting is called ``N-way"|
| --num_shots | A number of training images per class in meta-learning. When it is set to k, this setting is called ``k-shots" |
| --num_updates | A number of iterations of the inner loop while meta-learning. Due to the limitation of nn.grad, it should be set to 1 in the currect version of nnabla (nnabla v1.3.0) |
| --test_num_updates | A number of iterations of the inner loop while meta-test. It can be larger than 1 |
| --metatrain_iterations | A number of iterations of the outer loop |
| --update_lr | A learning rate for the inner loop |
| --meta_lr | A learning rate for the outer loop |
| --meta_batch_size | A number of episodes per the iteration of the outer loop |
| --logdir | A directory for saving logs |

The default setting is 5-way 1-shot learning without first-order approximation for Omniglot dataset, which is the same with the setting shown in the paper.

| | 5-way 1-shot test accuracy |
|:---|:---|
| Reported in the original paper | 98.7% |
| Reproduced by this implementation | 98.3% |

## Metric based meta learning

The script `metric_based_meta_learning.py` can demonstrate Matching networks and Prototypical networks.
The default setting is a prototypical network with euclid distance of 20-way, 1-shot and 5-query setting.
We have many options to change parameters including network structures.
The following is an example of setting hyperparameters with corresponding options.

```
python metric_based_meta_learning.py -nw 20 -ns 1 -nq 5 -nwt 20 -nst 1 -nqt 5
```

Example of options are as follows.

| Options | |
| :--- | :--- |
| -nw |	Number of ways in meta-test, typically 5 or 20 |
| -ns |	Number of shots per class in meta-test, typically 1 or 5 |
| -nq |	Number of queries per class in meta-test, typically 5 |
| -nwt|	Number of ways in meta-training, typically 60, or same as meta-test |
| -nst|	Number of shots per class in meta-training, typically same as meta-test |
| -nqt|	Number of queries per class in meta-test, typically same as meta-test |
| -d  | Similarity metric, you can select "cosine" or "euclid" |
| -n  | Network type, you can select "matching" and "prototypical" |

### Prototypical networks
The default setting of this script is a prototypical network with euclid distance.
The embedding architecture follows the typical network with 4 convolutions written in papers.
To avoid all zero output from the embedding network, we omitted the last relu activation.
You can refer the paper in the following site.
https://arxiv.org/abs/1703.05175
Following the recommendation in this paper, we adopted 60-way episodes for training instead of 1 or 5-way.

### Matching networks
You can also select matching networks by setting -n option to matching.
However, since we are interested in the aspect of the metric learning,
we implemented only the softmax attention part which works as soft nearest neighbor search.
You can refer the paper in the following site.
https://arxiv.org/abs/1606.04080
We omitted the full context embedding in this paper, which uses the context by using a LSTM module.

---



---

## License

See `LICENSE`.
