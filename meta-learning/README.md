# Few-shot learning examples

---

## Overview

This example contains several meta-learning methods for few-shot learning.
Currently, the following methods are implemented.
- Model Agnostic Meta-Learning (MAML): http://proceedings.mlr.press/v70/finn17a/finn17a.pdf
- Metric based meta-learning
    - Prototypical networks: https://arxiv.org/abs/1703.05175
    - Matching networks: https://arxiv.org/abs/1606.04080

---

## Start training

Prepare omniglot dataset by

```
python omniglot_data.py
```

in "./data" directory.

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
python prototypical.py
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

## Metric Based Meta Learning

The script `prototypical.py` can demonstrate prototypical networks and matching networks.
The default setting is 5-way 1-shot 5-query learning of the prototypical network with Euclidean distance.
We have many options to change parameters including network structures.
The following is an example of the prototypical network with corresponding options.

```
python prototypical.py -nw 5 -ns 1 -nq 5 -nwt 5 -nst 1 -nqt 5
```

We can also demonstrate an example of the matching network.

```
python prototypical.py --net-type matching --metric cosine
```
If you want to try the prototypical networks on CelebA dataset,
first, you need to download "img_align_celeba.zip" and "identity_CelebA.txt" from the following website
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
and put them into "./data" directory.
Then, generate compressed dataset files `*.npy` to `./data/celeb_a/data` by executing

```
python celeb_a_data.py
```
at "./data" directory.
Finally, you can start training by

```
python prototypical.py --dataset celeb_a
```

Important options are as follows.

| Options | |
| :--- | :--- |
| --n_class   |	Number of ways in meta-test, typically 5 or 20 |
| --n_shot    | Number of shots per class in meta-test, typically 1 or 5 |
| --n_query   | Number of queries per class in meta-test, typically 5 |
| --n_class_tr	|      Number of ways in meta-training, typically 60, or same as meta-test |
| --n_shot_tr	|      Number of shots per class in meta-training, typically same as meta-test |
| --n_query_tr	|      Number of queries per class in meta-test, typically same as meta-test |
| --net_type	|      "prototypical" and "matching" are available |
| --metric	|      "euclid" and "cosine" are available |
| --max_iteration     |		      Maximum number of iterations |
| --dataset	      |		      "omniglot" and "celeb_a" are available after setup |

### Prototypical networks
The default setting of this script is a prototypical network with Euclidean distance.
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
