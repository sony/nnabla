# Data Parallel Distributed Training Example using CIFAR-10

---

## Overview

The examples listed below demonstrate several deep learning algorithms on CIFAR-10 dataset, which is one of the most popular image classification datasets in the machine learning community. The CIFAR-10 dataset will be automatically downloaded when running any of the examples.

---

## Multi-Device Multi-Process Training

This example shows the naive `Data Parallel Distributed Training` for the object regognition task using CIFAR-10 dataset and 23-layers ResNet with [NCCL](https://github.com/NVIDIA/nccl) using `multi-process` in a single node. 

NOTE that if you would like to run this example, please follow the bulid instruction to enable the multi-device training and make sure to prepare environment where you can use multiple GPUs. 

When you run the script like the following, 

```
mpirun -n 4 python multi_device_multi_process_classification.py --context "cudnn" -b 64

```

you can execute the training of 23-layers ResNet in the `Data Parallel Distributed Training` manner with the batch size being 64 and 4 GPUs.

## Multi-Node Training

This example shows the naive `Data Parallel Distributed Training` for the object regognition task using CIFAR-10 dataset and 23-layers ResNet with [NCCL](https://github.com/NVIDIA/nccl) using `multi-process` over multiple nodes. 

NOTE that if you would like to run this example, please follow the bulid instruction to enable the multi-device training and make sure to prepare environment where you can use multiple GPUs over multiple nodes.

When you run the script like the following, 

```
mpirun --hostfile hostfile python multi_device_multi_process_classification.py --context "cudnn" -b 64

```

you can execute the training of 23-layers ResNet in the `Data Parallel Distributed Training` manner with the batch size being 64 and N-GPUs and M-Nodes specified by the hostfile.

## Overlapping All-Reduce with Backward

This example shows the naive `Data Parallel Distributed Training` for the object regognition task using CIFAR-10 dataset and 23-layers ResNet with [NCCL](https://github.com/NVIDIA/nccl) using `all_reduce_callback` API.
All-reduce introduces communication overhead because it requires inter-process and inter-node communications. The `all_reduce_callback` API overlaps these all-reduce communications with backward computation in order to decrease the execution time.

Warning: This API does not support shared parameters currently. Thus, you cannot use this API when you train RNN.

NOTE that if you would like to run this example, please follow the bulid instruction to enable the multi-device training and make sure to prepare environment where you can use multiple GPUs over multiple nodes.

When you run the script like the following,

```
mpirun --hostfile hostfile python multi_device_multi_process_classification.py --context "cudnn" -b 64 --with-all-reduce-callback

```

you can execute the training of 23-layers ResNet in the `Data Parallel Distributed Training` manner with the batch size being 64. And the all-reduce is pipelined with backward computation.
