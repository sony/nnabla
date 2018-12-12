# Efficient Neural Architecture Search

## Overview

Reproduction of the work, "Efficient Neural Architecture Search via Parameter Sharing" by NNabla. 
We offer 2 methods proposed by the paper above, Macro Search and Micro Search.

We strongly recommend you to run this code with a decent GPU (at least, NVIDIA GeForce GTX 1080 Ti or better).

### Dataset

By default, this example uses CIFAR-10 dataset, and the dataset will be automatically downloaded when you run the script.


### Configuration

In `args.py`, you can find configurations for both architecture search and evaluation. 


### Architecture Search

Process of architecture search can be done by a command below,

```python
python macro_search.py --device-id 0 --context 'cudnn' \
                           --monitor-path 'search.monitor' \
                           --recommended-arch <filename-you-want>
```

If you want to use micro search instead, just replace `macro_search.py` by `micro_search.py`.

It takes about 12 hours using a single Tesla P40. Also, With `--early-stop-over` option you can finish the search early (It terminates the search process once the validation accuracy surpasses the one you set, e.g, 0.80).
After the architecture search finishes, you will find s `.npy` file which contains the model architecture.
You can give it an arbitrary name by adding `--recommended-arch <filename-you-want>` option to `macro_search.py` or `micro_search.py`,
by default, its name is either `macro_arch.npy` or `micro_arch.npy`.

Note that this file does not contain weights parameters. It only has the list(s) which represents the model architecture.
During the architecture search, no weights parameters are stored.
However, as a side effect of displaying intermediate training results, the latest records of CNN training, for instance, `Training-loss.series.txt`, are stored in a directory set by `--monitor-path`.


### Architecture Evaluation

For re-training the model recommended as a result of architecture search, just run

```python
python macro_retrain.py --device-id 0 --context 'cudnn' \
                            --recommended-arch <path to npy file> \
                            --monitor-path 'result.monitor' \
                            --monitor-save-path 'result.monitor' 
```

This time the weights parameters are stored in `--monitor-save-path` along with other training records in `--monitor-path`.


### Architecture Derivation
Besides the architecture recommended by the controller during the search process, other architectures can be sampled by the controller. After the architecture search finishes, you also have `controller_params.h5` in the directory set by `--monitor-path`.
This contains controller's parameters which generated a architecture with the best validation accuracy. You can get other architectures by using the same script with `--sampling-only` option. If you want 5 more architectures, simply run 

```python
python macro_search.py --sampling-only True \
                           --num-sampling 5 
```

Now you have 5 `sampled_macro_arch_N.npy"`.  These newly sampled architectures can be trained by the evaluation process described above.


## NOTE
- Currently, we observe that the final accuracy when training after architecture search finishes (in short, when training the model recommended by the controller as a result of architecture search) does not reach as high as reported in the paper, however, that is very close to the result obtained by the author's code publicly available on Github. Also, we don't apply Cutout to the input images.


## References
- Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean, "Efficient Neural Architecture Search via Parameter Sharing", arXiv:1802.03268
- https://github.com/melodyguan/enas
