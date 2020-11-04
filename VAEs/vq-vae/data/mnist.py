import os
import sys
mnist_data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mnist-collection'))
sys.path.append(mnist_data_path)

import numpy as np
from mnist_data import MnistDataSource
from nnabla.utils.data_iterator import data_iterator

def data_iterator_mnist(config, comm, train=True):

    data_iterator_ =  data_iterator(MnistDataSource(train=train, shuffle=config['dataset']['shuffle'], rng=np.random.RandomState(config['model']['rng'])),
                         config['train']['batch_size'],
                         rng=config['model']['rng'],
                         with_memory_cache=config['dataset']['with_memory_cache'],
                         with_file_cache=config['dataset']['with_file_cache'])
    if comm.n_procs > 1:
        data_iterator_ = data_iterator.slice(rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    return data_iterator_
