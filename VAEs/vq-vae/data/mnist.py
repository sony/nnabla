import os
import sys
import numpy as np

mnist_data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mnist-collection'))
sys.path.append(mnist_data_path)
from mnist_data import data_iterator_mnist

def mnist_iterator(config, comm, train=True):

    data_iterator_= data_iterator_mnist(batch_size=config['train']['batch_size'], train=train, rng=np.random.RandomState(config['model']['rng']), 
                                      with_memory_cache=config['dataset']['with_memory_cache'], with_file_cache=config['dataset']['with_file_cache'])
    if comm.n_procs > 1:
        data_iterator_ = data_iterator_.slice(rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    return data_iterator_
