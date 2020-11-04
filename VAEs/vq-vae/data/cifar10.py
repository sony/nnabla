import os
import sys
cifar_data_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..','..', '..', 'cifar10-100-collection'))
sys.path.append(cifar_data_path)

import numpy as np
from cifar10_data import Cifar10DataSource
from nnabla.utils.data_iterator import data_iterator

def data_iterator_cifar10(config, comm, train=True):
    '''
    Provide DataIterator with :py:class:`Cifar10DataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`Cifar10DataSource` is able to store all data into memory.

    '''
    data_iterator_ = data_iterator(Cifar10DataSource(train=train, shuffle=config['dataset']['shuffle']), config['train']['batch_size'], 
        with_memory_cache = config['dataset']['with_memory_cache'],
        with_file_cache = config['dataset']['with_file_cache'])
    if comm.n_procs > 1:
        data_iterator_ = data_iterator_.slice(rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    return data_iterator_